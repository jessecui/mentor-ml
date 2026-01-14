import os
import sys
import warnings
import platform

# --- CONFIGURATION (MUST BE BEFORE ANY JAX/TF IMPORTS) ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Platform-specific configuration
IS_MACOS = platform.system() == "Darwin"
USE_GPU = os.environ.get("USE_GPU", "").lower() == "true"

if IS_MACOS or not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    if IS_MACOS:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["GRPC_POLL_STRATEGY"] = "poll"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
else:
    os.environ["JAX_PLATFORMS"] = "cuda"

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", message="Protobuf gencode version")
warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "model" else SCRIPT_DIR
REPO_DIR = os.path.join(PROJECT_ROOT, "big_vision_repo")
if REPO_DIR not in sys.path:
    sys.path.append(REPO_DIR)

# Pre-read tokenizer bytes BEFORE any heavy imports (avoids mutex issues)
TOKENIZER_PATH = os.path.join(SCRIPT_DIR, "paligemma_tokenizer.model")
TOKENIZER_BYTES = None
if os.path.exists(TOKENIZER_PATH):
    with open(TOKENIZER_PATH, 'rb') as f:
        TOKENIZER_BYTES = f.read()

# 1. PREVENT TENSORFLOW / JAX CONFLICT
# Even if you don't use TF, big_vision imports it internally.
# We must configure it to be "invisible" so it doesn't crash JAX.
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
# This specific setting prevents the mutex lock error on macOS
if platform.system() == "Darwin":
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# 2. NOW IMPORT JAX
import jax
if IS_MACOS or not USE_GPU:
    jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
import numpy as np
from PIL import Image
import sentencepiece
import ml_collections

import big_vision.models.proj.paligemma.paligemma as paligemma_model
from big_vision.utils import recover_tree, recover_dtype

print("All imports done")

def load_params_macos_safe(npz_path, model_config):
    """Load checkpoint with macOS-safe settings to avoid mutex issues."""
    # Load npz directly (no mmap - that was causing issues)
    print("   Loading npz file...")
    with np.load(npz_path, allow_pickle=False) as npz:
        keys = list(npz.keys())
        values = [np.array(npz[k]) for k in keys]
    
    # Recover tree structure
    print("   Reconstructing parameter tree...")
    checkpoint = recover_tree(keys, values)
    
    # Recover dtype (bfloat16 is stored as raw bytes)
    checkpoint = jax.tree.map(recover_dtype, checkpoint)
    
    # Extract params (checkpoint may be wrapped)
    if "params" in checkpoint:
        params = checkpoint["params"]
    elif "opt" in checkpoint:
        params = checkpoint["opt"]["target"]
    else:
        params = checkpoint
    
    # Now load the submodels using big_vision's fixup logic
    import importlib
    restored_params = {"img": None, "llm": None}
    
    # Load image encoder params
    if "img" in params:
        vit_module = importlib.import_module("big_vision.models.vit")
        restored_params["img"] = vit_module.fix_old_checkpoints(params["img"])
        # Handle scan conversion if needed
        if model_config.img.get("scan") and "encoderblock" not in restored_params["img"].get("Transformer", {}):
            restored_params["img"] = vit_module.pyloop_to_scan(restored_params["img"])
    
    # Load LLM params  
    if "llm" in params:
        restored_params["llm"] = params["llm"]
    
    return restored_params

def make_score_fn(model):
    """Create a JIT-compiled ColPali scoring function for the given model."""
    
    @jax.jit
    def score_colpali(params, images, text_tokens, text_mask):
        """ColPali-style MaxSim scoring between image and text embeddings."""
        # Image embeddings: [Batch, NumPatches, Dim]
        zimg, _ = model.apply({'params': params}, images, train=False,
                              method=model.embed_image)
        
        # Text embeddings: [Batch, SeqLen, Dim]
        ztxt, _ = model.apply({'params': params}, text_tokens, train=False,
                              method=model.embed_text)
        
        # Normalize embeddings for cosine similarity
        zimg = zimg / (jnp.linalg.norm(zimg, axis=-1, keepdims=True) + 1e-8)
        ztxt = ztxt / (jnp.linalg.norm(ztxt, axis=-1, keepdims=True) + 1e-8)
        
        # ColPali MaxSim: for each text token, find max similarity across image patches
        sim_matrix = jnp.einsum('btd,bpd->btp', ztxt, zimg)
        max_scores = jnp.max(sim_matrix, axis=-1)
        
        return jnp.sum(max_scores * text_mask, axis=-1)
    
    return score_colpali

def main():
    print("⚙️  Initializing JAX Scorer...")
    
    # Check for Tokenizer
    tokenizer_path = os.path.join(SCRIPT_DIR, "paligemma_tokenizer.model")
    if not os.path.exists(tokenizer_path):
        print(f"❌ ERROR: Missing {tokenizer_path}")
        return

    import kagglehub
    
    KAGGLE_HANDLE = "google/paligemma-2/jax/paligemma2-3b-pt-224"
    print("⏳ Downloading checkpoint from Kaggle...")
    MODEL_DIR = kagglehub.model_download(KAGGLE_HANDLE)
    MODEL_PATH = os.path.join(MODEL_DIR, "paligemma2-3b-pt-224.b16.npz")
    print(f"   Model path: {MODEL_PATH}")

    model_config = ml_collections.FrozenConfigDict({
        'llm': {'vocab_size': 257_152, 'variant': 'gemma2_2b', 'final_logits_softcap': 0.0},
        'img': {'variant': 'So400m/14', 'pool_type': 'none', 'scan': True, 'dtype_mm': 'float16'}
    })

    # Load Architecture
    model = paligemma_model.Model(**model_config)
    print("Model instantiated")

    # Create JIT function
    score_fn = make_score_fn(model)
    print("JIT function created!")

    # Load weights with custom loader
    print("Loading with custom loader...")
    params = load_params_macos_safe(MODEL_PATH, model_config)
    print("Checkpoint loaded!")

    # Prep test data
    if not os.path.exists("test.jpg"):
        Image.new('RGB', (224, 224), 'red').save("test.jpg")
    
    with Image.open("test.jpg") as img:
        img_arr = np.array(img.convert("RGB").resize((224, 224))).astype(np.float32) / 127.5 - 1.0
        img_batch = img_arr[None, ...]
    print("Image prepared!")

    # TEMPORARY: Skip sentencepiece on macOS due to mutex issues
    # Just use dummy tokens for testing
    if IS_MACOS:
        print("   (Using dummy tokens on macOS to avoid sentencepiece mutex)")
        tokens = [1, 2, 3, 4, 5] + [0]*123  # Dummy tokens
    else:
        tokenizer = sentencepiece.SentencePieceProcessor()
        tokenizer.Load(TOKENIZER_PATH)
        tokens = tokenizer.EncodeAsIds("Revenue growth")
        tokens = tokens[:128] + [0]*(128-len(tokens))
    
    txt_batch = jnp.array([tokens], dtype=jnp.int32)
    mask_batch = jnp.array([[1 if t!=0 else 0 for t in tokens]], dtype=jnp.int32)
    print("Text prepared!")

    print("🚀 Scoring...")
    score = score_fn(params, img_batch, txt_batch, mask_batch)
    print(f"✅ Score: {float(score[0]):.4f}")

if __name__ == "__main__":
    main()
