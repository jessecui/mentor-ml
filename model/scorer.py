import os
import sys

# --- CONFIGURATION ---
# Force CPU mode for Mac/No-GPU setups
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- 1. LINK THE ENGINE ---
# We assume "big_vision_repo" already exists because you ran the setup commands.
REPO_DIR = "big_vision_repo"
if REPO_DIR not in sys.path:
    sys.path.append(REPO_DIR)

# --- IMPORTS ---
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import sentencepiece
import ml_collections

# If this fails, it means you didn't run "git clone" in Phase 1
try:
    import big_vision.models.proj.paligemma.paligemma as paligemma_model
except ImportError:
    print("❌ ERROR: Could not import 'big_vision'.")
    print(f"   Did you run: git clone ... {REPO_DIR} ?")
    sys.exit(1)

# --- 2. THE JAX KERNEL ---
def make_score_fn(model):
    """Create a JIT-compiled ColPali scoring function for the given model."""
    
    @jax.jit
    def score_colpali(params, images, text_tokens, text_mask):
        """ColPali-style MaxSim scoring between image and text embeddings.
        
        Args:
            params: Model parameters
            images: [Batch, H, W, 3] image tensor
            text_tokens: [Batch, SeqLen] token IDs
            text_mask: [Batch, SeqLen] binary mask (1 for real tokens, 0 for padding)
        
        Returns:
            [Batch] scores for each image-text pair
        """
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
        # sim_matrix: [Batch, SeqLen, NumPatches]
        sim_matrix = jnp.einsum('btd,bpd->btp', ztxt, zimg)
        
        # Max over image patches for each text token
        max_scores = jnp.max(sim_matrix, axis=-1)  # [Batch, SeqLen]
        
        # Sum scores only for valid (non-padding) text tokens
        return jnp.sum(max_scores * text_mask, axis=-1)  # [Batch]
    
    return score_colpali

# --- 3. EXECUTION ---
def main():
    print("⚙️  Initializing JAX Scorer...")
    
    # Check for Tokenizer (relative to this file's directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(script_dir, "paligemma_tokenizer.model")
    if not os.path.exists(tokenizer_path):
        print(f"❌ ERROR: Missing {tokenizer_path}")
        print("   The tokenizer should be included in the repo at model/paligemma_tokenizer.model")
        return

    # --- Download model checkpoint from Kaggle ---
    # Requires: pip install kagglehub
    # And Kaggle credentials in ~/.kaggle/kaggle.json or env vars KAGGLE_USERNAME/KAGGLE_KEY
    # Request access at: https://www.kaggle.com/models/google/paligemma-2/
    import kagglehub  # type: ignore[import-untyped]
    
    KAGGLE_HANDLE = "google/paligemma-2/jax/paligemma2-3b-pt-224"
    print("⏳ Downloading checkpoint from Kaggle (first run only)...")
    MODEL_PATH = kagglehub.model_download(KAGGLE_HANDLE)
    print(f"   Model path: {MODEL_PATH}")

    # Model configuration (must match the checkpoint)
    model_config = ml_collections.FrozenConfigDict({
        "llm": {"vocab_size": 257_152, "variant": "gemma2_2b", "final_logits_softcap": 0.0},
        "img": {"variant": "So400m/14", "pool_type": "none", "scan": True, "dtype_mm": "float16"}
    })
    
    # Load Architecture
    model = paligemma_model.Model(**model_config)  # type: ignore[arg-type]
    
    # Create the scoring function bound to this model
    score_colpali = make_score_fn(model)
    
    # Load pretrained weights (not random init!)
    print("⏳ Loading pretrained weights...")
    params = paligemma_model.load(None, MODEL_PATH, model_config)
    
    # Prep Data (On the fly)
    if not os.path.exists("test.jpg"):
        Image.new('RGB', (224, 224), 'red').save("test.jpg")
    
    # Load Image
    with Image.open("test.jpg") as img:
        img_arr = np.array(img.convert("RGB").resize((224, 224))).astype(np.float32) / 127.5 - 1.0
        img_batch = img_arr[None, ...]

    # Load Text
    tokenizer = sentencepiece.SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)
    tokens = tokenizer.EncodeAsIds("Revenue growth")
    tokens = tokens[:128] + [0]*(128-len(tokens))
    txt_batch = jnp.array([tokens], dtype=jnp.int32)
    mask_batch = jnp.array([[1 if t!=0 else 0 for t in tokens]], dtype=jnp.int32)

    # Run
    print("🚀 Scoring...")
    score = score_colpali(params, img_batch, txt_batch, mask_batch)
    print(f"✅ Score: {score[0]:.4f}")

if __name__ == "__main__":
    main()