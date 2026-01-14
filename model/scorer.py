"""
ColPali-style image-text scoring using PaliGemma-2.

Computes MaxSim similarity between document images and text queries using
the PaliGemma-2 vision-language model. Useful for document retrieval tasks.

Usage:
    from model.scorer import PaliGemmaScorer
    
    scorer = PaliGemmaScorer()
    score = scorer.score("document.png", "revenue growth")
"""

import os
import sys
import platform
import warnings
import subprocess
from pathlib import Path

# =============================================================================
# ENVIRONMENT SETUP
#
# Environment variables must be set BEFORE importing TensorFlow/JAX.
# On macOS, threading settings prevent C++ mutex conflicts between libraries.
# Set USE_GPU=true to enable CUDA (for Cloud Run deployment).
# =============================================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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

# =============================================================================
# PATH SETUP
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_DIR = PROJECT_ROOT / "big_vision_repo"
TOKENIZER_PATH = SCRIPT_DIR / "paligemma_tokenizer.model"

if str(REPO_DIR) not in sys.path:
    sys.path.append(str(REPO_DIR))

# =============================================================================
# ML LIBRARY IMPORTS
#
# Import order matters: TensorFlow must be imported and disabled before JAX.
# big_vision imports TensorFlow internally, so we configure it first.
# =============================================================================

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
if IS_MACOS:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import ml_collections

if IS_MACOS or not USE_GPU:
    jax.config.update('jax_platform_name', 'cpu')

import big_vision.models.proj.paligemma.paligemma as paligemma_model
import big_vision.models.vit as vit_module
from big_vision.utils import recover_tree, recover_dtype

# =============================================================================
# CONSTANTS
# =============================================================================

# PaliGemma-2 3B model configuration
_MODEL_CONFIG_DICT = {
    'llm': {'vocab_size': 257_152, 'variant': 'gemma2_2b', 'final_logits_softcap': 0.0},
    'img': {'variant': 'So400m/14', 'pool_type': 'none', 'scan': True, 'dtype_mm': 'float16'}
}
MODEL_CONFIG: ml_collections.FrozenConfigDict = ml_collections.FrozenConfigDict(_MODEL_CONFIG_DICT)

KAGGLE_HANDLE = "google/paligemma-2/jax/paligemma2-3b-pt-224"
CHECKPOINT_FILENAME = "paligemma2-3b-pt-224.b16.npz"
IMAGE_SIZE = 224
MAX_SEQ_LEN = 128


# =============================================================================
# TOKENIZER
# =============================================================================

def tokenize(text: str, max_len: int = MAX_SEQ_LEN) -> list[int]:
    """
    Tokenize text using SentencePiece.
    
    Runs in a subprocess to avoid C++ mutex conflicts with TensorFlow/JAX on macOS.
    The subprocess overhead is minimal (~50ms) and ensures stability.
    
    Args:
        text: Input text to tokenize
        max_len: Maximum sequence length (padded with zeros)
    
    Returns:
        List of token IDs, padded to max_len
    """
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")
    
    script = f'''
import sentencepiece
sp = sentencepiece.SentencePieceProcessor()
sp.Load("{TOKENIZER_PATH}")
tokens = sp.EncodeAsIds({repr(text)})
tokens = tokens[:{max_len}] + [0] * ({max_len} - len(tokens))
print(tokens)
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f"Tokenization failed: {result.stderr}")
    return eval(result.stdout.strip())


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def load_params(checkpoint_path: str) -> dict:
    """
    Load model parameters from a PaliGemma checkpoint.
    
    Handles:
    - bfloat16 dtype recovery (stored as raw bytes in .npz)
    - ViT checkpoint format fixes (pyloop to scan conversion)
    
    Args:
        checkpoint_path: Path to .npz checkpoint file
    
    Returns:
        Dict with 'img' and 'llm' parameter trees
    """
    with np.load(checkpoint_path, allow_pickle=False) as npz:
        keys = list(npz.keys())
        values = [np.array(npz[k]) for k in keys]
    
    checkpoint = recover_tree(keys, values)
    checkpoint = jax.tree.map(recover_dtype, checkpoint)
    
    # Handle different checkpoint wrapper formats
    if "params" in checkpoint:
        params = checkpoint["params"]
    elif "opt" in checkpoint:
        params = checkpoint["opt"]["target"]
    else:
        params = checkpoint
    
    restored: dict = {"img": None, "llm": None}
    
    if "img" in params:
        restored["img"] = vit_module.fix_old_checkpoints(params["img"])
        # Convert ViT from pyloop to scan format if needed
        if _MODEL_CONFIG_DICT["img"].get("scan"):
            transformer = restored["img"].get("Transformer", {})
            if "encoderblock" not in transformer:
                restored["img"] = vit_module.pyloop_to_scan(restored["img"])
    
    if "llm" in params:
        restored["llm"] = params["llm"]
    
    return restored


# =============================================================================
# SCORING FUNCTION
# =============================================================================

def make_score_fn(model):
    """
    Create a JIT-compiled ColPali scoring function.
    
    ColPali uses "MaxSim" scoring: for each text token, find its maximum
    cosine similarity across all image patches, then sum these maxes.
    This captures fine-grained text-to-region matching.
    """
    
    @jax.jit
    def score_colpali(params, images, text_tokens, text_mask):
        """
        Args:
            params: Model parameters
            images: [B, H, W, 3] float32, normalized to [-1, 1]
            text_tokens: [B, SeqLen] int32 token IDs
            text_mask: [B, SeqLen] int32 mask (1=real token, 0=padding)
        
        Returns:
            [B] float32 similarity scores
        """
        # Embed image patches and text tokens
        zimg, _ = model.apply({'params': params}, images, train=False, method=model.embed_image)
        ztxt, _ = model.apply({'params': params}, text_tokens, train=False, method=model.embed_text)
        
        # L2 normalize for cosine similarity
        zimg = zimg / (jnp.linalg.norm(zimg, axis=-1, keepdims=True) + 1e-8)
        ztxt = ztxt / (jnp.linalg.norm(ztxt, axis=-1, keepdims=True) + 1e-8)
        
        # MaxSim: [B, TextLen, Patches] -> max over patches -> sum over text
        sim_matrix = jnp.einsum('btd,bpd->btp', ztxt, zimg)
        max_scores = jnp.max(sim_matrix, axis=-1)
        return jnp.sum(max_scores * text_mask, axis=-1)
    
    return score_colpali


# =============================================================================
# SCORER CLASS
# =============================================================================

class PaliGemmaScorer:
    """
    Document-query similarity scorer using PaliGemma-2.
    
    Uses ColPali-style MaxSim scoring to match text queries against
    document images. Higher scores indicate better matches.
    
    Example:
        scorer = PaliGemmaScorer()
        score = scorer.score("quarterly_report.png", "revenue growth")
    """
    
    def __init__(self, checkpoint_path: str | None = None):
        """
        Initialize the scorer. Downloads model from Kaggle if no path provided.
        
        Args:
            checkpoint_path: Path to .npz checkpoint, or None to auto-download
        """
        if checkpoint_path is None:
            import kagglehub
            model_dir = kagglehub.model_download(KAGGLE_HANDLE)
            checkpoint_path = os.path.join(model_dir, CHECKPOINT_FILENAME)
        
        self.model = paligemma_model.Model(**MODEL_CONFIG)  # type: ignore[arg-type]
        self.score_fn = make_score_fn(self.model)
        self.params = load_params(checkpoint_path)
    
    def preprocess_image(self, image: Image.Image | np.ndarray | str) -> np.ndarray:
        """
        Preprocess image: resize to 224x224 and normalize to [-1, 1].
        
        Args:
            image: PIL Image, numpy array [H,W,3], or file path
        
        Returns:
            [1, 224, 224, 3] float32 array
        """
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            image = np.array(image)
        
        image = image.astype(np.float32) / 127.5 - 1.0
        return image[None, ...]
    
    def preprocess_text(self, text: str) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Tokenize text and create attention mask.
        
        Returns:
            (tokens, mask) - both [1, 128] int32 arrays
        """
        tokens = tokenize(text, max_len=MAX_SEQ_LEN)
        mask = [1 if t != 0 else 0 for t in tokens]
        return jnp.array([tokens], dtype=jnp.int32), jnp.array([mask], dtype=jnp.int32)
    
    def score(self, image: Image.Image | np.ndarray | str, query: str) -> float:
        """
        Compute similarity between a document image and text query.
        
        Args:
            image: Document image (file path, PIL Image, or numpy array)
            query: Text query to match
        
        Returns:
            Similarity score (higher = more relevant)
        """
        img_batch = self.preprocess_image(image)
        txt_batch, mask_batch = self.preprocess_text(query)
        scores = self.score_fn(self.params, img_batch, txt_batch, mask_batch)
        return float(scores[0])


# =============================================================================
# DEMO
# =============================================================================

def main():
    """Demo: score a test image against a sample query."""
    print("⚙️  Initializing PaliGemma Scorer...")
    scorer = PaliGemmaScorer()
    print("✅ Model loaded!\n")
    
    # Create a simple test image if none exists
    test_image_path = "test.jpg"
    if not os.path.exists(test_image_path):
        Image.new('RGB', (224, 224), 'red').save(test_image_path)
    
    query = "Revenue growth"
    print(f"🔍 Query: '{query}'")
    print(f"📄 Image: {test_image_path}")
    
    score = scorer.score(test_image_path, query)
    print(f"✅ Score: {score:.4f}")


if __name__ == "__main__":
    main()
