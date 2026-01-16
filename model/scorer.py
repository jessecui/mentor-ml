"""
SigLIP-based image-text retrieval scoring using JAX/Flax.

Uses contrastive embeddings for fast, accurate image-text matching.
SigLIP (Sigmoid Language-Image Pretraining) is designed specifically
for retrieval tasks and outperforms CLIP on standard benchmarks.

Usage:
    from model.scorer import SigLIPScorer
    
    scorer = SigLIPScorer()
    score = scorer.score(image_path, query)
    
    # Or batch scoring
    scores = scorer.score_batch(image_paths, query)
"""

import os
import sys
import platform
import warnings
import subprocess
from pathlib import Path

# =============================================================================
# ENVIRONMENT SETUP
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
# SigLIP 2 uses Gemma tokenizer (256k vocab)
TOKENIZER_PATH = SCRIPT_DIR / "gemma_tokenizer.model"

if str(REPO_DIR) not in sys.path:
    sys.path.append(str(REPO_DIR))

# =============================================================================
# ML LIBRARY IMPORTS
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

import big_vision.models.proj.image_text.two_towers as two_towers
from big_vision.utils import recover_tree, recover_dtype

# =============================================================================
# CONSTANTS
# =============================================================================

# SigLIP 2 So400m/14 at 384px - high quality, same vision backbone as PaliGemma
# Direct download URL (no gsutil needed)
CHECKPOINT_URL = "https://storage.googleapis.com/big_vision/siglip2/siglip2_so400m14_384.npz"
CHECKPOINT_LOCAL = PROJECT_ROOT / "model" / "siglip2_so400m14_384.npz"
IMAGE_SIZE = 384
MAX_SEQ_LEN = 64  # SigLIP uses shorter sequences

# Model configuration matching the checkpoint (from SigLIP2_demo.ipynb)
VARIANT = "So400m/14"
TXTVARIANT = "So400m"
EMBDIM = 1152
VOCAB_SIZE = 256_000

MODEL_CONFIG = ml_collections.ConfigDict({
    "image_model": "vit",
    "image": {
        "variant": VARIANT,
        "pool_type": "map",
        "scan": True,
    },
    "text_model": "proj.image_text.text_transformer",
    "text": {
        "variant": TXTVARIANT,
        "vocab_size": VOCAB_SIZE,
        "scan": True,
    },
    "out_dim": (None, EMBDIM),  # None means use model's default
    "bias_init": -10.0,  # Required for SigLIP 2
})


# =============================================================================
# TOKENIZER
# =============================================================================

def tokenize(text: str, max_len: int = MAX_SEQ_LEN) -> list[int]:
    """
    Tokenize text using SentencePiece (Gemma tokenizer).
    
    SigLIP 2 works best with lowercase text.
    Uses BOS=no, EOS=sticky (appended at end).
    
    Runs in subprocess to avoid C++ conflicts on macOS.
    """
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}\n"
            "Download with: curl -L -o model/gemma_tokenizer.model "
            "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
        )
    
    # Lowercase for best SigLIP 2 performance
    text = text.lower()
    
    script = f'''
import sentencepiece
sp = sentencepiece.SentencePieceProcessor()
sp.Load("{TOKENIZER_PATH}")
tokens = sp.EncodeAsIds({repr(text)})
# No BOS, sticky EOS (Gemma-style for SigLIP 2)
eos_id = 1  # EOS token
tokens = tokens[:({max_len} - 1)] + [eos_id]
tokens = tokens + [0] * ({max_len} - len(tokens))
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

def download_checkpoint():
    """Download SigLIP checkpoint if not present."""
    if CHECKPOINT_LOCAL.exists():
        return str(CHECKPOINT_LOCAL)
    
    print(f"Downloading SigLIP 2 checkpoint (~1.5GB)...")
    print(f"  From: {CHECKPOINT_URL}")
    print(f"  To: {CHECKPOINT_LOCAL}")
    
    import urllib.request
    CHECKPOINT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_LOCAL)
    print(f"  ✅ Downloaded checkpoint")
    
    return str(CHECKPOINT_LOCAL)


def load_params(checkpoint_path: str) -> dict:
    """Load SigLIP model parameters from checkpoint."""
    with np.load(checkpoint_path, allow_pickle=False) as npz:
        keys = list(npz.keys())
        values = [np.array(npz[k]) for k in keys]
    
    checkpoint = recover_tree(keys, values)
    checkpoint = jax.tree.map(recover_dtype, checkpoint)
    
    return checkpoint


# =============================================================================
# SCORING FUNCTION
# =============================================================================

def make_siglip_score_fn(model):
    """Create JIT-compiled scoring function."""
    
    @jax.jit
    def encode_image(params, images):
        """Encode images to normalized embeddings."""
        zimg, _, out = model.apply(params, images, text=None, train=False)
        return out["img/normalized"]
    
    @jax.jit
    def encode_text(params, tokens):
        """Encode text to normalized embeddings."""
        _, ztxt, out = model.apply(params, image=None, text=tokens, train=False)
        return out["txt/normalized"]
    
    @jax.jit
    def score_with_params(img_emb, txt_emb, temperature, bias):
        """
        Compute SigLIP sigmoid probability score.
        
        SigLIP uses sigmoid(dot_product * temperature + bias) for scoring.
        Temperature and bias are learned parameters stored in the checkpoint.
        """
        logits = jnp.sum(img_emb * txt_emb, axis=-1) * temperature + bias
        return jax.nn.sigmoid(logits)
    
    return encode_image, encode_text, score_with_params


# =============================================================================
# SCORER CLASS
# =============================================================================

class SigLIPScorer:
    """
    Image-query relevance scorer using SigLIP 2 embeddings.
    
    Uses contrastive embeddings with sigmoid scoring for fast,
    accurate image-text matching. Much faster than PaliGemma VQA.
    
    Example:
        scorer = SigLIPScorer()
        score = scorer.score("diagram.png", "transformer attention mechanism")
    """
    
    def __init__(self, checkpoint_path: str | None = None):
        """
        Initialize the scorer. Downloads model from GCS if needed.
        
        Args:
            checkpoint_path: Path to .npz checkpoint, or None to auto-download
        """
        if checkpoint_path is None:
            checkpoint_path = download_checkpoint()
        
        # Initialize model
        self.model = two_towers.Model(**MODEL_CONFIG)  # type: ignore[arg-type]
        self.encode_image, self.encode_text, self.score_fn = make_siglip_score_fn(self.model)
        self.params = load_params(checkpoint_path)
        
        # Extract learned temperature and bias from checkpoint
        # t is stored as log(temperature), shape (1,)
        t_param = self.params.get("t", jnp.log(jnp.array([10.0])))
        self.temperature = float(jnp.exp(t_param).squeeze())
        b_param = self.params.get("b", jnp.array([-10.0]))
        self.bias = float(jnp.asarray(b_param).squeeze())
        
        # Cache for image embeddings
        self._image_cache: dict[str, np.ndarray] = {}
    
    def preprocess_image(self, image: Image.Image | np.ndarray | str) -> np.ndarray:
        """
        Preprocess image: resize and normalize to [-1, 1].
        
        Args:
            image: PIL Image, numpy array [H,W,3], or file path
        
        Returns:
            [1, IMAGE_SIZE, IMAGE_SIZE, 3] float32 array
        """
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            image = np.array(image)
        
        image = image.astype(np.float32) / 127.5 - 1.0
        return image[None, ...]
    
    def get_image_embedding(self, image: Image.Image | np.ndarray | str, use_cache: bool = True) -> np.ndarray:
        """
        Get normalized embedding for an image.
        
        Args:
            image: Image (file path, PIL Image, or numpy array)
            use_cache: Whether to cache embeddings (by file path)
        
        Returns:
            [1, EmbedDim] normalized embedding
        """
        cache_key = image if isinstance(image, str) else None
        
        if use_cache and cache_key and cache_key in self._image_cache:
            return self._image_cache[cache_key]
        
        img_batch = self.preprocess_image(image)
        embedding = self.encode_image(self.params, img_batch)
        embedding = np.array(embedding)
        
        if use_cache and cache_key:
            self._image_cache[cache_key] = embedding
        
        return embedding
    
    def get_text_embedding(self, query: str) -> np.ndarray:
        """
        Get normalized embedding for a text query.
        
        Args:
            query: Text query
        
        Returns:
            [1, EmbedDim] normalized embedding
        """
        tokens = tokenize(query, max_len=MAX_SEQ_LEN)
        txt_batch = jnp.array([tokens], dtype=jnp.int32)
        embedding = self.encode_text(self.params, txt_batch)
        return np.array(embedding)
    
    def score(self, image: Image.Image | np.ndarray | str, query: str) -> float:
        """
        Compute relevance score between an image and text query.
        
        Args:
            image: Image (file path, PIL Image, or numpy array)
            query: Search query to match
        
        Returns:
            Sigmoid probability score in [0, 1]. Higher = more relevant.
        """
        img_emb = self.get_image_embedding(image)
        txt_emb = self.get_text_embedding(query)
        score = self.score_fn(img_emb, txt_emb, self.temperature, self.bias)
        return float(score[0])
    
    def score_batch(
        self, 
        image_paths: list[str], 
        query: str, 
        show_progress: bool = True
    ) -> dict[str, float]:
        """
        Score a query against multiple images.
        
        Efficient: computes text embedding once, reuses for all images.
        
        Args:
            image_paths: List of image file paths
            query: Search query to match
            show_progress: Whether to print progress
        
        Returns:
            Dict mapping image_path -> relevance score
        """
        # Encode query once
        txt_emb = self.get_text_embedding(query)
        
        scores = {}
        for i, path in enumerate(image_paths):
            img_emb = self.get_image_embedding(path)
            score = self.score_fn(img_emb, txt_emb, self.temperature, self.bias)
            scores[path] = float(score[0])
            
            if show_progress and (i + 1) % 10 == 0:
                print(f"  Scored {i + 1}/{len(image_paths)} images")
        
        if show_progress:
            print(f"  ✅ Scored {len(scores)} images")
        
        return scores
    
    def clear_cache(self):
        """Clear the image embedding cache."""
        self._image_cache.clear()


# =============================================================================
# DEMO
# =============================================================================

def create_test_image(text: str, path: str = "test.jpg") -> str:
    """Create a test image with rendered text."""
    from PIL import ImageDraw, ImageFont
    
    img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), 'white')
    draw = ImageDraw.Draw(img)
    
    # Use default font, draw text centered
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    # Simple center positioning
    draw.text((20, 100), text, fill='black', font=font)
    img.save(path)
    return path


def main():
    """Demo: test SigLIP scoring."""
    print("=" * 60)
    print("SigLIP Scorer Demo")
    print("=" * 60)
    
    scorer = SigLIPScorer()
    print("✅ Model loaded")
    
    # Test with sample images if available
    test_dir = PROJECT_ROOT / "benchmark" / "corpus" / "images" / "diagrams"
    if test_dir.exists():
        images = sorted(test_dir.glob("*.png"))[:3]
        if images:
            query = "transformer attention mechanism"
            print(f"\nQuery: '{query}'")
            print("-" * 40)
            
            for img_path in images:
                score = scorer.score(str(img_path), query)
                print(f"  {img_path.name}: {score:+.4f}")
    else:
        print("\nNo test images found. Run with your own images:")
        print("  scorer.score('image.png', 'your query')")


if __name__ == "__main__":
    main()
