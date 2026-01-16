"""
Scrape ML diagrams from Jay Alammar's blog (jalammar.github.io).

Jay Alammar's blog contains the most iconic transformer/attention visualizations.
These diagrams are frequently referenced in ML education and are perfect for
teaching concepts like self-attention, embeddings, and language models.

Key posts:
- The Illustrated Transformer
- The Illustrated GPT-2
- The Illustrated BERT
- Visualizing A Neural Machine Translation Model
- The Illustrated Word2vec
- And more...

Usage:
    python benchmark/scripts/scrape_ai_ml_diagrams.py
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_URL = "https://jalammar.github.io"

# The Illustrated Series - Top 5 most cited/referenced posts
TARGET_POSTS = [
    # 1. The Illustrated Transformer
    "/illustrated-transformer/",
    # 2. The Illustrated BERT
    "/illustrated-bert/",
    # 3. The Illustrated GPT-2
    "/illustrated-gpt2/",
    # 4. The Illustrated Word2vec
    "/illustrated-word2vec/",
    # 5. The Illustrated Stable Diffusion
    "/illustrated-stable-diffusion/",
]

# Output paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "corpus" / "images" / "diagrams"
METADATA_FILE = PROJECT_ROOT / "benchmark" / "corpus" / "metadata" / "corpus.json"

# Request settings
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) MentorML-Benchmark/1.0"
}
REQUEST_DELAY = 0.5  # Be polite to the server


# =============================================================================
# SCRAPING FUNCTIONS
# =============================================================================


def get_post_title(soup: BeautifulSoup) -> str:
    """Extract the post title."""
    title_tag = soup.find("h1", class_="post-title") or soup.find("h1")
    if title_tag:
        return title_tag.get_text(strip=True)
    return "Unknown"


def extract_figures_from_post(post_url: str) -> list[dict]:
    """
    Extract all figures and their context from a blog post.

    Returns:
        List of dicts with figure info: src, alt, caption, context
    """
    full_url = urljoin(BASE_URL, post_url)
    print(f"  Fetching: {full_url}")

    try:
        resp = requests.get(full_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Error fetching {full_url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    post_title = get_post_title(soup)

    figures = []

    # Find all images in the post content
    content_div = (
        soup.find("div", class_="post-content") or soup.find("article") or soup
    )

    for img in content_div.find_all("img"):
        src = str(img.get("src", ""))

        # Skip tiny images, icons, and tracking pixels
        if not src or any(
            skip in src.lower()
            for skip in [
                "tracking",
                "pixel",
                "icon",
                "logo",
                "avatar",
                "badge",
                "button",
                "spinner",
                "loading",
                ".gif",
                "analytics",
            ]
        ):
            continue

        # Skip very small images (likely icons)
        width = str(img.get("width", ""))
        height = str(img.get("height", ""))
        if width and width.isdigit() and int(width) < 100:
            continue
        if height and height.isdigit() and int(height) < 100:
            continue

        # Get alt text and any caption
        alt_text = img.get("alt", "")

        # Look for caption in surrounding elements
        caption = ""
        parent = img.parent
        if parent:
            # Check for figcaption
            figcaption = parent.find("figcaption")
            if figcaption:
                caption = figcaption.get_text(strip=True)
            else:
                # Check for nearby small/em text
                next_sib = img.find_next_sibling(["small", "em", "p"])
                if next_sib and len(next_sib.get_text(strip=True)) < 200:
                    caption = next_sib.get_text(strip=True)

        # Get surrounding paragraph for context
        context = ""
        prev_p = img.find_previous("p")
        if prev_p:
            context = prev_p.get_text(strip=True)[:300]

        figures.append(
            {
                "src": src,
                "alt": alt_text,
                "caption": caption,
                "context": context,
                "post_title": post_title,
                "post_url": post_url,
            }
        )

    return figures


def download_figure(fig_info: dict, idx: int) -> dict | None:
    """
    Download a figure and return metadata.

    Args:
        fig_info: Figure information dict
        idx: Global figure index

    Returns:
        Metadata dict or None if download failed
    """
    src = fig_info["src"]

    # Resolve relative URLs
    if src.startswith("//"):
        src = "https:" + src
    elif src.startswith("/"):
        src = BASE_URL + src
    elif not src.startswith("http"):
        src = urljoin(BASE_URL + fig_info["post_url"], src)

    # Determine file extension
    parsed = urlparse(src)
    path = parsed.path.lower()

    if ".png" in path:
        ext = ".png"
    elif ".jpg" in path or ".jpeg" in path:
        ext = ".jpg"
    elif ".webp" in path:
        ext = ".webp"
    elif ".svg" in path:
        # Skip SVGs - they often don't render well and may be icons
        return None
    else:
        ext = ".png"  # Default

    try:
        resp = requests.get(src, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        # Check content type
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type and "octet-stream" not in content_type:
            print(f"    Skipping non-image: {src[:60]}...")
            return None

        # Check file size (skip tiny files)
        if len(resp.content) < 5000:  # Less than 5KB
            print(f"    Skipping tiny image: {src[:60]}...")
            return None

        # Generate filename
        filename = f"diagram_{idx:03d}{ext}"
        filepath = OUTPUT_DIR / filename

        filepath.write_bytes(resp.content)
        print(f"    Downloaded: {filename}")

        return {
            "id": f"diagram_{idx:03d}",
            "filename": filename,
            "source": "diagrams",
            "source_url": src,
            "post_title": fig_info["post_title"],
            "post_url": BASE_URL + fig_info["post_url"],
            "alt_text": fig_info["alt"],
            "caption": fig_info["caption"],
            "context": fig_info["context"],
            # To be filled in later (manually or with LLM)
            "concepts": [],
            "suggested_queries": [],
        }

    except requests.RequestException as e:
        print(f"    Failed to download {src[:60]}...: {e}")
        return None


def scrape_all_posts() -> list[dict]:
    """
    Scrape all target posts and download figures.

    Returns:
        List of metadata dicts for all downloaded figures
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_metadata = []
    global_idx = 0

    print(f"Scraping {len(TARGET_POSTS)} posts from Jay Alammar's blog...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    for post_url in TARGET_POSTS:
        print(f"Processing: {post_url}")

        figures = extract_figures_from_post(post_url)
        print(f"  Found {len(figures)} candidate figures")

        for fig in figures:
            metadata = download_figure(fig, global_idx)
            if metadata:
                all_metadata.append(metadata)
                global_idx += 1

        time.sleep(REQUEST_DELAY)
        print()

    return all_metadata


def save_metadata(metadata: list[dict]) -> None:
    """Save metadata to JSON file."""
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {METADATA_FILE}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Main entry point."""
    print("=" * 60)
    print("Jay Alammar Blog Scraper")
    print("=" * 60)
    print()

    metadata = scrape_all_posts()

    if metadata:
        save_metadata(metadata)

        print()
        print("=" * 60)
        print(f"✅ Successfully downloaded {len(metadata)} figures")
        print(f"📁 Images: {OUTPUT_DIR}")
        print(f"📝 Metadata: {METADATA_FILE}")
        print("=" * 60)
    else:
        print("❌ No figures were downloaded")


if __name__ == "__main__":
    main()
