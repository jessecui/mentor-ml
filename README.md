# MentorML
A conversational agent for learning ML Engineering skills, built with LangChain for orchestration and Google Cloud Vertex AI for the model endpoint.

## Setup

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/jessecui/mentor-ml.git
cd mentor-ml

# Clone big_vision (PaliGemma model architecture)
git clone --quiet --branch=main --depth=1 https://github.com/google-research/big_vision big_vision_repo

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Kaggle (for PaliGemma Weights)

The ColPali reranker uses PaliGemma-2 weights (~6GB), which are downloaded from Kaggle on first run.

1. **Create a Kaggle account** at [kaggle.com](https://www.kaggle.com/)

2. **Request model access:**
   - Go to [PaliGemma-2 model page](https://www.kaggle.com/models/google/paligemma-2/)
   - Click "Request Access" and accept the terms

3. **Get your API credentials:**
   - Go to [Kaggle Settings](https://www.kaggle.com/settings)
   - Click "Create New Token" → downloads `kaggle.json`

4. **Install credentials** (choose one):
   ```bash
   # Option A: Save to ~/.kaggle/
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json

   # Option B: Set environment variables
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   ```

### 3. Run the Scorer

```bash
python model/scorer.py
```

The first run will download the model weights to `~/.cache/kagglehub/` (cached for future runs).