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

3. **Get your API token:**
   - Go to [Kaggle Settings](https://www.kaggle.com/settings)
   - Under "API", click "Generate New Token"
   - Copy the token value

4. **Install credentials** (choose one):
   ```bash
   # Option A: Environment variable (recommended)
   export KAGGLE_API_TOKEN=your_token_here

   # Option B: Add to .env file
   echo "KAGGLE_API_TOKEN=your_token_here" >> .env

   # Option C: Save kaggle.json to ~/.kaggle/
   mkdir -p ~/.kaggle
   echo '{"username":"YOUR_USERNAME","key":"YOUR_TOKEN"}' > ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 3. Run the Scorer

```bash
python model/scorer.py
```

The first run will download the model weights to `~/.cache/kagglehub/` (cached for future runs).