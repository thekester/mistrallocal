
# Mistral Model Inference

This project uses the Mistral model to generate responses based on text prompts. The script uses `tqdm` to display the progress of token generation.

## Prerequisites

Make sure you have [Anaconda](https://www.anaconda.com/products/distribution) installed on your machine.

## Installation

### Step 1: Create a New Anaconda Environment

Open a terminal (or Anaconda Prompt on Windows) and create a new environment:

```bash
conda create --name mistral_env python=3.9
```

Then activate the environment:

```bash
conda activate mistral_env
```

### Step 2: Install PyTorch with CUDA Support

Install PyTorch with CUDA support:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Step 3: Install Additional Dependencies

Install the required additional packages:

```bash
pip install mistral-inference tqdm
```

### Step 4: Download the Models

Download the models from [Mistral AI Models](https://docs.mistral.ai/getting-started/open_weight_models/). For example, to download Mistral-7B-v0.3 (note that the TAR file is 17 GB):

```bash
wget https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-v0.3.tar
mkdir -p mistral_models/mistral-7b
tar -xf mistral-7B-v0.3.tar -C mistral_models/mistral-7b
```

### Step 5: Run the Script

Ensure that the model files are present in the `mistral_models/mistral-7b` directory.

Run the main script:

```bash
python path/to/your/script.py
```

## Notes

- The token generation phase ("Generating tokens") is the most time-consuming part.
