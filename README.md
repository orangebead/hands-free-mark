## Installation

### Option 1: Using Conda

1. **Install Miniconda:**
   - Download from: https://docs.conda.io/en/latest/miniconda.html
   - Or: `brew install --cask miniconda`

2. **Create environment:**
```bash
conda create -n camera-mark python=3.11
conda activate camera-mark
pip install -r requirements.txt
```

### Option 2: Using venv
```bash
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
conda activate microscope  # or: source venv/bin/activate
python pymark.py your_image.jpg
```
