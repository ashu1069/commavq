# My CommaVQ Compression Solution

## Challenge Overview

Losslessly compress 5,000 minutes of driving video (VQ tokens) from the commaVQ dataset.

**Goal**: Maximize compression rate = (original bits) / (compressed bytes)

## Data Format

- Dataset: `commaai/commavq` (splits 0 and 1)
- Each example contains:
  - `token.npy`: shape `(1200, 8, 16)` - 1200 frames, 8×16 tokens per frame
  - Token values: 0-1023 (10 bits each)
  - Original size per example: `1200 × 128 × 10 / 8 = 192,000 bytes`

## Solution Architecture

```
my_solution/
├── compress.py          # Main compression script (run this)
├── decompress.py        # Decompression script (included in submission)
├── config.py            # Configuration and hyperparameters
├── encoders/
│   ├── __init__.py
│   ├── arithmetic.py    # Arithmetic coding implementation
│   └── ans.py           # Asymmetric Numeral Systems (optional)
├── models/
│   ├── __init__.py
│   ├── base.py          # Base predictor interface
│   ├── statistical.py   # Statistical models (n-gram, Markov)
│   └── neural.py        # Neural network predictors (GPT-style)
├── scripts/
│   ├── train_model.py   # Train prediction model
│   └── test_local.py    # Local testing script
└── requirements.txt
```

## Approach

### Strategy 1: Statistical + Entropy Coding (Baseline++)
- Exploit spatial/temporal correlations
- Use context-adaptive arithmetic coding

### Strategy 2: Neural Network + Arithmetic Coding (Competitive)
- Train a GPT-style model to predict next token probabilities
- Use predictions for arithmetic coding
- Include model weights in submission

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Train a prediction model
python scripts/train_model.py

# 3. Compress the dataset
python compress.py

# 4. Test locally
python scripts/test_local.py

# 5. Submit: compression_challenge_submission.zip
```

## Compression Rate Targets

| Method | Expected Rate |
|--------|---------------|
| lzma (baseline) | ~1.6 |
| zpaq | ~2.2 |
| Arithmetic + GPT | ~2.6-3.0 |
| Self-compressing NN | ~3.4 |

## Key Optimizations

1. **Data Layout**: Transpose tokens to group similar values together
2. **Prediction**: Use spatial/temporal context for better probability estimates
3. **Entropy Coding**: Arithmetic coding with accurate probability distributions
4. **Model Size**: Balance model accuracy vs. size (model goes in the zip!)

## Submission Checklist

- [ ] Compressed data files
- [ ] `decompress.py` script
- [ ] Any model weights/data needed for decompression
- [ ] Total zip size should be minimal!

