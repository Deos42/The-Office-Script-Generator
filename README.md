# The Office Script Generator using LSTM

## Overview

This project implements a character-level LSTM language model in PyTorch to generate dialogue styled after _The Office_.

The model learns sequential dependencies directly at the character level and predicts the next character in an autoregressive manner.

This project demonstrates:

- Sequence modeling fundamentals
- Clean dataset construction
- Proper train/validation/test splitting (no data leakage)
- Stable LSTM training
- Autoregressive text generation

It reflects strong ML engineering discipline relevant to quantitative research, machine learning engineering, and applied AI roles.

---

## Problem Formulation

Given a sequence of characters:

(c1, c2, ..., ct)

The model learns to estimate:

P(c\_{t+1} | c1, c2, ..., ct)

This is a standard language modeling objective trained using cross-entropy loss.

---

## Model Architecture

The generator is a character-level recurrent neural network (RNN) with the following components:

- **Embedding Layer:** Maps characters to a 64-dimensional vector space.
- **Stacked LSTM:** Two LSTM layers with a hidden dimension of 256.
- **Dropout:** A rate of 0.2 to help prevent overfitting.
- **Linear Layer:** Projects the hidden state to the final vocabulary size (72) for character prediction.
- **Classification Layer:** Softmax on the last layer

---

## Data Preprocessing

### 1. Train / Validation / Test Split (Leakage-Free)

To prevent data leakage:

- Raw text is split contiguously
  - 90% Training
  - 10% Test
- Validation data is created as a contiguous subset of the training portion

This avoids the common mistake of randomly splitting overlapping sliding windows, which can leak near-identical sequences across splits.

---

### 2. Character-Level Tokenization

- Vocabulary built from training text only
- Each character mapped to integer ID
- Bidirectional encoding and decoding supported

This ensures strict evaluation integrity.

---

### 3. Sliding Window Dataset Construction

For a context length T:

- Input: text[i : i + T]
- Target: text[i + T]

This converts raw text into supervised learning samples suitable for autoregressive training.

---

### Forward Pass

1. Character IDs → Embeddings
2. Embeddings → LSTM
3. Final hidden state → Linear layer
4. Softmax → Next-character probability distribution

Loss Function: Cross-Entropy  
Optimizer: Adam

---

## Training Strategy

- Mini-batch gradient descent
- Shuffled training loader
- Non-shuffled validation/test loaders
- Cross-entropy monitoring
- Validation used to measure generalization

The training pipeline ensures:

- No train/validation overlap
- No window-based leakage
- Proper stochastic optimization

---

## Results & Performance

The model was trained for **5 epochs** using a context window of **120 characters**.

### Data Statistics

- **Total Characters Used:** 3,427,466
- **Vocabulary Size:** 72 characters
- **Context Length:** 120

### Training Performance

| Metric                      | Final Value |
| :-------------------------- | :---------- |
| **Training Loss**           | 1.1884      |
| **Validation Loss**         | 1.2758      |
| **Holdout Test Loss**       | 1.2678      |
| **Holdout Test Perplexity** | **3.553**   |

---

## Training Plots

The plots indicate a steady convergence without significant overfitting, as the validation loss follows the training loss closely throughout the 5 epochs.

---

## Conclusions

- **Learning Efficiency:** With a perplexity of ~3.553, the model effectively predicts the next character within the context of Dunder Mifflin dialogue.
- **Script Formatting:** The model successfully learned to mimic the script format (Character Name: Dialogue).
- **Future Improvements:** Further training or increasing the LSTM hidden layers could enhance the long-term logical flow of conversation.
