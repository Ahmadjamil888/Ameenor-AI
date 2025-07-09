# OPEN SOURCE AI MODEL - AMEENOR AI

Ameenor AI is an open-source sequence-to-sequence chatbot model implemented in PyTorch. It is designed for learning purposes and can be fine-tuned for a variety of NLP tasks. This repository provides all necessary files including model architecture, preprocessing, training loop, and inference API.

---

##  Repository Structure

| File/Folder         | Description |
|---------------------|-------------|
| `model/`            | Contains trained model weights, tokenizer, and data files |
| `train.py`          | Training script for Seq2Seq chatbot model |
| `api.py`            | Flask API to serve the trained chatbot model |
| `chatbot.py`        | Inference and response generation logic |
| `dataset.py`        | Dataset class for tokenized pairs |
| `preprocess.py`     | Tokenization and data cleaning utilities |
| `test_client.py`    | Local testing script for interacting with API |
| `README.md`         | This documentation file |

---

##  Model Architecture

The model follows a standard Seq2Seq architecture using LSTM layers in both encoder and decoder components.

### Encoder:
- Embedding Layer: Converts token indices to dense vectors
- Bidirectional LSTM Layer
- Linear layer to compress hidden state

### Decoder:
- Embedding Layer
- LSTM Layer
- Linear Layer (output layer)
- Softmax over vocabulary

```python
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        ...
```

---

##  Dataset Format

The dataset is expected to be a list of input-output sentence pairs, saved as `model/pairs.pkl`:
```python
[("hello", "hi"), ("how are you?", "i'm fine, thank you"), ...]
```

The `tokenizer.pkl` should contain:
- `word2index`: Dictionary of word → index
- `index2word`: Dictionary of index → word

---

##  Training Results

| Epoch | Average Loss |
|-------|---------------|
| 1     | 6.8489        |
| 2     | 5.1603        |
| 3     | 4.9830        |

**Final Accuracy Estimate:** ~73.6% (measured using token-wise comparison on small validation set)

Note: These metrics were achieved using only 500 sentence pairs for debugging purposes. Full dataset training is expected to significantly improve performance.

---

##  Training the Model

### Step 1: Prepare Dataset

Ensure you have:
- `model/pairs.pkl` with list of (input, response) pairs
- `model/tokenizer.pkl` containing `word2index`, `index2word`

### Step 2: Edit `train.py` if needed

```bash
nano train.py
```

Adjust the following variables to match your needs:
```python
BATCH_SIZE = 32
EPOCHS = 10
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
MAX_LEN = 20
```

### Step 3: Train the Model

Run the training script:
```bash
python train.py
```

After training, the model weights will be saved to:
```
model/chatbot_model.pt
```

---

##  Running the API

### Step 1: Start the Flask Server
```bash
python api.py
```

Server will start at:
```
http://127.0.0.1:5005
```

### Step 2: Send POST Requests

Use a tool like Postman or `curl`:
```bash
curl -X POST http://127.0.0.1:5005/chat -H "Content-Type: application/json" -d "{"message": "hello"}"
```

---

##  Testing Locally

```bash
python test_client.py
```

Example interaction:
```
You: hello
Bot: hi there
You: how are you?
Bot: i am good, thank you
```

---

##  Dependencies

Install required packages:

```bash
pip install torch flask numpy tqdm
```

Ensure Python 3.10+ and PyTorch (CPU or CUDA) is available.

---

##  License

This project is released under the MIT License. You may use, modify, and distribute it freely for educational or research purposes.

---

##  Contact

Project Maintainer: Ahmad Jamil  
GitHub: [https://github.com/Ahmadjamil888](https://github.com/Ahmadjamil888)
