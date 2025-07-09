from datasets import load_dataset
import re
import pickle
from collections import Counter
import os
from tokenizer_wrapper import Tokenizer

# Constants
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<START>", "<END>"]
MAX_VOCAB_SIZE = 10000
MAX_SEQ_LEN = 20

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def build_vocab(pairs):
    counter = Counter()
    for input_text, target_text in pairs:
        counter.update(tokenize(input_text))
        counter.update(tokenize(target_text))

    most_common = counter.most_common(MAX_VOCAB_SIZE - len(SPECIAL_TOKENS))
    words = SPECIAL_TOKENS + [word for word, _ in most_common]

    word2index = {word: idx for idx, word in enumerate(words)}
    index2word = {idx: word for word, idx in word2index.items()}

    return word2index, index2word

def extract_pairs():
    dataset = load_dataset("roskoN/dailydialog", split="train")
    pairs = []

    for item in dataset:
        utterances = item['utterances']
        if len(utterances) >= 2:
            for i in range(len(utterances) - 1):
                input_sentence = utterances[i]
                response_sentence = utterances[i + 1]
                pairs.append((input_sentence, response_sentence))

    return pairs

def save_data(pairs, word2index, index2word):
    os.makedirs("model", exist_ok=True)

    tokenizer = Tokenizer(word2index, index2word, len(word2index))

    with open("model/pairs.pkl", "wb") as f:
        pickle.dump(pairs, f)

    with open("model/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"✅ Saved {len(pairs)} pairs and vocab of size {len(word2index)}.")

if __name__ == "__main__":
    print("📥 Extracting pairs...")
    pairs = extract_pairs()

    print("🔤 Building vocabulary...")
    word2index, index2word = build_vocab(pairs)

    print("💾 Saving processed data...")
    save_data(pairs, word2index, index2word)
