import torch
from torch.utils.data import Dataset
import re

class ChatDataset(Dataset):
    def __init__(self, pairs, word2index, max_len=20):
        self.pairs = pairs
        self.word2index = word2index
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def tokenize(self, sentence):
        return re.findall(r"\b\w+\b", sentence.lower())  # 🔄 Replaces nltk.word_tokenize

    def encode(self, sentence):
        tokens = self.tokenize(sentence)
        ids = [self.word2index.get("<START>", 2)]  # start token

        for token in tokens:
            ids.append(self.word2index.get(token, self.word2index.get("<UNK>", 1)))

        ids.append(self.word2index.get("<END>", 3))  # end token

        # Pad/truncate
        if len(ids) < self.max_len:
            ids += [self.word2index.get("<PAD>", 0)] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        input_text, target_text = self.pairs[idx]
        input_encoded = self.encode(input_text)
        target_encoded = self.encode(target_text)
        return input_encoded, target_encoded
