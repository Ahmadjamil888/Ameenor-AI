class Tokenizer:
    def __init__(self, word2index, index2word, vocab_size):
        self.word2index = word2index
        self.index2word = index2word
        self.vocab_size = vocab_size
        self.tags = []
        self.responses = {}
