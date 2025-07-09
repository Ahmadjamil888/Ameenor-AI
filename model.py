import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, input_seqs):
        embedded = self.embedding(input_seqs)  # [B, T, D]
        outputs, (h_n, c_n) = self.lstm(embedded)
        h = torch.tanh(self.fc(torch.cat((h_n[0], h_n[1]), dim=1))).unsqueeze(0)  # [1, B, H]
        c = torch.tanh(self.fc(torch.cat((c_n[0], c_n[1]), dim=1))).unsqueeze(0)
        return outputs, (h, c)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)  # [B, 1, D]
        output, hidden = self.lstm(embedded, hidden)  # [B, 1, H]
        logits = self.fc(output.squeeze(1))  # [B, vocab_size]
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim)

    def forward(self, input_seqs, target_seqs=None, teacher_forcing_ratio=0.5, max_len=20):
        batch_size = input_seqs.size(0)
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, max_len, vocab_size).to(input_seqs.device)

        # Encode
        encoder_outputs, hidden = self.encoder(input_seqs)

        # First decoder input = <START> token (assume 2 is <START>)
        decoder_input = torch.full((batch_size, 1), 2, dtype=torch.long).to(input_seqs.device)

        for t in range(max_len):
            logits, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t, :] = logits

            if target_seqs is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target_seqs[:, t].unsqueeze(1)  # use actual next token
            else:
                decoder_input = logits.argmax(1).unsqueeze(1)  # use predicted token

        return outputs
