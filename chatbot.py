import torch
import torch.nn.functional as F
import re

# === Basic tokenizer ===
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# === Convert tokens to indices ===
def encode_input(tokens, word2index, max_len=20):
    unk_idx = word2index.get("<UNK>", 1)
    pad_idx = word2index.get("<PAD>", 0)
    indices = [word2index.get(token, unk_idx) for token in tokens]

    if len(indices) < max_len:
        indices += [pad_idx] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices

# === Convert indices to tokens ===
def decode_output(indices, index2word):
    tokens = []
    for idx in indices:
        word = index2word.get(idx, "")
        if word == "<EOS>":
            break
        if word not in ["<PAD>", "<SOS>"]:
            tokens.append(word)
    return tokens

# === Generate chatbot response ===
def generate_response(model, tokenizer, user_input, max_len=20):
    try:
        # Load vocab
        word2index = tokenizer.word2index
        index2word = tokenizer.index2word

        # Encode input
        tokens = tokenize(user_input)
        input_indices = encode_input(tokens, word2index, max_len)
        input_tensor = torch.tensor([input_indices], dtype=torch.long)

        # Generate output
        model.eval()
        with torch.no_grad():
            output = model(input_tensor, None, teacher_forcing_ratio=0.0, max_len=max_len)
            predicted_indices = output.argmax(dim=-1).squeeze(0).tolist()

        # Decode output
        response_tokens = decode_output(predicted_indices, index2word)
        return " ".join(response_tokens) if response_tokens else "..."

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"
