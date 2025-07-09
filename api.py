import os
import torch
import pickle
import traceback
from flask import Flask, request, jsonify

from model import Seq2Seq
from chatbot import generate_response

app = Flask(__name__)

# === Constants ===
MODEL_PATH = "model/chatbot_model.pt"
TOKENIZER_PATH = "model/tokenizer.pkl"

# === Model hyperparameters (must match train.py) ===
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
MAX_LEN = 20  # Optional: useful for inference control

# === Load Tokenizer ===
try:
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    print("✅ Tokenizer loaded successfully.")
except Exception as e:
    traceback.print_exc()
    raise RuntimeError(f"❌ Failed to load tokenizer: {e}")

# === Initialize & Load Model ===
try:
    word2index = tokenizer.word2index
    vocab_size = len(word2index)

    model = Seq2Seq(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    traceback.print_exc()
    raise RuntimeError(f"❌ Failed to load model: {e}")

# === Routes ===

@app.route("/", methods=["GET"])
def home():
    return "✅ Chatbot API is running."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        response = generate_response(model, tokenizer, user_input)
        return jsonify({"response": response})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Internal Error: {str(e)}"}), 500

# === Run Flask App ===
if __name__ == "__main__":
    print("🚀 Chatbot API running at http://127.0.0.1:5005")
    app.run(host="127.0.0.1", port=5005, debug=True)
