import requests

response = requests.post("http://127.0.0.1:5005/chat", json={"message": "Hello"})
print("Bot:", response.json())
