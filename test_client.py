import requests

while True:
    msg = input("You: ")
    if msg.lower() == "exit":
        break

    response = requests.post("http://127.0.0.1:5005/chat", json={"message": msg})
    print("Bot:", response.json().get("response", "Error in response"))
