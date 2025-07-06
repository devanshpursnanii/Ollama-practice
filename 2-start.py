import ollama

response = ollama.list()
#print(response)

res= ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "2 lines on penguins"}
    ],
    stream=False  # Set to True if you want streaming responses
)

#print(res["message"]["content"])

#print(ollama.show("llama3.2"))  # Show model details


res=ollama.generate(
    model="llama3.2",
    prompt="2 lines on penguins",
    format="json",
    stream=False  # Set to True if you want streaming responses
)   
print(res["response"])