import ollama
import os

model="llama3.2"

input_path="./list.txt"
output_path="./categorised.txt"

if not os.path.exists(input_path):
    exit(f"Input file {input_path} does not exist.")

with open(input_path, "r") as file:
    lines = file.read().strip()

prompt=f"""

You are a categoriser that categorises and sorts grocery items. 
You will be given a list of items, each on a new line:

{lines}

please do the following tasks:
1. Categorise each item into one of the following categories: dairy, meat, fish, fruit, vegetables, bakery etc
2. Sort the items within each category alphabetically.
3. present the categorised items in a clear and structured format using bullet points.
"""

try:
    response=ollama.generate(model=model, prompt=prompt, stream=False)
    text=response.get("response", "")
    
    with open(output_path, "w") as f:
        f.write(text.strip())

    print(f"Categorised items saved to {output_path}")

except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)

