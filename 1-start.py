import requests
import json

url="http://localhost:11434/api/generate"

data={
    "model": "llama3.2",
    "prompt": "2 lines on penguins"
    #,"stream": False
}

response= requests.post(url, json=data, stream=True)

if response.status_code == 200:
    print("generated text:", end=' ', flush=True)
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            # Attempt to parse the line as JSON
            try:
                json_line = json.loads(decoded_line)
                generated_text = json_line.get('response', '')
                print(generated_text, end='', flush=True)
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {decoded_line}")
else:
    print(f"Error: {response.status_code} - {response.text}")