import requests

# Hugging Face API configuration
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
HEADERS = {"Authorization": "Bearer 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"}

def generate_response(context, query):
    input_text = f"Context: {context}\nQuestion: {query}"
    payload = {"inputs": input_text, "parameters": {"max_length": 512, "num_return_sequences": 1}}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()[0]['generated_text']
