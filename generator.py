from transformers import pipeline

# Load a fine-tuned language model
generator = pipeline("text2text-generation", model="google/flan-t5-large")

def generate_response(context, query):
    input_text = f"Context: {context}\nQuestion: {query}"
    response = generator(input_text, max_length=512, num_return_sequences=1)
    return response[0]['generated_text']
