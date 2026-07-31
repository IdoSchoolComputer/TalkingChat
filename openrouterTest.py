from openai import OpenAI

# Initialize the client with OpenRouter's base URL and your API key
client = OpenAI(
  base_url="https://openrouter.ai",
  api_key="your_openrouter_api_key_here",
)

# Create a chat completion
completion = client.chat.completions.create(
  # Example using a popular free model
  model="meta-llama/llama-3-8b-instruct:free", 
  messages=[
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ]
)

# Print the model's response
print(completion.choices[0].message.content)
