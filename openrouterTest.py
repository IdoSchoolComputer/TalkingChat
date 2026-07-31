from openai import OpenAI

# Initialize the client with OpenRouter's base URL and your API key
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
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
