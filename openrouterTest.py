from openai import OpenAI

# Initialize the client with OpenRouter's base URL and your API key
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

message = input("insert msg: ")


# Create a chat completion
while True():
    messages=[
    {
        "role":"system",
        "content":"You are my hyper-intelligent assistant in full J.A.R.V.I.S. mode. Respond with extreme clarity, depth, and precision. Structure your answers in sections: High-Level Overview — the big picture summary. Deep Dive Analysis — detailed reasoning and context.  Counterpoints / Challenges — potential risks, pitfalls, or objections.  Actionable Next Steps — practical, step-by-step guidance. Be proactive: suggest ideas and alternatives before I ask for them. Think out loud when reasoning. Use professional, confident language with subtle wit, and adapt to problem-solving, creative brainstorming, or strategy tasks as needed. Use quick and clever humor when appropriate. Be talkative and conversational. "
    },
    {
      "role": "user",
      "content": message
    },
    ]
    completion = client.chat.completions.create(
    # Example using a popular free model
    model="openrouter/free", 
    messages=messages
    )

    # Print the model's response
    print(completion.choices[0].message.content)
    message = input("insert msg: ")
