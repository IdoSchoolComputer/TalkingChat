from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
import keyboard
import os
from collections import deque
import json
import numpy as np
import torch
import edge_tts
from playsound3 import playsound
import asyncio

device = "cuda" if torch.cuda.is_available() else "cpu" 

async def speak(content:str):
    # Uses the highly realistic 'Brian' neural voice
    communicate = edge_tts.Communicate(content, "en-US-BrianMultilingualNeural")
    await communicate.save("output.mp3")
    playsound("output.mp3")
    os.remove("output.mp3")


def trigger_quit():
    with open("memory.json", "w") as file:
        json.dump(messages, file)
    os._exit(67)

keyboard.add_hotkey('ctrl+4', trigger_quit)

print("Press Ctrl+4 at any time to quit. or type 'quit pls' in input ")
# Initialize the client with OpenRouter's base URL and your API key
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)
message = ""
messages = deque(maxlen=40)
try:
    with open("memory.json", "r") as file:
        messages = json.load(file)
except json.JSONDecodeError:
    messages = deque(maxlen=40)
except FileNotFoundError: 
    with open("memory.json", "x") as file:
        pass
sysPrompts = {
        "role":"system",
        "content":"""You are J.A.R.V.I.S., a highly advanced AI assistant. You speak with the poise, wit, and quiet confidence of a British butler-engineer hybrid: precise, articulate, and unflappable, even when handling chaos.

Core traits:
- Address the user as "Sir" or "Ma'am" (or their name, if given) unless told otherwise.
- Speak in clipped, efficient sentences. No filler, no rambling. Get to the point, then elaborate only if useful.
- Maintain dry, understated wit. A touch of sarcasm is welcome, but never at the expense of usefulness.
- Stay calm and composed regardless of the situation — urgency is conveyed through word choice, not panic.
- Be proactive: anticipate follow-up needs, flag risks, and offer next steps without being asked.
- When giving technical, scientific, or engineering answers, be rigorous and accurate — precision matters more than personality here.
- Never break character to explain that you are an AI language model unless explicitly asked.
- If a request is ambiguous, ask one crisp clarifying question rather than guessing broadly.
- Use structured formatting (short lists, steps) when it aids clarity, but avoid unnecessary headers or bloat.
- Default to concise answers. Expand only when the topic warrants depth.

You are not a generic chatbot — you are a dedicated, capable assistant who happens to have personality. Function first, flourish second."""
    }

messagesFinal = []
# Create a chat completion
while True:
    message = input("insert msg: ")
    if message == "quit pls":
        trigger_quit()
    messages.append({"role":"user","content":message})
    messagesFinal = [sysPrompts,*messages]
    completion = client.chat.completions.create(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    # in case i get rate limited 
    # model="openroute/free",
    messages=messagesFinal
    )

    console = Console()

    # Print the model's response
    before_mdtxt= completion.choices[0].message.content
    console.print(Markdown(before_mdtxt))
    asyncio.run(speak(before_mdtxt))

    messages.append({"role":"assistant","content":before_mdtxt})
    # print(messages)
