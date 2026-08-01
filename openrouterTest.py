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
import pygame
import asyncio
import socket
import pyttsx3
pygame.mixer.init()
import subprocess
import sys
import shutil
import time

def check_internet(host="8.8.8.8", port=53, timeout=3):
    """
    Returns True if internet is available, False otherwise.
    8.8.8.8 is Google's public DNS server.
    Port 53 is the standard port for DNS traffic.
    """
    try:
        # Create a socket connection
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except OSError:
        return False

device = "cuda" if torch.cuda.is_available() else "cpu" 

async def speak(content:str):
    if check_internet():
        # Uses the highly realistic 'Brian' neural voice
        communicate = edge_tts.Communicate(content, "en-US-BrianMultilingualNeural")
        await communicate.save("output.mp3")
        words = pygame.mixer.Sound("output.mp3")
        words.play()
        os.remove("output.mp3")
    elif not check_internet():
        engine = pyttsx3.init()
        # Plays directly out of your speakers without saving a file
        engine.say(content)
        engine.runAndWait()
"""need to add ollama, for when theres no intenet"""

def OllamaModelDownload(model_name: str):
    # --- STEP 0: LOCATE THE OLLAMA EXECUTABLE DIRECTLY ---
    ollama_exe = shutil.which("ollama")
    if ollama_exe is None:
        raise FileNotFoundError("Could not find 'ollama' on PATH. Is Ollama installed?")
    
    # --- STEP 1: CHECK IF THE MODEL IS ALREADY INSTALLED ---
    
    check_process = subprocess.run(
        [ollama_exe, "list"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    
    if model_name in check_process.stdout or f"{model_name}:latest" in check_process.stdout:
        print(f"Model '{model_name}' is already installed! Skipping download.")
        return
        
        
    # --- STEP 2: DOWNLOAD THE MODEL (IF NOT INSTALLED) ---
    
    print(f"🚀 Starting download for model: {model_name}...\n")
    
    # KEY CHANGE: we no longer set stdout=PIPE / stderr=PIPE. Ollama's progress
    # bar uses real ANSI terminal control codes (cursor movement, line erase,
    # "synchronized update" mode) to redraw itself smoothly — this is exactly
    # what a real terminal is built to interpret natively. By NOT capturing
    # the output, Ollama inherits our actual console window directly, detects
    # it's a real terminal, and draws its own progress bar perfectly — no
    # manual parsing/reimplementing needed on our end at all.
    result = subprocess.run([ollama_exe, "pull", model_name])
    
    return_code = result.returncode
    
    
    # --- STEP 3: FINAL STATUS AND SANITY CHECK ---
    
    if return_code == 0:
        print(f"\n\n✅ Success: {model_name} has been completely downloaded! \n Quitting...")
    else:
        print(f"\n\n❌ Error: Failed to pull {model_name}.\nExit code: {return_code} \n Quitting...")

def trigger_quit():
    with open("memory.json", "w") as file:
        json_messages = list(messages)
        json.dump(json_messages, file)
    os._exit(67)

keyboard.add_hotkey('ctrl+4', trigger_quit)
print("Press Ctrl+4 at any time to quit. or type 'quit pls' in input \n")
print("\nIn order for this service to work offline, we need to download a couple of models. \nType 'offline prep' in order to download the needed models")


# Initialize the client with OpenRouter's base URL and your API key
if check_internet():
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
    )
    textLLM = "nvidia/nemotron-3-ultra-550b-a55b:free"
else:
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
    if device=="cuda":
        print("device is cuda")
        textLLM = "qwen3:8b"
    elif device == "cpu":
        print("device is cpu")
        textLLM = "phi4-mini"
        
def GPUcheck():
    if device=="cuda":
        print("device has cuda GPU")
        OllamaModelDownload("qwen3:8b")
        trigger_quit()

    elif device == "cpu":
        print("device doesn't have cuda GPU")
        OllamaModelDownload("phi4-mini")
        trigger_quit()
    

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

def to_api_messages(msgs):
    return [{"role": m["role"], "content": m["content"]} for m in msgs]
# Create a chat completion
while True:
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    message = input("insert msg: ")
    if message == "quit pls":
        trigger_quit()
    if message == "offline prep":
        GPUcheck()
    messages.append({"role":"user","content":now+" "+message,"time":now})
    messagesFinal = [sysPrompts, *to_api_messages(messages)]
    completion = client.chat.completions.create(
    model=textLLM,
    # in case i get rate limited 
    # model="openroute/free",
    messages=messagesFinal
    )

    console = Console()

    before_mdtxt= completion.choices[0].message.content
    console.print(Markdown(before_mdtxt))
    asyncio.run(speak(before_mdtxt))

    messages.append({"role":"assistant","content":now+" "+before_mdtxt,"time":now})
    # print(messages)
