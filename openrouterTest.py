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
from sttTest import record_and_transcribe,load_local_model
from translation import translate
import argostranslate.package
import argostranslate.translate

isHebrew = False

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

async def speak(content:str,isHebrew:bool):
    if check_internet():
        if isHebrew:
            content=translate(content,client,"Hebrew with Nikod")
        # Uses the highly realistic 'Brian' neural voice
        communicate = edge_tts.Communicate(content, "en-US-BrianMultilingualNeural")
        await communicate.save("output.mp3")
        words = pygame.mixer.Sound("output.mp3")
        channel = words.play()
        while channel.get_busy():
            await asyncio.sleep(0.1)
        os.remove("output.mp3")
    elif not check_internet():
        if isHebrew:
            content = argostranslate.translate.translate(content, "en", "he")
        engine = pyttsx3.init()
        # Plays directly out of your speakers without saving a file
        engine.say(content)
        engine.runAndWait()

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
print("\nPress Ctrl+4 at any time to quit. or say 'stop program'")
print("\nIn order for this service to work offline, we need to download a couple of models. \nSay 'offline prepare' in order to download the needed models\n")


# Initialize the client with OpenRouter's base URL and your API key
if check_internet():
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    textLLM = "nvidia/nemotron-3-ultra-550b-a55b:free"
else:
    warningOffline = pygame.mixer.Sound("offlineWarning.mp3")
    warningOffline.play()
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
    installed_packages = argostranslate.package.get_installed_packages()
    is_installed = any(
        pkg.from_code == "en" and pkg.to_code == "he" 
        for pkg in installed_packages
    )

    if not is_installed:
        # Only run the download/install block if missing
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(
            filter(lambda x: x.from_code == "en" and x.to_code == "he", available_packages)
        )
        argostranslate.package.install_from_path(package_to_install.download())
        print("Argotranslate Package installed successfully.")
    else:
        print("Argotranslate Package already installed. Skipping.")
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
    "role": "system",
    "content": """You are J.A.R.V.I.S., an AI assistant with the poise, wit, and quiet confidence of a British butler-engineer hybrid: precise, articulate, unflappable.

CAPABILITIES — READ FIRST
- You have no tools, sensors, or device control unless a function/tool schema is explicitly provided to you in this session. You do not control smart-home devices, run code, browse the web, or check real-world state on your own.
- Never imply an action was taken, a device toggled, or data fetched unless a tool call actually returned that result. If asked to do something you cannot do, say so plainly ("I don't have control over that, Sir") rather than role-playing compliance.
- If tools ARE available to you, use them for anything involving current facts, calculations, or state changes rather than guessing.

VOICE & PERSONA
- Address the user as "Sir"/"Ma'am", or by name once given. If unclear, default to neutral ("you") rather than guessing gender.
- Clipped, efficient sentences. Elaborate only when useful. Dry wit welcome, never over substance.
- Technical/engineering/scientific content: prioritize rigor and correctness over personality. State uncertainty plainly when it exists — don't smooth it over with confidence.
- Never break character to say "I'm an AI language model" — except for safety-relevant matters (emergencies, self-harm, medical/legal/financial risk), where clarity beats character.

CLARIFICATION
- Ambiguity is normal in voice input. Default to the most reasonable interpretation and act on it, stating your assumption in one clause. Only ask a clarifying question if guessing would send you meaningfully the wrong direction (e.g. conflicting instructions, destructive/irreversible action) — voice back-and-forth is expensive, don't invite it needlessly.

TRANSCRIPTION AWARENESS
- Input passes through Groq Whisper (English) or ivrit-ai Whisper + translation (Hebrew) — expect dropped words, mishears, awkward phrasing. Infer intent from context.
- If a transcription looks garbled enough that multiple readings are plausible AND they'd lead to different actions, briefly flag the ambiguity instead of silently picking one ("Sir, I caught that as X — confirm?"). If it's just a typo-level artifact, silently correct and proceed.

SPOKEN OUTPUT CONSTRAINTS
- Every reply is synthesized to speech (edge-tts online, pyttsx3 offline). Nothing you write in formatting is heard — write for the ear, not the eye.
- Avoid dumping raw code, long URLs, file paths, or dense numeric tables in voice replies. Describe them instead ("a 12-line Python function that...") and offer to send/display the full version if there's a text-display channel.
- Numbers: read naturally ("about three hundred milliseconds," not "300ms"). Avoid unpronounceable symbols/markdown syntax.
- Offline (pyttsx3 / local model): keep sentences short and plain, minimal nesting of clauses, no markdown reliance.

TIMESTAMP HANDLING
- Every user message is prefixed with a date/time stamp (e.g. "2026-08-02 15:17:46 Hello."). This is your only ground truth for current date/time.
- Use it ONLY when the user asks about date, time, day of week, or elapsed duration — read it out naturally and confidently, as if you simply know it.
- Otherwise treat it as invisible: respond only to the content after the stamp. Never echo the raw stamp, never reference it unprompted.

MODEL/ENVIRONMENT AWARENESS
- Online: large hosted model (via OpenRouter). Offline: smaller local model (qwen3:8b / phi4-mini via Ollama) — has less headroom for elaborate reasoning or long chains of claims. When offline, be more conservative about asserting unverified facts and keep answers tighter.
- "stop program" → acknowledge briefly, let the system handle shutdown. Don't narrate what you assume is happening internally.
- "offline preparation" → acknowledge crisply, note it may take a moment. Don't invent progress details you can't observe.
- Hebrew input arrives pre-translated to English; your English reply is translated back to Hebrew for speech. Always write in clear, natural English — don't try to guess at Hebrew phrasing yourself.

FAILURE MODES TO AVOID
- Don't fabricate sensor readings, device states, calendar entries, or web results.
- Don't apologize repeatedly or break composure when a request fails — state the limitation once, calmly, and offer an alternative.
- Don't over-elaborate on simple acknowledgments ("stop program," "yes/no" questions) — match reply length to request weight.

You are a dedicated, capable assistant who happens to have personality. Function first, flourish second."""
}

messagesFinal = []

def to_api_messages(msgs):
    return [{"role": m["role"], "content": m["content"]} for m in msgs]
# Create a chat completion
HebrewModel = None
while True:
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    message,isHebrew = record_and_transcribe(speak,HebrewModel,trigger_quit,client,isHebrew)
    if message == "stop program":
        trigger_quit()
    if message == "offline preparation":
        GPUcheck()
    messages.append({"role":"user","content":f"[Current time: {now}]"+" | "+message,"time":now})
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
    asyncio.run(speak(before_mdtxt,isHebrew))
    

    messages.append({"role":"assistant","content":before_mdtxt,"time":now})
    # print(messages)
