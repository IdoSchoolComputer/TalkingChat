import openai
import edge_tts
import asyncio

def translate(text,client, target_lang,):
    prompt = f"""Translate the following text to {target_lang}.
    make sure to notice the context of words and translate them correctly    
    Text: {text}
    Return only the translation, nothing else."""
    try:
        response = client.chat.completions.create(
            model="google/gemma-4-31b-it:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
    except openai.RateLimitError:
        response = client.chat.completions.create(
            model="inclusionai/ling-3.0-flash:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

    return response.choices[0].message.content
async def speak():
    communicate = edge_tts.Communicate("You're offline, so I'm running locally now — things might feel a little slower, especially on a laptop without a dedicated GPU.", "en-US-BrianMultilingualNeural")
    await communicate.save("OfflineWarning.mp3")

asyncio.run(speak())