import openai

def translate(text,client, target_lang,):
    prompt = f"""Translate the following text to {target_lang}.
    make sure to notice the context of words and translate them correctly    
    Text: {text}
    Return only the translation, nothing else."""
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b:free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content