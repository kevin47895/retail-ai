import asyncio
import os
import time

from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY", "").replace('"', "").replace("'", "").strip()
client = AsyncGroq(api_key=api_key, timeout=60)

MODELOS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "allam-2-7b",
]

async def probar(modelo: str):
    t = time.time()
    try:
        r = await client.chat.completions.create(
            model=modelo,
            messages=[{"role": "user", "content": "Responde solo con la palabra: hola"}],
            max_tokens=300,
            temperature=0,
        )
        texto = (r.choices[0].message.content or "").strip()
        print(f"OK   {modelo:30} {time.time() - t:5.1f} s  -> {texto[:60]!r}")
    except Exception as e:
        print(f"FAIL {modelo:30} {time.time() - t:5.1f} s  -> {type(e).__name__}: {str(e)[:120]}")

async def main():
    for m in MODELOS:
        await probar(m)

asyncio.run(main())