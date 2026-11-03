import asyncio
import time

import ai_agent as ia

PREGUNTA = "¿Cuántas interacciones tuvo la leche Toni?"

async def main():
    t0 = time.time()
    d = await asyncio.to_thread(ia._obtener_datos_base)
    print(f"Lectura de BD: {time.time() - t0:.2f} s, {len(d['productos'])} productos")

    contexto = ia.SISTEMA_DATOS + "\n\n" + ia._bloque_datos(d, {})
    print(f"Tamaño del contexto: {len(contexto)} caracteres")

    for modelo in dict.fromkeys(ia.PREFERIDOS):  # sin repetidos
        t = time.time()
        try:
            await asyncio.wait_for(
                ia.client.chat.completions.create(
                    model=modelo,
                    temperature=0.2,
                    max_tokens=1000,
                    messages=[
                        {"role": "system", "content": contexto},
                        {"role": "user", "content": PREGUNTA},
                    ],
                ),
                timeout=30.0,
            )
            print(f"OK   {modelo:30} {time.time() - t:5.1f} s")
        except Exception as e:
            print(f"FAIL {modelo:30} {time.time() - t:5.1f} s  -> {type(e).__name__!r} {e!r}")

asyncio.run(main())