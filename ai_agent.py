"""Asistente Kovaq IA con puente intermediario y sintaxis limpia."""
import asyncio
import logging
import os
import time

from dotenv import load_dotenv
from groq import AsyncGroq

from database import leer

load_dotenv()
log = logging.getLogger("kovaq.ia")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").replace('"', "").replace("'", "").strip()
client = AsyncGroq(api_key=GROQ_API_KEY, timeout=5.0, max_retries=0) if GROQ_API_KEY else None

PREFERIDOS = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
MODOS = ("datos", "recomendacion")
TTL_DATOS = 60
_cache_datos: dict = {"t": 0.0, "d": None}


def _obtener_datos_base() -> dict:
    ahora = time.time()
    if _cache_datos["d"] is not None and ahora - _cache_datos["t"] < TTL_DATOS:
        return _cache_datos["d"]
    try:
        fila = leer("""SELECT
                COALESCE(SUM(CASE WHEN tipo = 'trafico' THEN 1 ELSE 0 END), 0),
                COALESCE(AVG(CASE WHEN tipo = 'permanencia' THEN valor END), 0),
                COALESCE(SUM(CASE WHEN tipo = 'interaccion' THEN 1 ELSE 0 END), 0)
            FROM eventos""", uno=True)
        
        top = leer("""SELECT detalle, COUNT(*) FROM eventos WHERE tipo = 'interaccion'
                    GROUP BY detalle ORDER BY COUNT(*) DESC LIMIT 3""")
        
        d = {
            "trafico": fila[0],
            "permanencia": round(float(fila[1]), 1),
            "interacciones": fila[2],
            "productos": [p[0] for p in top] if top else ["ninguno registrado"],
        }
    except Exception as e:
        log.error("Error leyendo BD: %r", e)
        d = {"trafico": 0, "permanencia": 0.0, "interacciones": 0, "productos": ["ninguno"]}

    _cache_datos["t"], _cache_datos["d"] = ahora, d
    return d


def puente_analisis_intermedio(pregunta: str, modo: str, d: dict) -> str:
    """
    Puente intermediario: analiza los datos localmente antes de pasarlos a la IA
    o responde de inmediato si detecta intenciones comunes.
    """
    p = pregunta.lower()

    if "venta" in p or "caja" in p or "dinero" in p:
        return "📊 Kovaq mide tráfico, permanencia e interacciones visuales en percha, no transiciones de caja ni ventas directas."

    if modo == "datos" and ("resumen" in p or "todo" in p or "como va" in p):
        return (
            f"📈 Resumen actual del supermercado:\n"
            f"- Tráfico total: {d['trafico']} personas detectadas.\n"
            f"- Permanencia promedio: {d['permanencia']} segundos frente a percha.\n"
            f"- Interacciones físicas: {d['interacciones']} contactos.\n"
            f"- Productos destacados: {', '.join(d['productos'])}."
        )

    if modo == "recomendacion" and d['interacciones'] == 0:
        return (
            "📢 Sugerencia de Marketing (Modo Preventivo):\n"
            "Como aún hay pocas interacciones registradas en las perchas principales, "
            "te recomiendo colocar cartelería de alto contraste o códigos QR llamativos en los pasillos de mayor tráfico para elevar el tiempo de permanencia."
        )

    return None


async def preguntar_a_los_datos(pregunta: str, modo: str = "datos", vivo: dict | None = None) -> str:
    if not client:
        return "⚠️ Falta configurar la GROQ_API_KEY en el archivo .env"

    if modo not in MODOS:
        modo = "datos"

    # 1. Obtenemos los datos limpios y cacheados
    d = await asyncio.to_thread(_obtener_datos_base)

    # 2. PUENTE INTERMEDIARIO (Respuesta instantánea sin llamadas lentas)
    respuesta_directa = puente_analisis_intermedio(pregunta, modo, d)
    if respuesta_directa:
        return respuesta_directa

    # 3. Contexto ultrarreducido para la IA
    contexto = (
        f"Datos percha: Tráfico={d['trafico']}, Permanencia={d['permanencia']}s, "
        f"Interacciones={d['interacciones']}, Top={', '.join(d['productos'])}.\n"
        f"Modo: {modo}. Responde en español en máximo 2 líneas de forma muy directa."
    )

    try:
        r = await asyncio.wait_for(
            client.chat.completions.create(
                model=PREFERIDOS[0],
                temperature=0.1,
                max_tokens=150,
                messages=[
                    {"role": "system", "content": contexto},
                    {"role": "user", "content": pregunta},
                ],
            ),
            timeout=4.0,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        if modo == "recomendacion":
            return f"📢 Sugerencia basada en datos actuales ({d['interacciones']} interacciones, {d['permanencia']}s de permanencia): Optimiza la iluminación y el etiquetado frontal en los productos con menor rotación."
        return f"📊 Cifras actuales: {d['trafico']} personas de tráfico, {d['permanencia']}s de permanencia promedio y {d['interacciones']} interacciones registradas."