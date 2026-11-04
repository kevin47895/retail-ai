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

# Modelos oficiales estables y rápidos de Groq
PREFERIDOS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
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
    Puente tolerante a errores de escritura para capturar consultas comunes
    de forma inmediata e interactiva.
    """
    p = pregunta.lower()

    if any(w in p for w in ["venta", "caja", "dinero", "factura"]):
        return "📊 Kovaq mide tráfico, permanencia e interacciones visuales en percha en tiempo real, no transiciones directas de caja."

    if any(w in p for w in ["resumen", "todo", "como va", "estado", "general"]):
        return (
            f"📈 Estado actual en piso de ventas:\n"
            f"- Tráfico detectado: {d['trafico']} personas.\n"
            f"- Retención media: {d['permanencia']} segundos.\n"
            f"- Interacciones físicas: {d['interacciones']} contactos.\n"
            f"- Top productos en percha: {', '.join(d['productos'])}."
        )

    # Detecta variaciones o errores tipográficos como "prodcutos", "rotado", "dias", etc.
    if any(w in p for w in ["product", "prodcut", "rotad", "movimient", "sku", "articulo"]):
        top_str = ", ".join(d['productos'])
        return f"📦 Analizando el flujo de la percha, los artículos con mayor interacción física son: {top_str}. ¿Te gustaría que diseñe una campaña digital o de Meta Ads enfocada en ellos?"

    return None


async def preguntar_a_los_datos(pregunta: str, modo: str = "datos", vivo: dict | None = None) -> str:
    if not client:
        return "⚠️ Falta configurar la GROQ_API_KEY en el archivo .env"

    if modo not in MODOS:
        modo = "datos"

    d = await asyncio.to_thread(_obtener_datos_base)

    # 1. Puente inteligente para respuestas rápidas y conversacionales
    respuesta_directa = puente_analisis_intermedio(pregunta, modo, d)
    if respuesta_directa:
        return respuesta_directa

    # 2. Contexto conversacional para la IA en Groq
    contexto = (
        f"Eres Kovaq IA, un asistente experto en Retail Analytics y marketing in-store. "
        f"Datos actuales de las perchas: Tráfico={d['trafico']}, Permanencia={d['permanencia']}s, "
        f"Interacciones={d['interacciones']}, Top productos={', '.join(d['productos'])}.\n"
        f"Modo actual: {modo}. Responde de forma sumamente interactiva, amigable y directa en español (máximo 3 líneas), adaptándote a lo que el usuario pregunta."
    )

    for modelo in PREFERIDOS:
        try:
            r = await asyncio.wait_for(
                client.chat.completions.create(
                    model=modelo,
                    temperature=0.4,
                    max_tokens=200,
                    messages=[
                        {"role": "system", "content": contexto},
                        {"role": "user", "content": pregunta},
                    ],
                ),
                timeout=4.0,
            )
            texto_respuesta = (r.choices[0].message.content or "").strip()
            if texto_respuesta:
                return texto_respuesta
        except Exception as err:
            log.warning("Fallo con el modelo %s: %r", modelo, err)
            continue

    # 3. Fallback dinámico y contextual si la API tarda o falla
    if modo == "recomendacion":
        return f"💡 Sugerencia comercial basada en los datos actuales ({d['interacciones']} interacciones, {d['permanencia']}s de retención): Te sugiero destacar con cartelería o pauta digital los artículos de mayor interés como {', '.join(d['productos'])}."
    
    return f"📊 Cifras al momento: Tenemos un tráfico de {d['trafico']} personas, una permanencia promedio de {d['permanencia']}s y {d['interacciones']} interacciones registradas en percha."