"""Asistente Kovaq (Groq). Elige el modelo vigente solo, explica los errores y no se queda colgado."""
import asyncio
import logging
import os

from dotenv import load_dotenv
from groq import AsyncGroq

from database import obtener_estadisticas

load_dotenv()
log = logging.getLogger("kovaq.ia")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").replace('"', "").replace("'", "").strip()
client = AsyncGroq(api_key=GROQ_API_KEY, timeout=15.0, max_retries=0) if GROQ_API_KEY else None

# Groq retira modelos cada pocos meses (llama-3.1-8b-instant se apagó el 16/08/2026 para cuentas
# gratuitas). Por eso NO se fija un modelo: se pregunta a Groq cuáles hay y se usa el primero válido.
PREFERIDOS = [m for m in [os.getenv("GROQ_MODEL", "").strip(), "openai/gpt-oss-20b", "openai/gpt-oss-120b",
                          "qwen/qwen3.6-27b"] if m]
NO_CHAT = ("whisper", "tts", "guard", "safeguard", "orpheus", "compound", "embed")
_validos: list[str] | None = None
_descartados: set[str] = set()


async def _modelos() -> list[str]:
    global _validos
    if _validos is None:
        try:
            lista = await asyncio.wait_for(client.models.list(), timeout=8)
            ids = [m.id for m in lista.data if not any(x in m.id.lower() for x in NO_CHAT)]
            _validos = [m for m in PREFERIDOS if m in ids] + [i for i in ids if i not in PREFERIDOS]
            log.info("Modelos de chat disponibles en Groq: %s", _validos[:6])
        except Exception as e:  # sin lista se prueban los preferidos y se vuelve a intentar la próxima vez
            log.warning("No se pudo listar modelos de Groq: %s", e)
            return [m for m in PREFERIDOS if m not in _descartados]
    return [m for m in _validos if m not in _descartados][:4]


def explicar(e: Exception) -> tuple[str, str]:
    """(tipo, mensaje claro para el usuario)."""
    s, code, nombre = str(e).lower(), getattr(e, "status_code", None), type(e).__name__
    if code == 401 or "invalid api key" in s:
        return "clave", "La GROQ_API_KEY no es válida o fue revocada. Crea una nueva en console.groq.com/keys y ponla en .env."
    if code == 429 or "rate limit" in s:
        return "limite", "Groq pidió esperar (límite de uso). Intenta de nuevo en un minuto."
    if "decommission" in s or "model_not_found" in s or "does not exist" in s or code == 404:
        return "modelo", "Ese modelo ya no existe en Groq."
    if nombre in ("APIConnectionError", "APITimeoutError") or "timeout" in s or "connection" in s:
        return "red", "No hay conexión con Groq. Revisa tu internet."
    return "otro", f"Error de Groq: {str(e)[:160]}"


async def preguntar_a_los_datos(pregunta: str, vivo: dict | None = None) -> str:
    if not client:
        return "⚠️ Falta la GROQ_API_KEY en el archivo .env"
    d = await asyncio.to_thread(obtener_estadisticas)  # con caché: no espera a la base remota cada vez
    if not isinstance(d, dict) or "error" in d:
        d = {"total_personas": 0, "tiempo_promedio_percha": 0, "productos_tomados": []}
    vivo = vivo or {}
    contexto = (
        "Eres Kovaq, analista de retail en Ecuador. Responde en español, en máximo 3 líneas. "
        f"Datos guardados: {d.get('total_personas', 0)} entradas, permanencia promedio "
        f"{d.get('tiempo_promedio_percha', 0)} s, productos tomados {d.get('productos_tomados', [])}. "
        f"Cámara ahora: {vivo.get('entradas', 0)} personas detectadas y {vivo.get('agarrados', 0)} interacciones. "
        "Si los datos son cero, dilo y no inventes cifras."
    )
    ultimo = "No hay modelos disponibles en Groq."
    for modelo in await _modelos():
        try:
            r = await asyncio.wait_for(client.chat.completions.create(
                model=modelo, temperature=0.3, max_tokens=600,
                extra_body={"reasoning_effort": "low"},  # los modelos gpt-oss razonan: así gastan pocos tokens
                messages=[{"role": "system", "content": contexto}, {"role": "user", "content": pregunta}]), timeout=20)
            texto = (r.choices[0].message.content or "").strip()
            if texto:
                return texto
            ultimo = f"El modelo {modelo} no devolvió texto."
        except Exception as e:
            tipo, ultimo = explicar(e)
            log.warning("Falló el modelo %s (%s): %s", modelo, tipo, e)
            if tipo == "modelo":
                _descartados.add(modelo)  # no se vuelve a intentar con ese modelo
                continue
            if tipo in ("clave", "limite", "red"):
                return "⚠️ " + ultimo  # probar otro modelo no ayuda
    return "❌ " + ultimo
