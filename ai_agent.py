import asyncio
import logging
import os
import time

from dotenv import load_dotenv
from groq import AsyncGroq

from database import leer, obtener_stats_por_franja, obtener_productos_por_franja, obtener_perfil_permanencia, obtener_datos_publicidad

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


def _obtener_contexto_avanzado() -> str:
    """Genera contexto enriquecido con franjas, perfiles y publicidad para el LLM."""
    partes = []
    try:
        franjas = {}
        raw = obtener_stats_por_franja(7)
        for franja, tipo, total, prom in raw:
            franjas.setdefault(franja, {})
            if tipo == "trafico":
                franjas[franja]["trafico"] = total
            elif tipo == "interaccion":
                franjas[franja]["interacciones"] = total
            elif tipo == "permanencia":
                franjas[franja]["permanencia"] = round(float(prom), 1)
        if franjas:
            resumen = "; ".join(
                f"{f.capitalize()}: {v.get('trafico',0)} personas, {v.get('interacciones',0)} interacciones, {v.get('permanencia',0)}s permanencia"
                for f, v in franjas.items()
            )
            partes.append(f"Análisis por franja (últimos 7 días): {resumen}.")
    except Exception:
        pass

    try:
        prods_franja = obtener_productos_por_franja(7)
        resumen_prods = {}
        for franja, producto, inter in prods_franja:
            resumen_prods.setdefault(franja, []).append(f"{producto}({inter})")
        if resumen_prods:
            txt = "; ".join(f"{f}: {', '.join(ps[:3])}" for f, ps in resumen_prods.items())
            partes.append(f"Productos top por franja: {txt}.")
    except Exception:
        pass

    try:
        perfiles = obtener_perfil_permanencia(7)
        if perfiles:
            txt = ", ".join(f"{p[0]}: {p[1]} casos ({p[2]}s promedio)" for p in perfiles)
            partes.append(f"Perfiles de cliente por permanencia: {txt}.")
    except Exception:
        pass

    try:
        pub = obtener_datos_publicidad(None, 7)
        if pub["top_productos"]:
            tasa = (sum(p["interacciones"] for p in pub["top_productos"]) / pub["trafico"] * 100) if pub["trafico"] else 0
            partes.append(
                f"Publicidad: Tasa de conversión global={tasa:.1f}%, "
                f"permanencia promedio={pub['permanencia_promedio']}s."
            )
    except Exception:
        pass

    return " ".join(partes)


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

    # Consultas sobre horarios / franjas
    if any(w in p for w in ["mañana", "manana", "tarde", "noche", "horario", "franja", "turno"]):
        try:
            franjas = {}
            raw = obtener_stats_por_franja(7)
            for franja, tipo, total, prom in raw:
                franjas.setdefault(franja, {})
                if tipo == "trafico":
                    franjas[franja]["trafico"] = total
                elif tipo == "interaccion":
                    franjas[franja]["interacciones"] = total
            if franjas:
                partes_f = []
                for f, v in franjas.items():
                    emoji = {"manana": "🌅", "tarde": "☀️", "noche": "🌙"}.get(f, "📊")
                    partes_f.append(f"{emoji} {f.capitalize()}: {v.get('trafico',0)} personas, {v.get('interacciones',0)} interacciones")
                return "📊 Actividad por franja horaria (últimos 7 días):\n" + "\n".join(partes_f)
        except Exception:
            pass

    # Consultas sobre perfiles de cliente
    if any(w in p for w in ["perfil", "tipo de cliente", "express", "explorador", "decisor", "comportamiento"]):
        try:
            perfiles = obtener_perfil_permanencia(7)
            if perfiles:
                partes_p = [f"• {p[0]}: {p[1]} clientes (promedio {p[2]}s)" for p in perfiles]
                return "👤 Perfiles de cliente por permanencia (últimos 7 días):\n" + "\n".join(partes_p)
        except Exception:
            pass

    # Consultas sobre publicidad o promociones
    if any(w in p for w in ["promoci", "publicidad", "campaña", "pauta", "ads", "reel", "meta"]):
        try:
            pub = obtener_datos_publicidad(None, 7)
            if pub["top_productos"]:
                prods = ", ".join(p["producto"] for p in pub["top_productos"][:3])
                return (
                    f"📢 Sugerencia publicitaria basada en datos reales:\n"
                    f"Los productos con más interacción son: {prods}.\n"
                    f"Permanencia promedio: {pub['permanencia_promedio']}s. "
                    f"Recomendación: campaña de Reels cortos enfocados en estos artículos."
                )
        except Exception:
            pass

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

    # 2. Contexto conversacional para la IA en Groq (ENRIQUECIDO con analítica avanzada)
    avanzado = await asyncio.to_thread(_obtener_contexto_avanzado)
    contexto = (
        "Eres Kovaq IA, un asistente experto en Retail Analytics, marketing in-store y publicidad digital (Meta Ads, Reels, Google Ads). "
        f"Datos actuales de las perchas: Tráfico={d['trafico']}, Permanencia={d['permanencia']}s, "
        f"Interacciones={d['interacciones']}, Top productos={', '.join(d['productos'])}.\n"
        f"{avanzado}\n"
        "Capacidades avanzadas: Puedes analizar por FRANJAS HORARIAS (mañana 6-12h, tarde 12-18h, noche 18-6h), "
        "PERFILES DE CLIENTE (Express <5s, Explorador 5-15s, Interesado 15-30s, Decisor 30s+), "
        "y generar SUGERENCIAS DE PUBLICIDAD DINÁMICA cruzando permanencia, conversión y franja horaria. "
        "Si el usuario pregunta qué promoción poner en una franja específica, genera una recomendación concreta "
        "con tipo de campaña, presupuesto estimado y producto sugerido.\n"
        f"Modo actual: {modo}. Responde de forma sumamente interactiva, amigable y directa en español (máximo 5 líneas), "
        "adaptándote a lo que el usuario pregunta. Usa emojis moderadamente."
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