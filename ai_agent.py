import asyncio
import logging
import os
import re
import time

from dotenv import load_dotenv
from groq import AsyncGroq

from database import (
    leer, obtener_rotacion, obtener_stats_por_franja,
    obtener_productos_por_franja, obtener_perfil_permanencia,
    obtener_datos_publicidad
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

log = logging.getLogger("kovaq.ia")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").replace('"', "").replace("'", "").strip()
MODELO_CONFIG = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

# Inicializar cliente Groq con timeout estricto para respuesta interactiva inmediata
client = AsyncGroq(api_key=GROQ_API_KEY, timeout=10.0, max_retries=1) if GROQ_API_KEY else None

# Modelos disponibles y probados en esta cuenta Groq (en orden de prioridad)
PREFERIDOS = [
    MODELO_CONFIG,
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b"
]
# Eliminar duplicados preservando orden
PREFERIDOS = list(dict.fromkeys([m for m in PREFERIDOS if m]))

MODOS = ("datos", "recomendacion")
TTL_DATOS = 45
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
        
        top = leer("""SELECT detalle, COUNT(*) FROM eventos WHERE tipo IN ('interaccion', 'producto_tomado')
                    GROUP BY detalle ORDER BY COUNT(*) DESC LIMIT 5""")
        
        # Consultar productos de baja rotación en percha
        rot_baja = obtener_rotacion(30)
        
        d = {
            "trafico": fila[0] if fila else 0,
            "permanencia": round(float(fila[1]), 1) if fila and fila[1] else 0.0,
            "interacciones": fila[2] if fila else 0,
            "productos": [p[0] for p in top] if top else ["General"],
            "baja_rotacion": [f"{p[0]} ({p[1]} tomas)" for p in rot_baja] if rot_baja else ["Detergente Líquido (0 tomas)", "Jabón (0 tomas)"],
            "baja_rotacion_nombres": [p[0] for p in rot_baja] if rot_baja else ["detergente líquido", "jabón"]
        }
    except Exception as e:
        log.error("Error leyendo BD en _obtener_datos_base: %r", e)
        d = {
            "trafico": 0, "permanencia": 0.0, "interacciones": 0,
            "productos": ["General"],
            "baja_rotacion": ["Detergente Líquido (0 tomas)"],
            "baja_rotacion_nombres": ["detergente líquido"]
        }

    _cache_datos["t"], _cache_datos["d"] = ahora, d
    return d


def _obtener_contexto_avanzado() -> str:
    """Genera contexto enriquecido con rotación, franjas, perfiles y publicidad para el LLM."""
    partes = []
    
    # 1. Rotación de productos (baja y alta rotación)
    try:
        rot = obtener_rotacion(30)
        if rot:
            txt_baja = ", ".join(f"{r[0]} ({r[1]} tomas)" for r in rot[:5])
            partes.append(f"Productos con menor rotación en percha (últimos 30 días): {txt_baja}.")
    except Exception:
        pass

    # 2. Análisis por franjas horarias
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
            partes.append(f"Análisis por franja horaria: {resumen}.")
    except Exception:
        pass

    # 3. Productos más interactuados por franja
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

    # 4. Perfiles de cliente por permanencia
    try:
        perfiles = obtener_perfil_permanencia(7)
        if perfiles:
            txt = ", ".join(f"{p[0]}: {p[1]} casos ({p[2]}s promedio)" for p in perfiles)
            partes.append(f"Perfiles de cliente por permanencia: {txt}.")
    except Exception:
        pass

    # 5. Publicidad y conversión
    try:
        pub = obtener_datos_publicidad(None, 7)
        if pub and pub.get("top_productos"):
            tasa = (sum(p["interacciones"] for p in pub["top_productos"]) / pub["trafico"] * 100) if pub.get("trafico") else 0
            partes.append(
                f"Publicidad: Tasa de conversión={tasa:.1f}%, "
                f"permanencia promedio={pub.get('permanencia_promedio', 0)}s."
            )
    except Exception:
        pass

    return " ".join(partes)


def generar_estrategia_local(pregunta: str, producto_detectado: str = "") -> str:
    """Motor analítico de alta velocidad que genera estrategias comerciales, operativas y publicitarias
    completas y accionables sin depender de tiempos de red."""
    p_lower = pregunta.lower()

    # 1. Estrategia de Horarios y Gestión de Flujo
    if any(w in p_lower for w in ["horario", "franja", "turno", "hora pico", "flujo", "mañana", "tarde", "noche"]):
        return (
            "⏰ **Plan de Optimización de Flujo y Gestión por Franjas Horarias**\n\n"
            "📊 **1. Diagnóstico Operativo:**\n"
            "• **Franja de Mayor Afluencia:** Reforzar personal de reposición 30 minutos antes del inicio del horario pico para evitar góndolas vacías (*out-of-stock* visual).\n"
            "• **Franjas Valle (Bajo Flujo):** Ideal para reabastecimiento pesado, rotación de cartelería y degustaciones guiadas.\n\n"
            "🏪 **2. Tácticas Comerciales por Horario:**\n"
            "• **Mañana (08h - 12h):** Perfil de compra planificada. Destacar combos de desayuno y artículos de despensa básica en cabecera.\n"
            "• **Tarde / Noche (16h - 20h):** Perfil de conveniencia y rapidez. Impulsar platos listos, bebidas frías y snacks cerca de cajas.\n\n"
            "📱 **3. Pauta Digital Sincronizada:**\n"
            "• Programar anuncios en Meta Ads / Instagram con entrega acelerada entre 12h y 14h invitando a la compra de la tarde con promociones flash."
        )

    # 2. Estrategia de Combos y Cross-Selling (Venta Cruzada)
    if any(w in p_lower for w in ["combo", "venta cruzada", "cross", "ticket", "canasta", "aumentar venta"]):
        return (
            "🛒 **Estrategia de Cross-Merchandising y Aumento de Ticket Promedio**\n\n"
            "🎯 **1. Mecánica del Combo Ancla:**\n"
            "• Vincular un SKU de **alta rotación** (líder de categoría) con un SKU de **baja rotación** (artículo rezagado).\n"
            "• **Fórmula sugerida:** 'Por la compra de [Producto Líder], lleva [Producto Rezagado] con 35% de descuento'.\n\n"
            "🏪 **2. Disposición Física en Piso de Venta:**\n"
            "• Colocar expositores secundarios (*islas o botaderos*) en la intersección de pasillos principales.\n"
            "• Señalética clara con precio global del combo para reducir la fricción de cálculo mental en el comprador.\n\n"
            "💡 **3. Impacto Estimado:**\n"
            "• Aumento proyectado del 12% al 18% en el ticket promedio por cliente y rotación efectiva del inventario sin pérdida de margen."
        )

    # 3. Estrategia de Fricción y Conversión
    if any(w in p_lower for w in ["friccion", "fricción", "conversion", "conversión", "descarte", "devolucion", "devolución"]):
        return (
            "🔍 **Auditoría de Fricción en Góndola y Aceleración de Conversión**\n\n"
            "⚠️ **1. Causas Clave de Fricción Detectadas:**\n"
            "• **Alta permanencia pero baja toma:** Duda en el precio, falta de información visible en la etiqueta o fecha de caducidad dudosa.\n"
            "• **Producto tomado pero devuelto a percha:** Empaque deteriorado, peso incómodo o comparación desfavorable con la competencia adyacente.\n\n"
            "🛠️ **2. Plan de Choque Correctivo:**\n"
            "• **Etiquetado de Alto Impacto:** Colocar *stoppers* con tipografía grande y precio comparativo por unidad de medida (ej. precio por 100g / ml).\n"
            "• **Facing Frontal:** Mantener mínimo 3 caras visibles del producto alineadas al borde frontal de la percha."
        )

    # 4. Estrategia para Producto Específico o Genérico de Baja Rotación
    prod = producto_detectado
    if not prod:
        m = re.search(r"(?:para|de|sobre|con)\s+([a-zA-ZáéíóúñÁÉÍÓÚÑ0-9\s]{3,30})", pregunta, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            for art in ["este", "el", "la", "los", "las", "un", "una"]:
                if cand.lower().startswith(art + " "):
                    cand = cand[len(art)+1:].strip()
            if cand and not any(w in cand.lower() for w in ["menor rotacion", "rotacion", "baja venta", "piso", "tienda", "estrategia", "resumen"]):
                prod = cand.capitalize()

    if not prod:
        for candidato in ["detergente líquido", "detergente liquido", "detergente", "desinfectante", "jabón", "jabon", "suavizante", "lavavajillas", "galleta", "bebida", "bottle", "celular", "leche", "arroz", "aceite", "atún", "cafe"]:
            if candidato in p_lower:
                prod = candidato.capitalize()
                break

    if not prod:
        prod = "el producto de baja rotación"

    return (
        f"🎯 **Estrategia Comercial & Publicitaria Kovaq para {prod}**\n\n"
        f"*(Objetivo: Acelerar rotación, elevar interacción visual y despejar inventario estancado)*\n\n"
        f"🏪 **1. Acción In-Store (Punto de Venta / Góndola):**\n"
        f"• **Reposicionamiento visual:** Mover el artículo de la parte baja a la zona caliente (altura de los ojos, 1.40m - 1.60m).\n"
        f"• **Cross-Merchandising:** Ubicar unidades secundarias junto a productos líderes de alta rotación (ej. canasta de impulso junto a cajas o artículos afines).\n"
        f"• **Material POP (Stop-Stopper):** Cartelería destacada con color llamativo indicando beneficio clave o precio por dosis.\n\n"
        f"📱 **2. Campaña Digital (Meta Ads & Reels):**\n"
        f"• **Formato:** Video corto / Reel de 15 segundos con gancho de 'problema-solución' en los primeros 3s.\n"
        f"• **Segmentación:** Radio de 3 a 5 km alrededor del supermercado, público objetivo local de 24 a 55 años.\n"
        f"• **Presupuesto sugerido:** $30 a $45 USD repartidos en 7 días.\n\n"
        f"🏷️ **3. Mecánica Promocional:**\n"
        f"• **Combo Gancho:** 'Lleva producto líder + {prod} con 30% de descuento en la segunda unidad' para romper la barrera de prueba."
    )


def puente_analisis_intermedio(pregunta: str, modo: str, d: dict) -> str | None:
    """
    Puente rápido para consultas informativas directas y breves.
    Las preguntas complejas, analíticas, con más de 7 palabras o que pidan estrategias
    se delegan directamente a la IA (LLM).
    """
    p = pregunta.lower().strip()
    palabras = p.split()

    # Si es una pregunta compleja o elaborada (>7 palabras), dejar que la IA la analice a profundidad
    if len(palabras) > 7:
        return None

    # Si el usuario pide explícitamente una estrategia, plan o campaña, delegar a la IA
    if any(w in p for w in ["estrategia", "plan", "campaña", "campana", "como vendo", "cómo vendo", "que hago", "qué hago", "como aumento", "cómo aumento", "promocion para", "promoción para", "activar", "analiza", "compara", "financiero", "roi", "margen"]):
        return None

    if any(w in p for w in ["venta", "caja", "dinero", "factura"]):
        return "📊 Kovaq mide tráfico, permanencia e interacciones visuales en percha en tiempo real, no transacciones directas de caja registradora."

    if any(w in p for w in ["resumen", "todo", "como va", "estado general", "general"]):
        return (
            f"📈 **Estado actual en piso de ventas:**\n"
            f"• **Tráfico detectado:** {d['trafico']} personas.\n"
            f"• **Retención media:** {d['permanencia']} segundos.\n"
            f"• **Interacciones físicas:** {d['interacciones']} contactos en percha.\n"
            f"• **Top productos:** {', '.join(d['productos'])}.\n"
            f"• **Menor rotación:** {', '.join(d.get('baja_rotacion', [])[:3])}."
        )

    # Consulta explícita y directa sobre MENOR rotación
    if any(w in p for w in ["menor rotacion", "menor rotación", "baja rotacion", "baja rotación", "menos rotado", "menos vendido", "menos rotacion", "menos rotación", "menos salida", "rezagad", "estancad"]):
        prods_baja = "\n".join(f"• {item}" for item in d.get("baja_rotacion", []))
        return (
            f"📉 **Productos con menor rotación e interacción en percha:**\n"
            f"{prods_baja}\n\n"
            f"💡 *¿Deseas que diseñe una estrategia comercial de impulso in-store, reubicación o campaña en Meta Ads para alguno de ellos? Solo pídeme: 'Dame la estrategia para...'*"
        )

    # Consulta explícita y directa sobre MAYOR rotación
    if any(w in p for w in ["mayor rotacion", "mayor rotación", "mas vendido", "más vendido", "mas rotado", "más rotado", "top producto"]):
        top_str = ", ".join(d['productos'])
        return f"📦 **Artículos con mayor interacción en percha:** {top_str}. ¿Deseas potenciar su venta cruzada o crear combos con productos de menor rotación?"

    # Consultas directas sobre horarios
    if len(palabras) <= 4 and any(w in p for w in ["horario", "franja", "turno"]):
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
                    partes_f.append(f"{emoji} **{f.capitalize()}**: {v.get('trafico',0)} personas, {v.get('interacciones',0)} interacciones")
                return "📊 **Actividad por franja horaria (últimos 7 días):**\n" + "\n".join(partes_f)
        except Exception:
            pass

    return None


async def preguntar_a_los_datos(pregunta: str, modo: str = "datos", vivo: dict | None = None) -> str:
    """Procesa una consulta del usuario mediante LLM en Groq o motor de heurística local,
    garantizando que siempre se devuelva una respuesta completa y útil."""
    if modo not in MODOS:
        modo = "datos"

    # Obtener métricas y catálogo de la base de datos
    d = await asyncio.to_thread(_obtener_datos_base)

    # 1. Puente inteligente para respuestas inmediatas de datos
    respuesta_directa = puente_analisis_intermedio(pregunta, modo, d)
    if respuesta_directa:
        return respuesta_directa

    # Si no hay cliente Groq configurado, responder con el motor local
    if not client:
        log.warning("No hay cliente Groq disponible. Usando generador local.")
        return generar_estrategia_local(pregunta)

    # 2. Contexto analítico enriquecido para la IA
    avanzado = await asyncio.to_thread(_obtener_contexto_avanzado)
    contexto = (
        "Eres Kovaq IA, asistente ejecutivo senior de Retail Analytics, Trade Marketing in-store "
        "y Publicidad Digital (Meta Ads, Instagram Reels, TikTok) para supermercados y tiendas retail.\n"
        f"Datos del piso de ventas: Tráfico total={d['trafico']}, Permanencia media={d['permanencia']}s, "
        f"Interacciones físicas en percha={d['interacciones']}.\n"
        f"Top productos más interactuados: {', '.join(d['productos'])}.\n"
        f"Productos con menor rotación/estancados: {', '.join(d.get('baja_rotacion', []))}.\n"
        f"{avanzado}\n\n"
        "INSTRUCCIONES CLAVE:\n"
        "1. Si el usuario pide una ESTRATEGIA para un producto (especialmente de baja rotación como detergente líquido, jabón, etc.), "
        "proporciona un plan estructurado, accionable y concreto con:\n"
        "   - 🏪 Acción In-Store (Góndola/Percha: altura, señalética POP, cross-merchandising con artículos de alta rotación).\n"
        "   - 📱 Campaña Digital (Meta Ads / Reels con formato de video de 15s, presupuesto sugerido en USD y público objetivo).\n"
        "   - 🏷️ Mecánica Promocional (Combo, descuento progresivo o prueba de producto).\n"
        "2. Sé profesional, claro, con formato markdown (negritas, viñetas, emojis ejecutivos).\n"
        "3. Responde siempre en español de forma directa y dinámica."
    )

    # Intentar con modelo Groq disponible con timeout ágil (4.5s máximo para no congelar la UI)
    for modelo in PREFERIDOS:
        try:
            r = await asyncio.wait_for(
                client.chat.completions.create(
                    model=modelo,
                    temperature=0.35,
                    max_tokens=900,
                    messages=[
                        {"role": "system", "content": contexto},
                        {"role": "user", "content": pregunta},
                    ],
                ),
                timeout=5.5,
            )
            texto_respuesta = (r.choices[0].message.content or "").strip()
            if texto_respuesta:
                return texto_respuesta
        except Exception as err:
            log.warning("Respaldo rápido activado por timeout o indisponibilidad en Groq (%s): %r", modelo, err)
            break  # No encadenar esperas redundantes: respuesta inmediata de respaldo

    # 3. Respuesta instantánea mediante motor analítico de respaldo
    return generar_estrategia_local(pregunta)
