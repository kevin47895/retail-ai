"""
analytics.py — Router de Analítica Avanzada para Kovaq Retail Analytics.
Módulos: Heatmaps, Categorías, Horarios/Perfiles, Publicidad Dinámica.
Se monta en /api/analytics y no altera ningún endpoint existente.
"""
import asyncio

from fastapi import APIRouter, Request, HTTPException, Query

from database import (
    obtener_heatmap_hora, obtener_heatmap_dia_hora,
    obtener_interacciones_por_hora, obtener_top_productos,
    buscar_productos, obtener_categorias_con_productos,
    obtener_stats_por_franja, obtener_productos_por_franja,
    obtener_perfil_permanencia, obtener_tendencia_diaria,
    obtener_datos_publicidad
)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


# ---------- utilidad de autenticación (reutiliza main.obtener_usuario_actual) ----------
async def _auth(request: Request):
    from main import obtener_usuario_actual
    return await obtener_usuario_actual(request)


# ==========================================
# 1. HEATMAPS
# ==========================================
@router.get("/heatmap/hora")
async def heatmap_hora(request: Request, dias: int = Query(7, ge=1, le=90)):
    """Tráfico total por hora del día (0-23). Llena huecos con cero."""
    await _auth(request)
    data = await asyncio.to_thread(obtener_heatmap_hora, dias)
    horas = {str(i).zfill(2): 0 for i in range(24)}
    for row in data:
        horas[row[0]] = row[1]
    return {"labels": list(horas.keys()), "data": list(horas.values())}


@router.get("/heatmap/matriz")
async def heatmap_matriz(request: Request, dias: int = Query(30, ge=1, le=90)):
    """Matriz 7 días × 24 horas de tráfico con métricas ejecutivas e intuitivas."""
    await _auth(request)
    data = await asyncio.to_thread(obtener_heatmap_dia_hora, dias)
    matriz = [[0] * 24 for _ in range(7)]
    nombres_dias = ["Dom", "Lun", "Mar", "Mié", "Jue", "Vie", "Sáb"]
    
    total_semanal = 0
    max_hora_val = -1
    max_hora_dia = ""
    max_hora_txt = ""

    for row in data:
        dia, hora, total = int(row[0]), int(row[1]), int(row[2])
        matriz[dia][hora] = total
        total_semanal += total
        if total > max_hora_val:
            max_hora_val = total
            max_hora_dia = nombres_dias[dia]
            max_hora_txt = f"{hora:02d}:00"

    # Resumen por Franjas Comerciales (Mañana, Mediodía, Tarde, Noche)
    resumen_franjas = []
    totales_dia = []
    for d in range(7):
        m_val = sum(matriz[d][h] for h in range(6, 11))
        md_val = sum(matriz[d][h] for h in range(11, 14))
        t_val = sum(matriz[d][h] for h in range(14, 18))
        n_val = sum(matriz[d][h] for h in range(18, 23))
        mad_val = sum(matriz[d][h] for h in [23, 0, 1, 2, 3, 4, 5])
        tot_dia = sum(matriz[d])
        totales_dia.append(tot_dia)

        resumen_franjas.append({
            "dia": nombres_dias[d],
            "manana": m_val,
            "mediodia": md_val,
            "tarde": t_val,
            "noche": n_val,
            "madrugada": mad_val,
            "total": tot_dia
        })

    # Encontrar Día Pico
    max_dia_idx = max(range(7), key=lambda i: totales_dia[i]) if totales_dia else 0
    dia_pico = {
        "dia": nombres_dias[max_dia_idx],
        "total": totales_dia[max_dia_idx] if totales_dia else 0
    }

    hora_pico = {
        "hora": max_hora_txt or "20:00",
        "dia": max_hora_dia or "Miércoles",
        "total": max_hora_val if max_hora_val >= 0 else 0
    }

    return {
        "dias": nombres_dias,
        "horas": [f"{h:02d}:00" for h in range(24)],
        "matriz": matriz,
        "total_semanal": total_semanal,
        "dia_pico": dia_pico,
        "hora_pico": hora_pico,
        "resumen_franjas": resumen_franjas
    }


@router.get("/heatmap/interacciones")
async def heatmap_interacciones(request: Request, dias: int = Query(7, ge=1, le=90)):
    """Interacciones con productos agrupadas por hora."""
    await _auth(request)
    data = await asyncio.to_thread(obtener_interacciones_por_hora, dias)
    resultado = {}
    for hora, producto, total in data:
        resultado.setdefault(hora, []).append({"producto": producto, "total": total})
    return resultado


# ==========================================
# 2. CATEGORÍAS, BÚSQUEDA Y TOP
# ==========================================
@router.get("/productos/top")
async def top_productos(
    request: Request,
    limite: int = Query(10, ge=1, le=50),
    dias: int = Query(30, ge=1, le=90),
):
    await _auth(request)
    data = await asyncio.to_thread(obtener_top_productos, limite, dias)
    return {"productos": [{"nombre": p[0], "interacciones": p[1]} for p in data]}


@router.get("/productos/buscar")
async def buscar(request: Request, q: str = Query(..., min_length=1)):
    await _auth(request)
    data = await asyncio.to_thread(buscar_productos, q)
    return {"resultados": [{"nombre": p[0], "interacciones": p[1]} for p in data]}


@router.get("/categorias")
async def categorias(request: Request):
    await _auth(request)
    data = await asyncio.to_thread(obtener_categorias_con_productos)
    cats = {}
    for cat, prod, inter in data:
        cats.setdefault(cat, []).append({"producto": prod, "interacciones": inter})
    return {"categorias": cats}


# ==========================================
# 3. HORARIOS Y PERFILES
# ==========================================
@router.get("/horarios/franjas")
async def stats_franjas(request: Request, dias: int = Query(7, ge=1, le=90)):
    """KPIs por franja: mañana (6-12), tarde (12-18), noche (18-6)."""
    await _auth(request)
    data = await asyncio.to_thread(obtener_stats_por_franja, dias)
    franjas = {
        "manana": {"trafico": 0, "interacciones": 0, "permanencia": 0.0},
        "tarde":  {"trafico": 0, "interacciones": 0, "permanencia": 0.0},
        "noche":  {"trafico": 0, "interacciones": 0, "permanencia": 0.0},
    }
    for franja, tipo, total, prom in data:
        if franja not in franjas:
            continue
        if tipo == "trafico":
            franjas[franja]["trafico"] = total
        elif tipo == "interaccion":
            franjas[franja]["interacciones"] = total
        elif tipo == "permanencia":
            franjas[franja]["permanencia"] = round(float(prom), 1)
    return franjas


@router.get("/horarios/productos")
async def productos_por_franja(request: Request, dias: int = Query(7, ge=1, le=90)):
    await _auth(request)
    data = await asyncio.to_thread(obtener_productos_por_franja, dias)
    resultado = {"manana": [], "tarde": [], "noche": []}
    for franja, producto, interacciones in data:
        if franja in resultado:
            resultado[franja].append({"producto": producto, "interacciones": interacciones})
    return resultado


@router.get("/perfiles")
async def perfiles(request: Request, dias: int = Query(7, ge=1, le=90)):
    """Segmentación de clientes por tiempo de permanencia."""
    await _auth(request)
    data = await asyncio.to_thread(obtener_perfil_permanencia, dias)
    return {"perfiles": [{"tipo": p[0], "cantidad": p[1], "promedio_seg": p[2]} for p in data]}


@router.get("/tendencia")
async def tendencia(request: Request, dias: int = Query(14, ge=1, le=90)):
    """Tendencia diaria de métricas principales."""
    await _auth(request)
    data = await asyncio.to_thread(obtener_tendencia_diaria, dias)
    return {
        "fechas": [r[0] for r in data],
        "trafico": [r[1] for r in data],
        "interacciones": [r[2] for r in data],
        "permanencia": [r[3] or 0 for r in data],
    }


# ==========================================
# 4. PUBLICIDAD DINÁMICA
# ==========================================
@router.get("/publicidad")
async def publicidad(
    request: Request,
    franja: str = Query(None, regex="^(manana|tarde|noche)$"),
    dias: int = Query(7, ge=1, le=90),
):
    """Datos cruzados + sugerencias automáticas de pauta por franja horaria."""
    await _auth(request)
    data = await asyncio.to_thread(obtener_datos_publicidad, franja, dias)

    sugerencias = []
    if data["trafico"] > 0:
        total_inter = sum(p["interacciones"] for p in data["top_productos"])
        tasa = (total_inter / data["trafico"]) * 100 if total_inter else 0

        nombre_franja = {"manana": "mañana", "tarde": "tarde", "noche": "noche"}.get(
            data["franja"], "todo el día"
        )

        if tasa < 10:
            sugerencias.append({
                "tipo": "urgente",
                "icono": "bi-exclamation-triangle-fill",
                "titulo": f"Baja conversión en {nombre_franja}",
                "descripcion": (
                    f"Solo {tasa:.1f}% de conversión. "
                    "Sugerimos Reels cortos (<15 s) geolocalizados a 2 km del supermercado."
                ),
                "presupuesto": "$40-80 USD / 2 semanas",
            })

        if data["permanencia_promedio"] > 20 and tasa < 15:
            sugerencias.append({
                "tipo": "oportunidad",
                "icono": "bi-lightbulb-fill",
                "titulo": "Alta permanencia, baja conversión",
                "descripcion": (
                    f"Clientes permanecen {data['permanencia_promedio']}s pero no interactúan. "
                    "Cupón digital o promo en punto de venta."
                ),
                "presupuesto": "$30-60 USD / semana",
            })

        for prod in data["top_productos"][:3]:
            sugerencias.append({
                "tipo": "impulso",
                "icono": "bi-graph-up-arrow",
                "titulo": f"Potenciar: {prod['producto']}",
                "descripcion": (
                    f"{prod['interacciones']} interacciones registradas en {nombre_franja}. "
                    "Alto interés — ideal para campaña de Meta Ads con foto de percha."
                ),
                "presupuesto": "$25-50 USD / semana",
            })

        if data["permanencia_promedio"] < 5 and data["trafico"] > 20:
            sugerencias.append({
                "tipo": "alerta",
                "icono": "bi-clock-history",
                "titulo": "Tránsito rápido sin detención",
                "descripcion": (
                    "Los clientes pasan pero no se detienen. "
                    "Evaluar señalización, iluminación o reubicación de productos estrella."
                ),
                "presupuesto": "Inversión en PDV",
            })

    data["sugerencias"] = sugerencias
    return data
