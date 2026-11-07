import os
import libsql
from dotenv import load_dotenv
from passlib.context import CryptContext
import asyncio

load_dotenv()
TURSO_URL = os.getenv("TURSO_URL", "").strip()
TURSO_TOKEN = os.getenv("TURSO_TOKEN", "").strip()
DB_LOCAL = os.getenv("DB_LOCAL", "kovaq_local.db")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def _abrir():
    if TURSO_URL:
        # La conexión remota a Turso ya maneja su propia concurrencia
        return libsql.connect(database=TURSO_URL, auth_token=TURSO_TOKEN)
    
    # Configuración para la base de datos local (SQLite)
    conn = libsql.connect(database=DB_LOCAL)
    
    # Activa la lectura y escritura simultánea en procesos separados
    conn.execute("PRAGMA journal_mode=WAL;")
    # Relaja la sincronización estricta de disco para que YOLO no pierda FPS al hacer INSERTs
    conn.execute("PRAGMA synchronous=NORMAL;") 
    
    return conn

def obtener_rotacion(dias: int):
    """Consulta la rotación de productos según el período que pida el cliente."""
    query = """
        SELECT detalle as producto, COUNT(*) as interacciones 
        FROM eventos 
        WHERE tipo = 'interaccion' AND creado >= date('now', ?)
        GROUP BY detalle 
        ORDER BY interacciones ASC 
        LIMIT 5
    """
    # Se formatea el string para que SQLite entienda "-5 days", "-15 days", etc.
    modificador_fecha = f"-{dias} days"
    return leer(query, (modificador_fecha,))

def leer(query: str, params: tuple = (), uno: bool = False):
    conn = _abrir()
    try:
        cur = conn.execute(query, params)
        return cur.fetchone() if uno else cur.fetchall()
    finally:
        conn.close()

def escribir(query: str, params: tuple = ()):
    conn = _abrir()
    try:
        conn.execute(query, params)
        conn.commit()
    finally:
        conn.close()

def init_db_sync():
    conn = _abrir()
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS roles (id INTEGER PRIMARY KEY AUTOINCREMENT, nombre TEXT UNIQUE)")
        conn.execute("""CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            nombre_completo TEXT,
            rol_id INTEGER,
            sucursal_id INTEGER,
            estado TEXT DEFAULT 'activo',
            FOREIGN KEY(rol_id) REFERENCES roles(id))""")

        conn.execute("""CREATE TABLE IF NOT EXISTS eventos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tipo TEXT,
            detalle TEXT,
            valor REAL,
            creado TEXT DEFAULT CURRENT_TIMESTAMP)""")

        # Roles: INSERT OR IGNORE agrega los que falten aunque la tabla ya tenga datos
        for rol in ("admin", "admin_tecnico", "dueno", "encargado", "vendedor"):
            conn.execute("INSERT OR IGNORE INTO roles (nombre) VALUES (?)", (rol,))
        conn.commit()

        if conn.execute("SELECT COUNT(*) FROM usuarios").fetchone()[0] == 0:
            rol_admin = conn.execute("SELECT id FROM roles WHERE nombre = 'admin'").fetchone()[0]
            conn.execute(
                "INSERT INTO usuarios (username, password_hash, nombre_completo, rol_id, estado) VALUES (?, ?, ?, ?, 'activo')",
                ("admin", pwd_context.hash("KovaqSeguro2026"), "Administrador Maestro", rol_admin)
            )
            conn.commit()
    finally:
        conn.close()

def guardar_trafico():
    escribir("INSERT INTO eventos (tipo) VALUES ('trafico')")

def guardar_interaccion(producto: str):
    escribir("INSERT INTO eventos (tipo, detalle) VALUES ('interaccion', ?)", (producto,))

def guardar_producto_tomado(producto: str, cliente_id: int = None):
    detalle = f"{producto} (Cliente #{cliente_id})" if cliente_id else producto
    escribir("INSERT INTO eventos (tipo, detalle, valor) VALUES ('producto_tomado', ?, ?)", (detalle, float(cliente_id or 0)))
    escribir("INSERT INTO eventos (tipo, detalle) VALUES ('interaccion', ?)", (producto,))

def guardar_producto_devuelto(producto: str, cliente_id: int = None):
    detalle = f"{producto} (Cliente #{cliente_id})" if cliente_id else producto
    escribir("INSERT INTO eventos (tipo, detalle, valor) VALUES ('producto_devuelto', ?, ?)", (detalle, float(cliente_id or 0)))

def guardar_permanencia(segundos: float):
    escribir("INSERT INTO eventos (tipo, valor) VALUES ('permanencia', ?)", (segundos,))

async def init_db():
    await asyncio.to_thread(init_db_sync)


# ==========================================
# MÓDULO DE ANALÍTICA AVANZADA (v2)
# ==========================================

def init_analytics_sync():
    """Inicializa tablas para categorización de productos. No altera tablas existentes."""
    conn = _abrir()
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS categorias (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT UNIQUE)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS producto_categoria (
            producto TEXT,
            categoria_id INTEGER,
            FOREIGN KEY(categoria_id) REFERENCES categorias(id),
            UNIQUE(producto, categoria_id))""")
        cats = {
            "Bebidas": ["bottle", "cup", "wine glass"],
            "Frutas": ["banana", "apple", "orange"],
            "Electrónica": ["cell phone", "laptop", "remote"],
            "Snacks": ["cake", "donut", "sandwich", "hot dog", "pizza"]
        }
        for cat, prods in cats.items():
            conn.execute("INSERT OR IGNORE INTO categorias (nombre) VALUES (?)", (cat,))
            row = conn.execute("SELECT id FROM categorias WHERE nombre = ?", (cat,)).fetchone()
            if row:
                for p in prods:
                    conn.execute(
                        "INSERT OR IGNORE INTO producto_categoria (producto, categoria_id) VALUES (?, ?)",
                        (p, row[0])
                    )
        conn.commit()
    finally:
        conn.close()

async def init_analytics():
    await asyncio.to_thread(init_analytics_sync)


# --- Heatmap: tráfico por hora del día ---
def obtener_heatmap_hora(dias=7):
    return leer("""
        SELECT strftime('%H', creado) as hora, COUNT(*) as total
        FROM eventos WHERE tipo = 'trafico' AND creado >= datetime('now', ?)
        GROUP BY hora ORDER BY hora
    """, (f"-{dias} days",))


# --- Heatmap: matriz día-de-semana × hora ---
def obtener_heatmap_dia_hora(dias=30):
    return leer("""
        SELECT strftime('%w', creado) as dia, strftime('%H', creado) as hora, COUNT(*) as total
        FROM eventos WHERE tipo = 'trafico' AND creado >= datetime('now', ?)
        GROUP BY dia, hora
    """, (f"-{dias} days",))


# --- Interacciones agrupadas por hora y producto ---
def obtener_interacciones_por_hora(dias=7):
    return leer("""
        SELECT strftime('%H', creado) as hora, detalle as producto, COUNT(*) as total
        FROM eventos WHERE tipo = 'interaccion' AND creado >= datetime('now', ?)
        GROUP BY hora, producto ORDER BY hora, total DESC
    """, (f"-{dias} days",))


# --- Top productos por cantidad de interacciones ---
def obtener_top_productos(limite=10, dias=30):
    return leer("""
        SELECT detalle as producto, COUNT(*) as interacciones
        FROM eventos WHERE tipo = 'interaccion' AND creado >= datetime('now', ?)
        GROUP BY producto ORDER BY interacciones DESC LIMIT ?
    """, (f"-{dias} days", limite))


# --- Búsqueda de productos por nombre parcial ---
def buscar_productos(termino):
    return leer("""
        SELECT detalle as producto, COUNT(*) as interacciones
        FROM eventos WHERE tipo = 'interaccion' AND LOWER(detalle) LIKE LOWER(?)
        GROUP BY producto ORDER BY interacciones DESC
    """, (f"%{termino}%",))


# --- Categorías con productos y conteo ---
def obtener_categorias_con_productos():
    return leer("""
        SELECT c.nombre as categoria, pc.producto,
               COALESCE(cnt.total, 0) as interacciones
        FROM categorias c
        LEFT JOIN producto_categoria pc ON c.id = pc.categoria_id
        LEFT JOIN (
            SELECT detalle, COUNT(*) as total
            FROM eventos WHERE tipo = 'interaccion' GROUP BY detalle
        ) cnt ON pc.producto = cnt.detalle
        ORDER BY c.nombre, interacciones DESC
    """)


# --- Estadísticas segmentadas por franja horaria ---
def obtener_stats_por_franja(dias=7):
    return leer("""
        SELECT
            CASE
                WHEN CAST(strftime('%H', creado) AS INTEGER) BETWEEN 6 AND 11  THEN 'manana'
                WHEN CAST(strftime('%H', creado) AS INTEGER) BETWEEN 12 AND 17 THEN 'tarde'
                ELSE 'noche'
            END as franja,
            tipo,
            COUNT(*) as total,
            COALESCE(AVG(CASE WHEN tipo = 'permanencia' THEN valor END), 0) as prom
        FROM eventos WHERE creado >= datetime('now', ?)
        GROUP BY franja, tipo
    """, (f"-{dias} days",))


# --- Productos más interactuados por franja ---
def obtener_productos_por_franja(dias=7):
    return leer("""
        SELECT
            CASE
                WHEN CAST(strftime('%H', creado) AS INTEGER) BETWEEN 6 AND 11  THEN 'manana'
                WHEN CAST(strftime('%H', creado) AS INTEGER) BETWEEN 12 AND 17 THEN 'tarde'
                ELSE 'noche'
            END as franja,
            detalle as producto, COUNT(*) as interacciones
        FROM eventos
        WHERE tipo = 'interaccion' AND creado >= datetime('now', ?)
        GROUP BY franja, producto ORDER BY franja, interacciones DESC
    """, (f"-{dias} days",))


# --- Perfiles de comportamiento basados en permanencia ---
def obtener_perfil_permanencia(dias=7):
    return leer("""
        SELECT
            CASE
                WHEN valor < 5  THEN 'Express (<5s)'
                WHEN valor < 15 THEN 'Explorador (5-15s)'
                WHEN valor < 30 THEN 'Interesado (15-30s)'
                ELSE 'Decisor (30s+)'
            END as perfil,
            COUNT(*) as cantidad,
            ROUND(AVG(valor), 1) as promedio_seg
        FROM eventos
        WHERE tipo = 'permanencia' AND creado >= datetime('now', ?)
        GROUP BY perfil ORDER BY promedio_seg
    """, (f"-{dias} days",))


# --- Tendencia diaria (tráfico, interacciones, permanencia) ---
def obtener_tendencia_diaria(dias=14):
    return leer("""
        SELECT date(creado) as fecha,
            SUM(CASE WHEN tipo = 'trafico'     THEN 1 ELSE 0 END) as trafico,
            SUM(CASE WHEN tipo = 'interaccion'  THEN 1 ELSE 0 END) as interacciones,
            ROUND(AVG(CASE WHEN tipo = 'permanencia' THEN valor END), 1) as prom
        FROM eventos WHERE creado >= datetime('now', ?)
        GROUP BY fecha ORDER BY fecha
    """, (f"-{dias} days",))


# --- Datos cruzados para publicidad dinámica ---
def obtener_datos_publicidad(franja=None, dias=7):
    mod = f"-{dias} days"
    franja_sql = {
        'manana': "AND CAST(strftime('%H', creado) AS INTEGER) BETWEEN 6 AND 11",
        'tarde':  "AND CAST(strftime('%H', creado) AS INTEGER) BETWEEN 12 AND 17",
        'noche':  "AND (CAST(strftime('%H', creado) AS INTEGER) >= 18 OR CAST(strftime('%H', creado) AS INTEGER) < 6)"
    }
    extra = franja_sql.get(franja, "")

    trafico = leer(
        f"SELECT COUNT(*) FROM eventos WHERE tipo = 'trafico' AND creado >= datetime('now', ?) {extra}",
        (mod,), uno=True
    )[0]
    perm = leer(
        f"SELECT COALESCE(AVG(valor),0), COUNT(*) FROM eventos WHERE tipo = 'permanencia' AND creado >= datetime('now', ?) {extra}",
        (mod,), uno=True
    )
    top = leer(
        f"SELECT detalle, COUNT(*) FROM eventos WHERE tipo = 'interaccion' AND creado >= datetime('now', ?) {extra} GROUP BY detalle ORDER BY COUNT(*) DESC LIMIT 5",
        (mod,)
    )
    return {
        "trafico": trafico,
        "permanencia_promedio": round(float(perm[0]), 1),
        "permanencia_eventos": perm[1],
        "top_productos": [{"producto": p[0], "interacciones": p[1]} for p in top] if top else [],
        "franja": franja or "todas"
    }
