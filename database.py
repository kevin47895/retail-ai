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
        return libsql.connect(database=TURSO_URL, auth_token=TURSO_TOKEN)
    return libsql.connect(database=DB_LOCAL)

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

def guardar_permanencia(segundos: float):
    escribir("INSERT INTO eventos (tipo, valor) VALUES ('permanencia', ?)", (segundos,))      
async def init_db():
    await asyncio.to_thread(init_db_sync)