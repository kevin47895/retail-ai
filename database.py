"""Base de datos (Turso o archivo local). Conexiones reutilizadas, escritura por lotes y caché de lectura."""
import os
import queue
import threading
import time

import libsql
from dotenv import load_dotenv
from passlib.context import CryptContext

load_dotenv()
TURSO_URL = os.getenv("TURSO_URL", "").strip()
TURSO_TOKEN = os.getenv("TURSO_TOKEN", "").strip()
DB_LOCAL = os.getenv("DB_LOCAL", "kovaq_local.db")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
db_queue: queue.Queue = queue.Queue()


def _abrir():
    if TURSO_URL:
        return libsql.connect(database=TURSO_URL, auth_token=TURSO_TOKEN)
    return libsql.connect(database=DB_LOCAL)


def get_connection():
    return _abrir()


def _worker_db():
    conn = None
    while True:
        lote = [db_queue.get()]
        try:
            while len(lote) < 50:
                lote.append(db_queue.get_nowait())
        except queue.Empty:
            pass
        try:
            if conn is None:
                conn = _abrir()
            for query, params in lote:
                conn.execute(query, params)
            conn.commit()
        except Exception as e:
            print(f"⚠️ Error escribiendo en la base: {e}")
            try:
                conn.close()
            except Exception:
                pass
            conn = None
        for _ in lote:
            db_queue.task_done()


threading.Thread(target=_worker_db, daemon=True).start()

_lectura = {"conn": None}
_lock_lectura = threading.Lock()


def leer(query: str, params: tuple = (), uno: bool = False):
    with _lock_lectura:
        for intento in (1, 2):
            try:
                if _lectura["conn"] is None:
                    _lectura["conn"] = _abrir()
                cur = _lectura["conn"].execute(query, params)
                return cur.fetchone() if uno else cur.fetchall()
            except Exception:
                try:
                    _lectura["conn"].close()
                except Exception:
                    pass
                _lectura["conn"] = None
                if intento == 2:
                    raise


def escribir(query: str, params: tuple = ()):
    """Ejecuta sentencias de escritura (INSERT, UPDATE, DELETE) de forma directa."""
    with _lock_lectura:
        conn = _abrir()
        try:
            conn.execute(query, params)
            conn.commit()
        finally:
            conn.close()


def init_db_sync():
    try:
        conn = _abrir()
        conn.execute("CREATE TABLE IF NOT EXISTS roles (id INTEGER PRIMARY KEY AUTOINCREMENT, nombre TEXT UNIQUE)")
        conn.execute("""CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password_hash TEXT,
            nombre_completo TEXT, rol_id INTEGER, FOREIGN KEY(rol_id) REFERENCES roles(id))""")
        conn.execute("CREATE TABLE IF NOT EXISTS trafico (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT DEFAULT CURRENT_TIMESTAMP, tipo_evento TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS permanencia (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT DEFAULT CURRENT_TIMESTAMP, tiempo_segundos REAL)")
        conn.execute("CREATE TABLE IF NOT EXISTS interacciones (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT DEFAULT CURRENT_TIMESTAMP, producto TEXT)")
        conn.commit()
        if conn.execute("SELECT COUNT(*) FROM usuarios").fetchone()[0] == 0:
            conn.execute("INSERT INTO roles (nombre) VALUES ('Admin')")
            clave = os.getenv("ADMIN_PASSWORD", "admin123")
            conn.execute("INSERT INTO usuarios (username, password_hash, nombre_completo, rol_id) VALUES (?, ?, ?, 1)",
                         ("admin", pwd_context.hash(clave), "Admin Kovaq"))
            conn.commit()
            print("✅ Base de datos inicializada y Admin creado.")
        conn.close()
        print(f"✅ Base de datos: {'Turso' if TURSO_URL else 'archivo local ' + DB_LOCAL}")
    except Exception as e:
        print(f"🚨 ERROR CON LA BASE DE DATOS: {e}")


async def init_db():
    import asyncio
    await asyncio.to_thread(init_db_sync)


def guardar_trafico():
    db_queue.put(("INSERT INTO trafico (tipo_evento) VALUES ('entrada')", ()))


def guardar_permanencia(tiempo):
    db_queue.put(("INSERT INTO permanencia (tiempo_segundos) VALUES (?)", (tiempo,)))


def guardar_interaccion(prod):
    db_queue.put(("INSERT INTO interacciones (producto) VALUES (?)", (prod,)))


_cache = {"t": 0.0, "d": None}


def obtener_estadisticas(ttl: float = 4.0) -> dict:
    if _cache["d"] is not None and time.time() - _cache["t"] < ttl:
        return dict(_cache["d"])
    try:
        trafico = leer("SELECT COUNT(*) FROM trafico", uno=True)
        perm = leer("SELECT AVG(tiempo_segundos) FROM permanencia", uno=True)
        inter = leer("SELECT producto, COUNT(*) AS veces FROM interacciones GROUP BY producto ORDER BY veces DESC")
        d = {"total_personas": trafico[0] if trafico else 0,
             "tiempo_promedio_percha": round(perm[0], 2) if perm and perm[0] else 0,
             "productos_tomados": [{"producto": r[0], "veces": r[1]} for r in inter]}
        _cache["t"], _cache["d"] = time.time(), d
        return dict(d)
    except Exception as e:
        if _cache["d"] is not None:
            return dict(_cache["d"])
        return {"error": str(e), "total_personas": 0, "tiempo_promedio_percha": 0, "productos_tomados": []}