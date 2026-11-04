import asyncio, cv2, time, threading, websockets, queue, sqlite3
from collections import deque
from ultralytics import YOLO

URL_WEBSOCKET = "wss://kovaq-analytics.onrender.com/ws/video?token=TU_TOKEN"
buffer_frames = deque(maxlen=1)

# 1. Crear colas para comunicar los hilos sin que se traben entre sí
cola_captura = queue.Queue(maxsize=3) # Evita acumular frames viejos
cola_db = queue.Queue()               # Cola para enviar datos a SQLite

def hilo_capturar_video():
    """Hilo 1: Dedicado EXCLUSIVAMENTE a leer la cámara a máxima velocidad"""
    cap = cv2.VideoCapture(0) # Cambiar a URL RTSP para las cámaras del súper
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            time.sleep(0.1)
            continue
            
        if cola_captura.full():
            cola_captura.get() # Saca el frame viejo si YOLO está lento
            
        cola_captura.put(frame)
        time.sleep(0.01) # Pequeña pausa para no quemar la CPU

def hilo_base_datos():
    """Hilo 2: Guarda las interacciones en SQLite en segundo plano (Modo WAL)"""
    conn = sqlite3.connect('kovaq.db', check_same_thread=False)
    conn.execute('PRAGMA journal_mode=WAL;') # Evita error "database is locked" con FastAPI
    cursor = conn.cursor()
    
    while True:
        datos = cola_db.get() # Se queda esperando hasta que YOLO le envíe algo
        if datos is None: break
        
        # AQUÍ HACES EL INSERT REAL
        # cursor.execute("INSERT INTO interacciones (cliente, zona, producto) VALUES (?, ?, ?)", datos)
        # conn.commit()
        print(f"[BD] Datos guardados: {datos}")

def hilo_procesar_ia_local():
    """Hilo 3: Proceso pesado de IA (YOLO)"""
    # Recomendación: Cuando termines de entrenar, cambia 'yolov8n.pt' por 'best.onnx'
    model = YOLO("yolov8n.pt") 
    
    while True:
        if not cola_captura.empty():
            frame = cola_captura.get()
            
            # Usar model.track para que asigne IDs temporales a los clientes (Requisito RF1)
            results = model.track(frame, persist=True, verbose=False)
            
            for r in results:
                for box in r.boxes:
                    clase_id = int(box.cls[0])
                    # Si es cliente (0)
                    if clase_id == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # LOGICA DE ZONAS (Simulación):
                        # Si el código detecta que interactuó, enviamos a la BD sin detener YOLO
                        # cola_db.put(("Cliente_15", "Zona_Bebidas", "Coca-Cola"))
                        
            # Compresión y envío a la cola del WebSocket
            _, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
            buffer_frames.append(jpg.tobytes())
        else:
            time.sleep(0.01)

async def transmitir_fluido():
    """Proceso Asíncrono: Envía el video a la nube / web"""
    async with websockets.connect(URL_WEBSOCKET) as ws:
        while True:
            if buffer_frames:
                frame_data = buffer_frames[-1] 
                try:
                    await ws.send(frame_data)
                except Exception:
                    pass
            await asyncio.sleep(0.04)

if __name__ == "__main__":
    # Arrancamos los 3 motores en paralelo
    threading.Thread(target=hilo_capturar_video, daemon=True).start()
    threading.Thread(target=hilo_base_datos, daemon=True).start()
    threading.Thread(target=hilo_procesar_ia_local, daemon=True).start()
    
    # Arrancamos el transmisor web
    asyncio.run(transmitir_fluido())