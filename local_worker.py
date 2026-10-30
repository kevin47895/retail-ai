import asyncio, cv2, time, threading, websockets
from collections import deque
from ultralytics import YOLO

# Configuraciones
RETRASO_SEGUNDOS = 60 # 1 minuto de retraso para evitar lag en la web
URL_WEBSOCKET = "wss://kovaq-analytics.onrender.com/ws/video?token=TU_TOKEN"
buffer_frames = deque() # La cola de almacenamiento en memoria
model = YOLO("yolov8n.pt") # Carga YOLO localmente

def procesar_ia_local():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (640, 480))
        
        # 1. Ejecutar YOLO y dibujar (Usa el procesador de tu Laptop)
        results = model(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0: # Si es persona
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
        # Aquí puedes llamar a tu base de datos: guardar_trafico()
                    
        # 2. Comprimir y guardar en el búfer con la hora actual
        _, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        buffer_frames.append({"timestamp": time.time(), "data": jpg.tobytes()})

async def transmitir_con_retraso():
    async with websockets.connect(URL_WEBSOCKET) as ws:
        while True:
            ahora = time.time()
            # 3. Solo enviar si el frame tiene más de 60 segundos guardado
            if buffer_frames and (ahora - buffer_frames[0]["timestamp"]) >= RETRASO_SEGUNDOS:
                item = buffer_frames.popleft() # Saca el frame de la cola
                await ws.send(item["data"])    # Lo envía a la web
            await asyncio.sleep(0.03)

if __name__ == "__main__":
    threading.Thread(target=procesar_ia_local, daemon=True).start()
    asyncio.run(transmitir_con_retraso())