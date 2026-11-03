import asyncio, cv2, time, threading, websockets
from collections import deque
from ultralytics import YOLO

URL_WEBSOCKET = "wss://kovaq-analytics.onrender.com/ws/video?token=TU_TOKEN"
# Cola con límite 1: retiene solo el fotograma más reciente y elimina el anterior automáticamente
buffer_frames = deque(maxlen=1)
model = YOLO("yolov8n.pt")

def procesar_ia_local():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        results = model(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
        # Compresión agresiva para WebSocket (45% mantiene el balance rendimiento/visibilidad)
        _, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
        buffer_frames.append(jpg.tobytes())
        time.sleep(0.04) # Limita a ~25 FPS máximo

async def transmitir_fluido():
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
    threading.Thread(target=procesar_ia_local, daemon=True).start()
    asyncio.run(transmitir_fluido())