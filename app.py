import cv2
import time
import sqlite3
import threading
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
import supervision as sv
from rfdetr import RFDETRSmall

app = Flask(__name__)

# Configuración Base de Datos SQLite
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            persons INTEGER,
            vehicles INTEGER,
            animals INTEGER
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Cargar Modelo RF-DETR
model = RFDETRSmall()

# Estado Global de la Stream
stream_state = {
    "source": None,
    "running": False,
    "last_counts": {"persons": 0, "vehicles": 0, "animals": 0}
}

lock = threading.Lock()

PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
ANIMAL_CLASSES = {"dog", "cat", "bird", "horse", "sheep", "cow"}

def log_detection_to_db(p, v, a):
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO detections_log (persons, vehicles, animals) VALUES (?, ?, ?)', (p, v, a))
        conn.commit()
        conn.close()
    except Exception as e:
        print("Error registrando en BBDD:", e)

def generate_frames():
    global stream_state
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    last_log_time = 0

    while True:
        with lock:
            if not stream_state["running"] or stream_state["source"] is None:
                time.sleep(0.1)
                continue
            source = stream_state["source"]

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            with lock:
                stream_state["running"] = False
            break

        while cap.isOpened():
            with lock:
                if not stream_state["running"]:
                    break

            ret, frame = cap.read()
            if not ret:
                break

            # Inferencia RF-DETR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = model.predict(rgb_frame, conf_threshold=0.35)

            p_count, v_count, a_count = 0, 0, 0
            labels = []
            class_names = detections.data.get("class_name", [])

            for idx, conf in enumerate(detections.confidence):
                name = class_names[idx].lower() if idx < len(class_names) else ""
                if name in PERSON_CLASSES:
                    p_count += 1
                elif name in VEHICLE_CLASSES:
                    v_count += 1
                elif name in ANIMAL_CLASSES:
                    a_count += 1
                labels.append(f"{name} {conf:.2f}")

            with lock:
                stream_state["last_counts"] = {"persons": p_count, "vehicles": v_count, "animals": a_count}

            # Guardar en la BD cada 5 segundos para mantener histórico de gráficos
            if time.time() - last_log_time > 5:
                log_detection_to_db(p_count, v_count, a_count)
                last_log_time = time.time()

            # Anotar escena
            annotated = box_annotator.annotate(scene=frame, detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

            # Codificar a JPEG para streaming HTTP
            _, buffer = cv2.imencode('.jpg', annotated)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start', methods=['POST'])
def start_stream():
    data = request.json
    source = data.get('source')
    # Convertir a entero si es un índice de webcam (ej: "0")
    if source.isdigit():
        source = int(source)

    with lock:
        stream_state["source"] = source
        stream_state["running"] = True
    return jsonify({"status": "started", "source": source})

@app.route('/api/stop', methods=['POST'])
def stop_stream():
    with lock:
        stream_state["running"] = False
    return jsonify({"status": "stopped"})

@app.route('/api/metrics')
def get_metrics():
    with lock:
        current = stream_state["last_counts"]

    # Traer historial agrupado por minutos/horas para la gráfica
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT strftime('%H:%M', timestamp) as minute, 
               AVG(persons), AVG(vehicles), AVG(animals)
        FROM detections_log 
        WHERE timestamp >= datetime('now', '-1 hour')
        GROUP BY minute 
        ORDER BY minute ASC
    ''')
    rows = cursor.fetchall()
    conn.close()

    history = {
        "labels": [r[0] for r in rows],
        "persons": [round(r[1], 1) for r in rows],
        "vehicles": [round(r[2], 1) for r in rows],
        "animals": [round(r[3], 1) for r in rows],
    }

    return jsonify({"current": current, "history": history})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
