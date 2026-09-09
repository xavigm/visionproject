import cv2
import time
import sqlite3
import threading
from flask import Flask, render_template, Response, jsonify, request
import supervision as sv
from rfdetr import RFDETRMedium

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    # Guardamos los "nuevos detectados" en el intervalo, no el frame instantáneo
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            new_persons INTEGER,
            new_vehicles INTEGER,
            new_animals INTEGER,
            instant_persons INTEGER,
            instant_vehicles INTEGER,
            instant_animals INTEGER
        )
    ''')
    conn.commit()
    conn.close()

init_db()

model = RFDETRMedium()

stream_state = {
    "source": None,
    "running": False,
    "last_counts": {"persons": 0, "vehicles": 0, "animals": 0}
}

lock = threading.Lock()

PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
ANIMAL_CLASSES = {"dog", "cat", "bird", "horse", "sheep", "cow"}

def generate_frames():
    global stream_state
    
    # 1. Inicializar Tracker de Objetos (ByteTrack)
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # Sets para registrar IDs únicos ya contados en la sesión activa
    tracked_person_ids = set()
    tracked_vehicle_ids = set()
    tracked_animal_ids = set()

    # Acumuladores de "Nuevos objetos" entre intervalos de guardado en BBDD
    new_p_interval = 0
    new_v_interval = 0
    new_a_interval = 0
    
    last_log_time = time.time()

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

        # Reiniciamos trackers si cambia la fuente de vídeo
        tracker = sv.ByteTrack()
        tracked_person_ids.clear()
        tracked_vehicle_ids.clear()
        tracked_animal_ids.clear()

        while cap.isOpened():
            with lock:
                if not stream_state["running"]:
                    break

            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = model.predict(rgb_frame, conf_threshold=0.3)

            # 2. Asignar IDs con ByteTrack
            detections = tracker.update_with_detections(detections)

            curr_p, curr_v, curr_a = 0, 0, 0
            labels = []
            class_names = detections.data.get("class_name", [])

            # Iterar sobre las detecciones con tracker ID asignado
            if detections.tracker_id is not None:
                for idx, (tracker_id, conf) in enumerate(zip(detections.tracker_id, detections.confidence)):
                    name = class_names[idx].lower() if idx < len(class_names) else ""
                    
                    if name in PERSON_CLASSES:
                        curr_p += 1
                        if tracker_id not in tracked_person_ids:
                            tracked_person_ids.add(tracker_id)
                            new_p_interval += 1
                            
                    elif name in VEHICLE_CLASSES:
                        curr_v += 1
                        if tracker_id not in tracked_vehicle_ids:
                            tracked_vehicle_ids.add(tracker_id)
                            new_v_interval += 1
                            
                    elif name in ANIMAL_CLASSES:
                        curr_a += 1
                        if tracker_id not in tracked_animal_ids:
                            tracked_animal_ids.add(tracker_id)
                            new_a_interval += 1

                    labels.append(f"#{tracker_id} {name} {conf:.2f}")

            with lock:
                stream_state["last_counts"] = {"persons": curr_p, "vehicles": curr_v, "animals": curr_a}

            # 3. Registrar en BBDD solo las NUEVAS incorporaciones cada 5 segundos
            if time.time() - last_log_time > 5:
                try:
                    conn = sqlite3.connect('database.db')
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO detections_log 
                        (new_persons, new_vehicles, new_animals, instant_persons, instant_vehicles, instant_animals) 
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (new_p_interval, new_v_interval, new_a_interval, curr_p, curr_v, curr_a))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print("Error BBDD:", e)

                # Reseteamos los contadores de "Nuevos" para el siguiente intervalo
                new_p_interval = 0
                new_v_interval = 0
                new_a_interval = 0
                last_log_time = time.time()

            # Anotar escena
            annotated = box_annotator.annotate(scene=frame, detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

            _, buffer = cv2.imencode('.jpg', annotated)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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
    if source.isdigit():
        source = int(source)

    with lock:
        stream_state["source"] = source
        stream_state["running"] = True
    return jsonify({"status": "started"})

@app.route('/api/stop', methods=['POST'])
def stop_stream():
    with lock:
        stream_state["running"] = False
    return jsonify({"status": "stopped"})

@app.route('/api/metrics')
def get_metrics():
    with lock:
        current = stream_state["last_counts"]

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # 1. Total en la última hora (Suma de NUEVOS detectados en los últimos 60 min)
    cursor.execute('''
        SELECT COALESCE(SUM(new_persons), 0), COALESCE(SUM(new_vehicles), 0), COALESCE(SUM(new_animals), 0)
        FROM detections_log
        WHERE timestamp >= datetime('now', '-1 hour')
    ''')
    lh = cursor.fetchone()
    last_hour = {"persons": lh[0], "vehicles": lh[1], "animals": lh[2]}

    # 2. Total histórico (Suma de todos los NUEVOS detectados)
    cursor.execute('''
        SELECT COALESCE(SUM(new_persons), 0), COALESCE(SUM(new_vehicles), 0), COALESCE(SUM(new_animals), 0)
        FROM detections_log
    ''')
    tot = cursor.fetchone()
    total_all = {"persons": tot[0], "vehicles": tot[1], "animals": tot[2]}

    # 3. Serie temporal para la gráfica (Promedio de la presencia instantánea)
    cursor.execute('''
        SELECT strftime('%H:%M', timestamp) as minute, 
               AVG(instant_persons), AVG(instant_vehicles), AVG(instant_animals)
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

    return jsonify({
        "current": current,
        "last_hour": last_hour,
        "total_all": total_all,
        "history": history
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
