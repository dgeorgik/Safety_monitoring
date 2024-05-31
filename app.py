import os
import cv2
import time
import asyncio
import threading
from queue import Queue
from flask import Flask, render_template, request, flash, redirect, url_for, Response, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from datetime import datetime
from telegram_bot import send_telegram_notification  # Ваша функция для отправки уведомлений
import concurrent.futures

app = Flask(__name__)

app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = 'static/uploads/'
DETECTIONS_FOLDER = 'static/detections/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTIONS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTIONS_FOLDER'] = DETECTIONS_FOLDER

model_path = 'static/model_learning/last_vs/best (1).pt'
model = YOLO(model_path)

uploaded_files = []
detections = []
interval_detections = []
notification_queue = Queue()

@app.route('/')
def upload_form():
    return render_template('upload.html', uploaded_files=uploaded_files)

@app.route('/', methods=['POST'])
def upload_video():
    if 'file' not in request.files or 'title' not in request.form:
        flash('No file or title part')
        return redirect(request.url)
    file = request.files['file']
    title = request.form['title']
    if file.filename == '':
        flash('No video selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        upload_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        uploaded_files.append({'filename': filename, 'title': title, 'upload_date': upload_date})

        flash({
            'title': 'Video successfully uploaded',
            'filename': filename,
            'upload_time': upload_date
        }, 'success')

        return redirect(url_for('upload_form'))

@app.route('/delete_video/<filename>', methods=['POST'])
def delete_video(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        uploaded_files[:] = [d for d in uploaded_files if d['filename'] != filename]
        flash({
            'title': 'Video successfully deleted',
            'filename': filename
        }, 'error')
    else:
        flash('Error: Video not found')
    return redirect(url_for('upload_form'))

@app.route('/play/<path:filename>', methods=['GET', 'POST'])
def play_video(filename):
    selected_classes = request.form.getlist('classes') if request.method == 'POST' else []
    return render_template('play.html', filename=filename, detections=detections, interval_detections=interval_detections, selected_classes=selected_classes)

@app.route('/detections', methods=['GET', 'POST'])
def view_detections():
    filtered_detections = detections
    if request.method == 'POST':
        filter_video = request.form.get('filter_video')
        filter_label = request.form.get('filter_label')
        filter_class = request.form.get('filter_class')
        filter_time = request.form.get('filter_time')

        if filter_video:
            filtered_detections = [d for d in filtered_detections if d['video'] == filter_video]
        if filter_label:
            filtered_detections = [d for d in filtered_detections if d['label'] == filter_label]
        if filter_class:
            filtered_detections = [d for d in filtered_detections if d['score'] == filter_class]
        if filter_time:
            filtered_detections = [d for d in filtered_detections if filter_time in d['time']]

    return render_template('detections.html', detections=filtered_detections, videos=[f['filename'] for f in uploaded_files])

def process_notifications():
    while True:
        detection = notification_queue.get()
        if detection is None:
            break
        asyncio.run(send_telegram_notification(detection))
        time.sleep(10)

notification_thread = threading.Thread(target=process_notifications)
notification_thread.start()

def generate_video(filename, selected_classes):
    global interval_detections
    interval_detections = []

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detection_count = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1.0 / fps

    class_names = ['face', 'helms', 'no-helms', 'no-reflective-jacket', 'people', 'reflective-jacket']
    class_colors = {
        'face': (0, 255, 0),  # Green
        'helms': (0, 255, 0),  # Green
        'no-helms': (0, 0, 255),  # Red
        'no-reflective-jacket': (0, 0, 255),  # Red
        'people': (0, 255, 0),  # Green
        'reflective-jacket': (0, 255, 0),  # Green
    }

    selected_classes = set(selected_classes)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            frame_count += 1

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            future = executor.submit(model.predict, frame_rgb)
            results = future.result()

            current_time = frame_count / fps

            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            labels = results[0].boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                box = boxes[i]
                score = scores[i]
                label = labels[i]

                if label < 0 or label >= len(class_names):
                    print(f"Warning: Label {label} is out of range for class_names list")
                    continue

                class_name = class_names[label]
                if selected_classes and class_name not in selected_classes:
                    continue

                color = class_colors.get(class_name, (255, 0, 0))

                if score > 0.5:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                    cv2.putText(frame, f'{class_name}: {score:.2f}', (int(box[0]), int(box[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if class_name in ['no-helms', 'no-reflective-jacket']:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        snapshot_filename = f"{timestamp}_{frame_count}.jpg"
                        snapshot_path = os.path.join(app.config['DETECTIONS_FOLDER'], snapshot_filename)
                        cv2.imwrite(snapshot_path, frame)
                        print(f"Snapshot saved at: {snapshot_path}")

                        detection = {
                            'frame': frame_count,
                            'label': class_name,
                            'score': f"{score:.2f}",
                            'time': timestamp,
                            'snapshot': snapshot_filename,
                            'video': filename
                        }
                        detections.append(detection)
                        notification_queue.put(detection)

                        if int(current_time) % 15 == 0 and (
                                len(interval_detections) == 0 or interval_detections[-1]['time'] != int(current_time)):
                            interval_detection = {
                                'detection_number': detection_count + 1,
                                'color': color,
                                'time': f"{int(current_time // 60):02}:{int(current_time % 60):02}"
                            }
                            interval_detections.append(interval_detection)
                            detection_count += 1

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            elapsed_time = time.time() - start_time
            if elapsed_time < frame_duration:
                time.sleep(frame_duration - elapsed_time)

    cap.release()

@app.route('/video_feed/<path:filename>')
def video_feed(filename):
    selected_classes = request.args.getlist('classes')
    return Response(generate_video(filename, selected_classes), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections/<path:filename>')
def get_detection(filename):
    try:
        return send_from_directory(app.config['DETECTIONS_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving {filename}: {e}")
        return "Error serving file", 500

if __name__ == '__main__':
    app.run(threaded=True)
