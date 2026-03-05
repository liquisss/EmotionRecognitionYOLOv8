import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Параметры
SMOOTHING_WINDOW = 5  # Размер окна для сглаживания эмоций
CONFIDENCE_THRESHOLD = 0.3  # Минимальная уверенность для детекции
ROI_SCALE = 1.5  # Коэффициент расширения области лица
INPUT_SIZE = 640  # Размер входного изображения для YOLO
CSV_FILE = 'emotion_log.csv'

model = YOLO('best.pt')
print("Модель загружена. Классы:", model.names)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Ошибка: не удалось загрузить каскад Хаара")
    exit()

# Подготовка CSV-файла
if os.path.exists(CSV_FILE):
    os.remove(CSV_FILE)
csv_columns = ['timestamp', 'emotion', 'confidence']
with open(CSV_FILE, 'a') as f:
    pd.DataFrame(columns=csv_columns).to_csv(f, index=False)

# Инициализация веб-камеры
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: не удалось открыть камеру")
    exit()

# Очередь для сглаживания эмоций
emotion_history = deque(maxlen=SMOOTHING_WINDOW)

start_time = datetime.now()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: не удалось получить кадр")
        break

    # Конвертация кадра в градации серого для детекции лиц
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Найдено лиц: {len(faces)}")

    # Список для сохранения данных в CSV
    csv_data = []

    # Обработка каждого лица
    for (x, y, w, h) in faces:
        x1, y1, x2, y2 = x, y, x + w, y + h

        # Извлечение ROI для лица
        width, height = x2 - x1, y2 - y1
        x1_roi = max(0, x1 - int(width * (ROI_SCALE - 1) / 2))
        y1_roi = max(0, y1 - int(height * (ROI_SCALE - 1) / 2))
        x2_roi = min(frame.shape[1], x2 + int(width * (ROI_SCALE - 1) / 2))
        y2_roi = min(frame.shape[0], y2 + int(height * (ROI_SCALE - 1) / 2))

        # Вырезаем и масштабируем ROI
        roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]
        if roi.size == 0:
            continue
        roi_resized = cv2.resize(roi, (INPUT_SIZE, INPUT_SIZE))

        # Инференс YOLO на ROI для классификации эмоций
        results = model(roi_resized, conf=CONFIDENCE_THRESHOLD, imgsz=INPUT_SIZE)
        print(f"Результаты YOLO для ROI: {len(results)}")

        roi_emotion = None
        roi_conf = 0.0

        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf.item()
                cls = box.cls.item()
                class_name = model.names[int(cls)]
                print(f"Эмоция: {class_name} {conf:.2f}")

                if conf > roi_conf:
                    roi_emotion = class_name
                    roi_conf = conf

        if roi_emotion:
            # Добавление эмоции в историю
            emotion_history.append(roi_emotion)

            # Сглаживание эмоций
            valid_emotions = [e for e in emotion_history if e is not None]
            if valid_emotions:
                emotion_counts = np.bincount([list(model.names.values()).index(e) for e in valid_emotions])
                most_common_idx = np.argmax(emotion_counts)
                smoothed_emotion = list(model.names.values())[most_common_idx]
            else:
                smoothed_emotion = roi_emotion

            # Сохранение данных
            timestamp = (datetime.now() - start_time).total_seconds()
            csv_data.append({
                'timestamp': timestamp,
                'emotion': smoothed_emotion,
                'confidence': roi_conf
            })

            # Отрисовка рамки и метки
            label = f"{smoothed_emotion} {roi_conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Сохранение данных в CSV
    if csv_data:
        with open(CSV_FILE, 'a') as f:
            pd.DataFrame(csv_data).to_csv(f, header=False, index=False)

    # Отображение кадра
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Анализ данных из CSV
df = pd.read_csv(CSV_FILE)

# Визуализация
if not df.empty:
    emotion_counts = df['emotion'].value_counts(normalize=True) * 100
    print("Эмоциональный фон (%):")
    print(emotion_counts)

    plt.figure(figsize=(15, 10))
    plt.rcParams.update({'font.size': 12})

    for emotion in model.names.values():
        if emotion in df['emotion'].values:
            emo_data = df[df['emotion'] == emotion]
            plt.scatter(
                emo_data['timestamp'], emo_data['confidence'],
                label=f"{emotion} ({emotion_counts.get(emotion, 0):.1f}%)",
                marker='o', s=100
            )

    plt.xlabel('Время (секунды)', fontsize=14)
    plt.ylabel('Уверенность', fontsize=14)
    plt.title('Эмоциональный фон', fontsize=16, pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('emotion_trend.png', bbox_inches='tight', dpi=300)
    plt.show()
else:
    print("Нет данных для анализа")