import cv2
import pandas as pd
import os
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

def load_model(model_path):
    model = YOLO(model_path)
    return model

def process_frame(frame, model):
    results = model(frame)
    if len(results) > 0 and len(results[0].boxes) > 0:
        highest_conf = 0.5
        for result in results:
            cursor = result.boxes
            xyxy = cursor.xyxy.cpu().numpy()[0]
            conf = cursor.conf.cpu().numpy()[0]
            if conf > highest_conf:
                highest_conf = conf
                x1, y1, x2, y2 = xyxy[:4]
        if highest_conf > 0.5:
            return [x1, y1, x2, y2, highest_conf]
        else:
            return None
    return None

def process_videos(model_path, record_video, record_csv, selected_videos):
    model = load_model(model_path)

    for video in selected_videos:
        cap = cv2.VideoCapture(video)
        cursor_data = []

        if record_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f'processed_videos/{os.path.basename(video)[:-4]}_processed.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                                  (int(cap.get(3)), int(cap.get(4))))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cursor = process_frame(frame, model)
            if cursor is not None:
                x1, y1, x2, y2, conf = cursor
                if record_csv:
                    cursor_data.append([frame_idx / cap.get(cv2.CAP_PROP_FPS), x1, y1])

                if record_video:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'Conf: {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)

            if record_video:
                out.write(frame)

            frame_idx += 1

        cap.release()
        if record_video:
            out.release()

        if record_csv:
            df = pd.DataFrame(cursor_data, columns=['Time', 'X', 'Y'])
            df.to_csv(f'mouse_positions/{os.path.basename(video)[:-4]}.csv', index=False)

def select_files():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(title='Select Video Files')
    return list(file_paths)

record_video = input("Do you want the boxed videos? (y/n): ").lower() == 'y'
record_csv = input("Do you want the mouse positions? (y/n): ").lower() == 'y'

selected_videos = select_files()

os.makedirs('processed_videos', exist_ok=True)
os.makedirs('mouse_positions', exist_ok=True)

model_path = 'cursorDetector_x.pt'

process_videos(model_path, record_video, record_csv, selected_videos)
