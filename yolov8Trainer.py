from ultralytics import YOLO

model = YOLO('yolov8x.pt')
# model = YOLO('yolov8n.pt')

results = model.train(data='dataset.yaml', epochs=100, workers=0, patience=5, batch=128, imgsz=640)