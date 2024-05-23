import subprocess

# Запуск команди для тренування YOLOv8
subprocess.run([
    "yolo", "task=detect", "mode=train", 
    "model=models/yolov8/yolo8m.pt", "data=data.yaml", 
    "epochs=5", "imgsz=640"
])

