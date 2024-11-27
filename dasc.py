from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

result = model(source='download.mp4', show=True, conf=0.4, save=True)
