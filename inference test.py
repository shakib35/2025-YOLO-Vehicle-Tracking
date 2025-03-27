from ultralytics import YOLO

model_path = '/Users/shaki/Documents/GitHub/2025-YOLO-Vehicle-Tracking/Car Tracking/yolov8s/weights/best.pt'
video_path = '/Users/shaki/Documents/School/CS7367/Final Paper Stuff/Highway No Audio.mp4'
model = YOLO(model_path)

results = model.track(video_path, show=True, tracker='bytetrack.yaml', device = "mps")  