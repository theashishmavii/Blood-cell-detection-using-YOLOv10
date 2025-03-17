### train.py
from ultralytics import YOLO

# Load YOLOv10n model (nano version for faster training)
model = YOLO('yolov10n.pt')

# Train the model
model.train(
    data='dataset/data.yaml',  # Dataset configuration
    epochs=50,                 # Number of training epochs
    imgsz=640,                 # Image size
    batch=16,                  # Batch size
    project='runs/train',      # Save training runs here
    name='BCCD_training'       # Custom name for run folder
)

print("Training complete. Results are saved in the 'runs/train/BCCD_training' folder.")