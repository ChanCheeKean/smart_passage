from ultralytics import YOLO

# Load a model, # build from YAML and transfer weights
model = YOLO('yolov8n.pt') 

# Train the model
model.train(
    data="./config/data.yaml", 
    epochs=2, 
    imgsz=640,
    batch=64,
    device=None
)