import torch
from PIL import Image
from preprocessing import preprocess_image  # Import the preprocess function

# Load the fine-tuned YOLOv10 model (replace with your model path)
model = torch.hub.load('ultralytics/yolov10', 'yolov10', pretrained=True)  # Replace with your model path

# Perform inference on the image
def perform_inference(image_path):
    # Preprocess the image using the preprocessing module
    preprocessed_image = preprocess_image(image_path)
    
    # Run the model on the preprocessed image
    results = model(preprocessed_image)  # Make sure the image is in a format the model expects (Tensor, etc.)
    
    # Get bounding boxes, class names, and confidence scores
    boxes = results.pandas().xywh  # For YOLOv5/YOLOv10, this will give the bounding boxes in xywh format
    return boxes[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']]

# Test inference
if __name__ == "__main__":
    image_path = 'path_to_test_image.jpg'
    results = perform_inference(image_path)
    print("Inference Results:")
    print(results)
