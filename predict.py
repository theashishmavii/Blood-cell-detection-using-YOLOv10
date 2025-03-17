### predict.py
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained model
model = YOLO('runs/train/BCCD_training/weights/best.pt')  # Path to best weights

# Perform prediction
results = model.predict(source='dataset/images/test/', save=True)

# Display sample prediction result
img = cv2.imread('runs/detect/predict/BloodImage_00011.jpg')  # Sample predicted image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

print("Prediction complete. Results are saved in 'runs/detect/predict' folder.")