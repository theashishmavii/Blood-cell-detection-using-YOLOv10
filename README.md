# 🚑 **Blood Cell Detection using YOLOv10**

## 📖 **Project Overview**
This project aims to build an **Object Detection Model** to identify and classify different types of blood cells using the **YOLOv10** model. The model is trained on the **BCCD (Blood Cell Count Dataset)**, which includes images of various blood cells like red blood cells (RBCs), white blood cells (WBCs), and platelets.

The goal of this project is to automate the detection of blood cells, which is crucial for medical applications like automated blood cell counting, disease detection, and more. Using YOLOv10, a state-of-the-art real-time object detection model, the project demonstrates how deep learning can be applied to healthcare tasks.

---

## 💡 **Motivation**
Automating blood cell detection can significantly enhance the speed and accuracy of medical diagnoses. Manual blood cell counting is a labor-intensive and error-prone task. By leveraging computer vision techniques, we can build an efficient and automated solution that assists healthcare professionals in diagnosing diseases, evaluating blood health, and monitoring medical conditions.

This project highlights the application of deep learning in healthcare and aims to contribute to the ongoing efforts in **medical automation**.

---

## ⚙️ **Technologies Used**
- **YOLOv10**: For real-time object detection and fine-tuning on the BCCD dataset.
- **PyTorch**: Framework used for building, training, and fine-tuning the model.
- **Ultralytics YOLOv10**: The YOLOv10 model implementation used in this project for training and inference. More info: [Ultralytics YOLOv10](https://github.com/ultralytics/yolov5).
- **OpenCV**: Used for image processing and augmentations.
- **NumPy & Pandas**: For data manipulation and processing.
- **Matplotlib**: For visualizing model performance and evaluation metrics.
- **BCCD Dataset**: Blood Cell Count Dataset used for training the model.

---

## 🚀 **How It Works**

1. **Data Preprocessing**:
   - The dataset is preprocessed with augmentation techniques like **rotation**, **cropping**, and **scaling** to improve the model’s generalization and performance.
   
2. **Model Training**:
   - YOLOv10 is fine-tuned on the BCCD dataset. During training, the model learns to detect different classes of blood cells, such as red blood cells, white blood cells, and platelets.
   
3. **Model Inference**:
   - After training, the model is used for inference on unseen images. The output includes the bounding boxes around detected blood cells, their class labels, and confidence scores.
   
4. **Evaluation**:
   - The performance of the model is evaluated using metrics like **precision**, **recall**, and **F1 score**. The evaluation is visualized through a **precision-recall curve** and F1 score metrics.

---

📦 Blood-cell-detection-using-YOLOv10/
│
├── 📂 model/                        # Folder containing the saved YOLOv10 model weights
├── 📂 evaluation/                   # Folder containing evaluation results
│   ├── 📜 f1_score.png              # Placeholder for F1 score image
│   ├── 📜 pr_curve.png              # Placeholder for PR curve image
├── 📂 dataset/                      # Folder containing the BCCD dataset
│   ├── 📂 images/                   # Blood cell images for training and testing
│   └── 📂 annotations/              # Label files in YOLO format
├── 📜 inference.py                  # Code for running inference with the trained model
├── 📜 preprocessing.py              # Image preprocessing and augmentation code
└── 📜 README.md                     # Project documentation (this file)



