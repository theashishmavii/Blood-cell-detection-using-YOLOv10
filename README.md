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

## 📂 **Folder Structure**

📦 **Blood Cell Detection using YOLOv10**  
├── 📂 **models/**                 # Folder containing the saved YOLOv10 model  
├── 📂 **weights/**                # Folder containing the saved YOLOv10 model weights  
├── 📂 **runs/detect/**            # Folder containing the results of the model  
│   ├── 📂 **train**               # Containing the results like F1-score, PR curve, etc.  
│   ├── 📂 **train2**              # Containing the jpg files of annotated images  
│   ├── 📂 **train3**              # Containing the best results like train folder  
├── 📂 **dataset/**                # Folder containing the dataset  
│   ├── 📂 **files**               # Containing the text files with addresses for splitting into train, test, and validation  
│   ├── 📂 **images**              # Containing the jpg files  
├── 📜 **main_collab_code.py**     # Main code by which models are trained and implemented on Google Colab  
├── 📜 **inference.py**            # Code for running inference with the trained model  
├── 📜 **preprocessing.py**        # Image preprocessing and augmentation code  
├── 📜 **predict.py**              # For prediction on image  
└── 📜 **README.md**               # Project documentation  


---

## 📌 Results:

![BloodImage_00007](https://github.com/user-attachments/assets/cacbf0eb-7d5c-4185-a51e-566e0b0b720b)
![BloodImage_00323](https://github.com/user-attachments/assets/c9671864-4271-4b2b-a0f6-8b5d7fc261d1)
![BloodImage_00161](https://github.com/user-attachments/assets/c17d87fe-74a9-41f3-9ed5-0477d5dfd756)

---

## 📌 Evalution Parameters Results:

![confusion_matrix](https://github.com/user-attachments/assets/f5281459-348b-42a2-9546-a707dd9da5e0)
![confusion_matrix_normalized](https://github.com/user-attachments/assets/1d86ceeb-8b64-451f-8e09-21cd1aec612f)
![F1_curve](https://github.com/user-attachments/assets/b6b57d3e-b442-4d3a-9aaf-6286c0906826)
![P_curve](https://github.com/user-attachments/assets/7dfe5af5-a02e-4ac7-818f-edfb240572f4)
![PR_curve](https://github.com/user-attachments/assets/8b8bb262-3661-47a9-8b39-bd04b411c0b5)
![R_curve](https://github.com/user-attachments/assets/c903218c-f3b4-447d-bccb-6194cea612ac)
![results](https://github.com/user-attachments/assets/e1926c91-6fa3-4807-b41b-f771845649b6)
![labels](https://github.com/user-attachments/assets/fa431fde-7218-4156-80b3-2152a7b136e1)
![labels_correlogram](https://github.com/user-attachments/assets/1c89ce1f-c6ba-41c1-ad46-4c0e2bb37f8a)









