# **Action Recognition Using Mediapipe Holistic Model**

This project implements an action recognition system that uses images and landmarks captured via webcam and Mediapipe's Holistic model. The system predicts actions based on human movements using deep learning techniques.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Data Collection](#data-collection)
3. [Model Training](#model-training)
4. [Results](#results)
5. [Setup and Usage](#setup-and-usage)
6. [Acknowledgements](#acknowledgements)

---

## **Overview**

The project is divided into two main stages:

1. **Data Collection**: Using a webcam and Mediapipe's Holistic model to collect both raw images and landmarks.  
2. **Model Training**: Training a deep learning model in Google Colab to classify actions based on the collected data.  

### **Key Features**
- **Real-Time Data Collection**: Leverages Mediapipe for extracting face, pose, and hand landmarks in real-time using a webcam.
- **Deep Learning Model**: A Sequential LSTM model trained on structured keypoint data.
- **Cloud-Based Training**: Conducts model training using Google Colab for GPU acceleration.

This project addresses local GPU constraints by utilizing cloud resources for deep learning model training.

---

## **Data Collection**

Data was collected using the following tools:  
- **Mediapipe Holistic Model**: Extracts 468 face landmarks, 33 pose landmarks, and 21 hand landmarks per frame.  
- **Webcam**: Captures images and landmarks in real-time.  
- **Jupyter Notebook**: Facilitates data collection and saves keypoints as structured `.npy` files.

### **Dataset Details**
- **Actions**: "Hello," "Thanks," and "I Love You."  
- **Saved Format**: Landmarks stored as `.npy` files, organized by actions and sequences.

### **Steps to Collect Data**
1. Open the Jupyter Notebook: `data_preparation/data_collection.ipynb`.
2. Configure the actions, sequence length, and data paths.
3. Run the notebook to collect data using your webcam.
4. Collected keypoints will be saved in the `data/MP_Data/` directory.

---

## **Model Training**

The model was trained using a cloud-based environment with the following components:

- **Architecture**: Long Short-Term Memory (LSTM) network for temporal sequence classification.  
- **Input**: Preprocessed landmark data representing human body movements.  
- **Output**: Predicted action labels.

### **Steps to Train the Model**
1. Open the Colab Notebook: `model_training/model_training.ipynb`.  
2. Upload the `MP_Data/` directory or set the correct data path.  
3. Train the model and evaluate its accuracy.  

The training notebook is available in the `model_training/` directory.

---

## **Results**

### **Action Classification Examples**
- **Predicted**: "Thanks"  
- **Actual**: "Thanks"  

### **Model Performance**
- [Include accuracy or evaluation metrics when available.]

---

## **Setup and Usage**

Follow these steps to set up and use the project:

1. Clone the repository:  
   ```bash
   git clone https://github.com/your_username/action-recognition.git
   cd action-recognition


#### **6. Acknowledgements**
```markdown
## Acknowledgements
- [Mediapipe](https://google.github.io/mediapipe/) for the Holistic model.
- [Google Colab](https://colab.research.google.com/) for providing GPU resources.


