# Face Recognition Model Using TensorFlow

## Introduction
This project aims to develop a face recognition model capable of predicting the identity of a person based on an input image. Utilizing advanced techniques like transfer learning with pre-trained models, we ensure high accuracy and robust performance. The project is implemented using TensorFlow and leverages state-of-the-art models such as Facenet and OpenFace.

Here's the revised table with icons included in the Technology section:

## 🛠️ Technology Stack
| Technology                                                   | Description                    |
|--------------------------------------------------------------|--------------------------------|
| ![Python](https://img.icons8.com/color/48/000000/python.png) Python                | Programming Language           |
| ![TensorFlow](https://img.icons8.com/color/48/000000/tensorflow.png) TensorFlow    | Deep Learning Framework        |
| ![MTCNN](https://img.icons8.com/color/48/000000/face-id.png) MTCNN                 | Face Detection                 |
| ![Facenet](https://img.icons8.com/color/48/000000/facial-recognition-scan.png) Facenet | Pre-trained Face Recognition   |
| ![OpenFace](https://img.icons8.com/color/48/000000/facial-recognition-scan.png) OpenFace | Pre-trained Face Recognition   |
| ![Matplotlib](https://img.icons8.com/color/48/000000/bar-chart.png) Matplotlib & Seaborn | Data Visualization             |
| ![NumPy](https://img.icons8.com/color/48/000000/numpy.png) NumPy & Pandas         | Data Manipulation              |
| ![Scikit-learn](https://camo.githubusercontent.com/632fad41ad7b3cbaf743281aa2332b3215ad9c621f7e3ffeb9f2274207a82a88/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f2d5363696b69745f4c6561726e2d4637393331453f7374796c653d666c6174266c6f676f3d7363696b69742d6c6561726e266c6f676f436f6c6f723d7768697465) Scikit-learn | Model Evaluation               |

## 📊 Implementation Overview

### 1. 📄 Data Preparation [view notebook](data_preparation.ipynb)   
   - **Libraries:** MTCNN, PIL, Pandas, NumPy
   - **Steps:**
     - Load and inspect the dataset.
     - Use MTCNN to detect and extract faces from images.
     - Split the dataset into training, validation, and test sets.
     - Address class imbalance through data augmentation.
   - **Reasoning:** MTCNN is chosen for its accuracy and reliability in face detection, ensuring high-quality face extraction for subsequent recognition tasks.

### 2. 🤖⚙️ Model Training   [view notebook](model_training_ensemble.ipynb)
   - **Libraries:** TensorFlow, Keras, DeepFace
   - **Steps:**
     - Load and preprocess images using `ImageDataGenerator`.
     - Utilize transfer learning models Facenet and OpenFace for feature extraction.
     - Ensemble these models using stacking by adding a meta-learner on top of the predictions of these two models.
     - This will learn the optimal way to combine the predictions from FaceNet and OpenFace Models.
     - Train the model with early stopping, l2 regularization, Dropout to prevent overfitting.
   - **Reasoning:** Transfer learning with Facenet and OpenFace leverages pre-trained models known for excellent face recognition performance, thus reducing training time and improving accuracy.

### 3. 📈 Model Evaluation
   - **Libraries:** Scikit-learn, Matplotlib, Seaborn
   - **Steps:**
     - Evaluate the model on the test set.
     - Generate and visualize the confusion matrix.
     - Produce a classification report detailing precision, recall, and F1-score.
   - **Reasoning:** Using comprehensive evaluation metrics and visualizations helps in understanding the model's performance and areas of improvement.
   

## Conclusion
This project demonstrates the effectiveness of using transfer learning for face recognition tasks. The use of advanced pre-trained models and robust data handling techniques ensures high accuracy and reliable performance.

---


Feel free to contribute or raise issues if you encounter any problems or have suggestions for improvements. Happy coding!

![TensorFlow](https://img.icons8.com/color/48/000000/tensorflow.png)
