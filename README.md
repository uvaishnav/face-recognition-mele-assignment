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


## 🎭 Face Recognition Web Application Overview

### Initial Work
As discussed in the Implementation Overview, we have developed a face recognition model by leveraging transfer learning and ensemble techniques using state-of-the-art pretrained models like FaceNet and OpenFace. This model is trained to identify 31 different celebrities. 
- 📊 [Sample Prediction](sample_prediction.ipynb)

### Building the Application
We have a meta-learner that can take features from FaceNet and OpenFace to effectively identify celebrities. To use it for an application, we need to design a clear pipeline that:
1. **Predicts Celebrity Identity**:
   - Takes an input image
   - Extracts features
   - Returns predictions

Our application must also facilitate adding an image of a new person for future recognition. The pipeline should:
2. **Store Embeddings for New Users**:
   - Take an input image
   - Extract features
   - Store embeddings

3. **Recognize Persons Based on Stored Embeddings**:
   - Take an input image
   - Extract features
   - Recognize the person based on stored embeddings

We have implemented a `FaceRecognizer` class to handle all these tasks.  🧩 [View Code](predictor.py)

### Features of the Application
- 📥 **Upload Images of New Users**: Store their embeddings for future recognition.
- 🔍 **Recognize Persons**: Identify persons based on stored embeddings.
- 🌟 **User celebrity Match**: Our meta-learner, trained to classify given images among 31 celebrities, can predict the celebrity that the user's facial features most closely resemble. **This is a fun feature to enhance user interaction with the application.**

To handle backend routes, we have used the Flask framework. 🧩 [View Code](app.py)

## 🎥 Demo
[Watch the demo video](https://github.com/uvaishnav/face-recognition-mele-assignment/assets/104910465/9b6207a6-c37b-42a1-96c1-26070fdb0516)

## 🖥️ Run Locally

#### Clone the project

```bash
  git clone https://github.com/uvaishnav/face-recognition-mele-assignment.git
```

#### Create a conda environment after opening the repository

```bash
  conda create -n faceenv python=3.11 -y
```

```bash
  conda activate faceenv
```

#### Install requirements

```bash
  pip install -r requirements.txt
```

#### Start the server

```bash
python app.py
```
##### Now,

```bash
open up you local host and port
```

## Conclusion
This project demonstrates the effectiveness of using transfer learning for face recognition tasks. The use of advanced pre-trained models and robust data handling techniques ensures high accuracy and reliable performance.

---


Feel free to contribute or raise issues if you encounter any problems or have suggestions for improvements. Happy coding!

![TensorFlow](https://img.icons8.com/color/48/000000/tensorflow.png)
