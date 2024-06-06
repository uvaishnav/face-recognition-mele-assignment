# Standard libraries
import os
import pickle

# Numerical and data manipulation libraries
import numpy as np

# Image processing libraries
import cv2

# Machine learning libraries
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize

# Face detection and recognition libraries
from mtcnn import MTCNN
from deepface.basemodels import Facenet, OpenFace


"""
The idea is to create a classinstance file with common pipeline, 
we use for predicting person from image or adding a new person and identifying him/her
"""

class FaceRecognizer(BaseEstimator, TransformerMixin):
    """
    This class provides functionalities for:
    1) Predicting the identity of a person based on an image.
    2) Adding the embedding of a new person and recognizing him/her in future images.
    """

    def __init__(self, meta_learner, embedding_db_path='embeddings_db.pkl') -> None:
        # Load required models
        self.facenet_model = Facenet.load_facenet128d_model()
        self.openface_model = OpenFace.load_model()
        self.meta_learner = meta_learner

        # Initialize face detector
        self.detector = MTCNN()

        # Load or initialize embeddings_db
        self.embedding_db_path = embedding_db_path
        if os.path.exists(embedding_db_path):
            with open(embedding_db_path, 'rb') as f:
                self.embeddings_db = pickle.load(f)
        else:
            self.embeddings_db = {}

    def extract_faces(self, image):
        """
        Extract faces from the image using MTCNN detector.
        
        Args:
            image (numpy array): The input image.

        Returns:
            numpy array: An array of detected faces.
        """
        detections = self.detector.detect_faces(image)
        faces = []
        for detection in detections:
            x, y, width, height = detection['box']
            face = image[y:y+height, x:x+width]
            face = cv2.resize(face, (160, 160))  # Assuming 160x160 is the required size for the model
            face = face.astype('float32') / 255.0  # Normalize pixel values between 0 and 1
            faces.append(face)
            print(faces[0].shape)
        return np.array(faces)
    
    def extract_features(self, image):
        """
        Extract features from the detected faces using Facenet and OpenFace models.

        Args:
            image (numpy array): The input image.

        Returns:
            numpy array: Concatenated features from Facenet and OpenFace models.
        """
        # Extract faces from image
        faces = self.extract_faces(image)

        if len(faces) == 0:
            print("No faces detected")
            return None
        else:
            print("{} faces detected".format(len(faces)))
        
        print("features for facenet :",faces.shape)
        # Extract features using Facenet
        facenet_features = self.facenet_model.predict(faces)

        # Resize faces to (96,96) for OpenFace model
        open_faces = [cv2.resize(face, (96, 96)) for face in faces]
        open_faces = np.array(open_faces)
        print("features for openface : ",open_faces.shape)

        # Extract features using OpenFace
        openface_features = self.openface_model.predict(open_faces)
        
        # Return concatenated features
        return np.concatenate([facenet_features, openface_features], axis=1)
    
    def add_embeddings(self, label, image):
        """
        Store the embeddings of a new face for future recognition.

        Args:
            label (str): The label for the new face.
            image (numpy array): The input image containing the face.
        """
        features = self.extract_features(image)
        if features is not None:
            normalized_features = normalize(features)
            self.embeddings_db[label] = normalized_features

            # Save the updated embeddings_db to file
            with open(self.embedding_db_path, 'wb') as f:
                pickle.dump(self.embeddings_db, f)

    def predict(self, image):
        """
        Predict the identity of the person in the given image.

        Args:
            image (numpy array): The input image.

        Returns:
            numpy array: Prediction result from the meta learner.
        """
        features = self.extract_features(image)
        
        
        if features is not None:
            print("Features for training :",features.shape)
            return self.meta_learner.predict(features)
        return None
    
    def recognize(self, image, threshold=0.8):
        """
        Recognize the identity of the person in the given image.

        Args:
            image (numpy array): The input image.
            threshold (float): The threshold for recognizing a face.

        Returns:
            str: The recognized identity or "Unknown".
        """
        new_embedding = self.extract_features(image)
        if new_embedding is not None:
            new_embedding = normalize(new_embedding)

            min_dist = float('inf')
            identity = None

            print("No of embeddings till now : {}".format(len(self.embeddings_db)))

            for label, embeddings in self.embeddings_db.items():
                dist = np.linalg.norm(embeddings - new_embedding)
                if dist < min_dist:
                    min_dist = dist
                    identity = label
            
            if min_dist < threshold:
                return identity
        return "Unknown"
