from flask import Flask, render_template, request, redirect, url_for, flash, Response
import cv2
import tensorflow as tf
import numpy as np
import os
import pickle
from werkzeug.utils import secure_filename
from predictor import FaceRecognizer

app = Flask(__name__)

# getting our face recognizer Instance
meta_learner = tf.keras.models.load_model("ensemble_meta_learner.h5")
face_recognizer = FaceRecognizer(meta_learner=meta_learner)

# get the labels of celebraties
labels = os.listdir("artifacts/extracted_faces/test")   # used this path just to get the laels
labels = sorted(labels)

# Global variable to control the video stream
video_streaming = True

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    upload_url = url_for('upload')
    recognize_url = url_for('recognize')
    celebrity_match_url = url_for('celebrity_match')
    return render_template('index.html', upload_url=upload_url, recognize_url=recognize_url, celebrity_match_url=celebrity_match_url)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            print("Image uploaded successfully")
            
            # Read image
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            print("Read the uploaded Image")

            # Get the name of the person
            name = request.form['name']

            # Add embeddings
            face_recognizer.add_embeddings(name, image)

            print("Your face embeddings stored Successfully")

            # Save the updated embeddings_db
            with open('embeddings_db.pkl', 'wb') as f:
                pickle.dump(face_recognizer.embeddings_db, f)
            
            return redirect(url_for('index'))
    return render_template('upload.html')

def generate_frames(type):
    global video_streaming
    cap = cv2.VideoCapture(0)
    while video_streaming:
        ret, frame = cap.read()
        if not ret:
            print("Could not open webcam")
            break

        # Recognize faces in the frame
        identity = face_recognizer.recognize(frame)

        if type=="person":
            identity = face_recognizer.recognize(frame)
        elif type == "celeb":
            prediction_prob = face_recognizer.predict(frame)
            if prediction_prob is not None:
                confidence = int(np.max(prediction_prob,axis=1)*100)
                prediction = np.argmax(prediction_prob)
                predicted_celeb = labels[prediction]
                identity = "{} \nconfidence: {} %".format(predicted_celeb,confidence)
            else:
                identity = "Unknown"

        else:
            identity = "Unidentified"

        print("identified you!!! {}".format(identity))

        # Draw bounding box and label
        for detection in face_recognizer.detector.detect_faces(frame):
            x, y, width, height = detection['box']
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, identity, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as a byte array
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/person_feed')
def person_feed():
    global video_streaming
    video_streaming = True
    return Response(generate_frames("person"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video_feed', methods=['GET'])
def stop_video_feed():
    global video_streaming
    video_streaming = False
    return redirect(url_for('index'))

@app.route('/recognize')
def recognize():
    # Render the HTML page that contains the video feed
    return render_template('recognize.html')
   

@app.route('/celebrity_match')
def celebrity_match():
    return render_template('celebrity_match.html')

@app.route('/celeb_feed')
def celeb_feed():
    global video_streaming
    video_streaming = True
    return Response(generate_frames("celeb"), mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    app.run(debug=True)