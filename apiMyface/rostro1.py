import cv2
import numpy as np
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

class CameraApp:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
        self.model = load_model('model3.h5')
        self.cap = cv2.VideoCapture(0)

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        ret, imagen = self.cap.read()
        if not ret:
            return None
        imagen_espejo = cv2.flip(imagen, 1)
        rostro = self.detector.detectMultiScale(imagen_espejo, 1.3, 5)
        for (x, y, w, h) in rostro:
            cara = imagen_espejo[y:y + h, x:x + w]
            cara = cv2.resize(cara, (150, 150))
            cara = cara.astype("float") / 255.0
            cara = img_to_array(cara)
            cara = np.expand_dims(cara, axis=0)
            es_mi_cara = self.model.predict(cara)[0][0]
            if es_mi_cara > 0.5:
                # Dibujar rectángulo alrededor de la cara
                cv2.rectangle(imagen_espejo, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Añadir texto "Osvaldo" sobre la cara
                cv2.putText(imagen_espejo, "Osvaldo", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                cv2.rectangle(imagen_espejo, (x, y), (x + w, y + h), (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', imagen_espejo)
        return jpeg.tobytes()

@app.route('/')
def index():
    return render_template('index.html', template_folder='path/to/templates')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(CameraApp()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
