import cv2
from ultralytics import YOLO
from Model.facenet_architecture import InceptionResNetV2
from detect_and_recognization import *
from sklearn.preprocessing import Normalizer

# Load Model YOLO
detect_model = YOLO('Model\yolov8n-face.pt')
facenet_model = InceptionResNetV2()
weight_path = 'Model/facenet_keras_weights.h5'
facenet_model.load_weights(weight_path)
embedding_df = pd.read_csv('embedding_vector.csv')
l2_normalize = Normalizer('l2')
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Can't open webcam")
    exit()

while True:
    ret, frame = cap.read()

    # Check frame successfull
    if not ret:
        print("Can't receive frame")
        break
    detect_and_recognization(frame,detect_model,facenet_model, l2_normalize, embedding_df)
    cv2.imshow('Webcam', frame)

    # Exit using 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
