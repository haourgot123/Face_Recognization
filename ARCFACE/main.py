import cv2
from ultralytics import YOLO
from model.arcface_model import build_model
from utils.recognization import detect_and_recognization, load_database

detect_model = YOLO('model/yolov8n-face.pt')
face_model = build_model()
weights = 'model/arcface_weights.h5'
face_model.load_weights(weights)

embedding_database, person_name_database = load_database(
    embedding_path = 'embeding/embeddings_vector_arcface.npy', 
    name_path = 'embeding/person_name.npy'
)


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
    detect_and_recognization(frame, detect_model, face_model, embedding_database, person_name_database)
    cv2.imshow('Webcam', frame)

    # Exit using 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
