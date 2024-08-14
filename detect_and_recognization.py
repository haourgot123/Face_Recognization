from utils.detectFace import *
from utils.extract_feature import *
from utils.face_recognize import *
import cv2
def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std


def detect_and_recognization(frame, detect_model, facenet_model,l2_normalize, embedding_df):
    _, bounding_box_list = detect(detect_model, frame)
    for box in bounding_box_list:
        x1, y1, x2, y2 = box
        frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
        img_crop = frame[int(y1) : int(y2), int(x1) : int(x2)]
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        img_crop = normalize(img_crop)
        img_crop = cv2.resize(img_crop, (160, 160)) 
        vector_embedding = extract_feature(facenet_model,img_crop)
        vector_embedding = l2_normalize.transform(vector_embedding.reshape(1, -1))[0]
        person_name = face_recognization(vector_embedding, embedding_df)
        cv2.putText(frame, person_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)