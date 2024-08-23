from utils.extract_face import extract_and_embedding_face
import cv2
import numpy as np

threshold = 0.68


def detect_and_recognization(img, detect_model, face_model, embedding_database, person_name):
    objs = extract_and_embedding_face(detect_model, face_model, img)
    min_distance, id = float('inf'), None
    for obj in objs:
        embedding = obj['embedding']
        for i, embedding_ in enumerate(embedding_database):
            distance = cosin_distance(embedding, embedding_)
            if distance < min_distance:
                min_distance, id = distance, i
        
        if min_distance < threshold:
            draw(obj, person_name[id], min_distance)
        else:
            draw(obj, 'Unknown', min_distance)
        
        



def draw(obj, obj_name, min_distance):
    img = obj['img']
    x1, y1, x2, y2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
    cv2.putText(img, obj_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f'{min_distance:.2f}', (int(x1) + 100, int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
def cosin_distance(vectora, vectorb):
    a = np.dot(vectora, vectorb)
    b = np.linalg.norm(vectora)
    c = np.linalg.norm(vectorb)
    return 1 - a / (b * c)


def load_database(embedding_path, name_path):
    embedding = np.load(embedding_path)
    obj_name = np.load(name_path)
    return embedding, obj_name