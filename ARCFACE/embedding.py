from utils import detect
from ultralytics import YOLO
import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.preprocessing import normalize_input, resize_image
from model.arcface_model import build_model
model = YOLO('Model\yolov8n-face.pt')
arcface_model = build_model()
weight = 'model/arcface_weights.h5'
arcface_model.load_weights(weight)
embeddings_vector = []
person_name = []
data_path = 'dataset'

for dict_path in os.listdir(data_path):
    image_paths = os.listdir(os.path.join(data_path,dict_path))
    number_image = len(image_paths)
    embedding_vector = []
    for image in image_paths:
        image_path = os.path.join(data_path, dict_path, image)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, bouding_box_list = detect.detect(model,img)
        try:
            x1, y1, x2, y2 = bouding_box_list[0]
        except:
            number_image -= 1
            print("Cant Detect")
        img = img[int(y1) : int(y2), int(x1) : int(x2)]
        img = img[:, :, ::-1]
        img = resize_image(img, target_size=(112, 112))
        img = normalize_input(img)
        embedding = arcface_model.predict(img)[0]
        print(embedding)
        embedding_vector.append(embedding)
    average_embedding = np.mean(embedding_vector, axis = 0)
    average_embedding = average_embedding / np.linalg.norm(average_embedding)
    embeddings_vector.append(np.array(average_embedding))
    person_name.append(dict_path)

np.save('embeddings_vector_arcface.npy', embeddings_vector)
np.save('person_name.npy', person_name)
        
        






