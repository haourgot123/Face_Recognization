from utils.detect import detect
from utils.preprocessing import normalize_input, resize_image
import numpy as np

def l2_normalize_embedding(embedding):
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def embedding_face(face_model, img):
    embedding = face_model.predict(img)
    return embedding[0]


def extract_and_embedding_face(detect_model, face_model, img):
    objs =[
        {
        'img' : img,
        'x1' : None,
        'y1' : None,
        'x2' : None,
        'y2' : None,
        'embedding': None
        }
    ]
    # detect face
    _, bbox_list = detect(detect_model, img)
    for i, box in enumerate(bbox_list):
        x1, y1, x2, y2 = box
        img_crop = img[int(y1) : int(y2), int(x1) : int(x2)]
        img_crop = img_crop[:, :, ::-1]
        # resize image
        img_crop  = resize_image(img_crop, target_size = (112, 112))
        # Normalize imgae
        img_crop = normalize_input(img_crop)
        embedding = embedding_face(face_model, img_crop)
        #embedding = l2_normalize_embedding(embedding)
        objs.append(
            {
                'img' : img,
                'x1' : x1,
                'y1' : y1,
                'x2' : x2,
                'y2' : y2,
                'embedding' : embedding
            }
        )
    return objs[1:]
    
