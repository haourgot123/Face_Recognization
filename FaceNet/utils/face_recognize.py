import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

def convert_str2array(input_str):
    input_str = input_str.strip('[]')
    number_list = [float(num) for num in input_str.split()]
    array = np.array(number_list, dtype = np.float32)
    return array


def face_recognization(vectorC, embedding_df):
    distances = []
    for row in range(embedding_df.shape[0]):
        dis = cosine(convert_str2array(embedding_df.iloc[row, 0]), vectorC)
        distances.append(dis)
    min_distance_index =  np.argmin(distances, axis=0)
    person_name = ""
    if distances[min_distance_index] > 0.3:
        person_name = "This person is not in database"
    else:
        person_name = embedding_df.iloc[min_distance_index, 1]
    return person_name