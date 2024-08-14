
import numpy as np
import tensorflow as tf


def extract_feature(model, img):
    img = img.astype(np.float32)
    img = tf.expand_dims(img, axis = 0)
    embedding = model.predict(img)
    embedding = np.array(embedding).reshape(-1)
    return embedding