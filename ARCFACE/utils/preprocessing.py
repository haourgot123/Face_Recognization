import numpy as np
import cv2

def normalize_input(img):
    
    img *= 255
    img -= 127.5
    img /= 128
    return img
    
    return img

def resize_image(img, target_size):
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)
    dsize = (
        int(img.shape[1] * factor),
        int(img.shape[0] * factor),
    )
    img = cv2.resize(img, dsize)
    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]
    img = np.pad(
        img,
        (
            (diff_0 // 2, diff_0 - diff_0 // 2),
            (diff_1 // 2, diff_1 - diff_1 // 2),
            (0, 0),
        ),
        "constant",
    )
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    img = np.expand_dims(img, axis=0)
    if img.max() > 1:
        img = (img.astype(np.float32) / 255.0).astype(np.float32)
    return img
