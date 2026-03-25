import os
import cv2
import numpy as np

def load_images(data_dir, img_size=(128, 64)):
    images = []
    labels = []
    class_names = os.listdir(data_dir)

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)

        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels), class_names
