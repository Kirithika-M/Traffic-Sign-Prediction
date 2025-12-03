from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras


def load_dataset_from_folders():
    data_path = os.path.join("datasets", "gtsrb", "Train")

    images = list()
    labels = list()

    ## Looping through the class folders (0 to 42)
    for class_num in range(43):
        class_path = os.path.join(data_path, str(class_num))

        ## Skip if folder doesn't exist
        if not os.path.exists(class_path):
            continue

        img_count = 0

        ## Load all the images in this class folder
        for img_file in os.listdir(class_path):
            if img_file.endswith((".png", ".jpg", ".ppm", ".jpeg")):

                try:
                    ### Read image
                    img_path = os.path.join(class_path, img_file)
                    img = Image.open(img_path).resize((32, 32))

                    ### Convert to RGB if not already
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    img_array = np.array(img)

                    ### Add to the lists
                    images.append(img_array)
                    labels.append(class_num)
                    img_count += 1

                except Exception as e:
                    print(f"Error loading {img_file}: {e}")

    if img_count > 0:
        print(f"Class {class_num}: {img_count} images loaded")
    
    ## Convert to numpy arrays
    x = np.array(images)
    y = np.array(labels)

    return x, y


def pre_process(x, y):

    ## Normalize to [0, 1]
    x = x.astype("float32") / 255.0

    ## One - hot encode labels
    y = keras.utils.to_categorical(y, num_classes=43)

    return x, y

def split_dataset(x, y):
    
    ## First split: 80% training + validation, 20% testing
    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=42,
        stratify=y.argmax(axis=1)   ### Keep class distribution balanced
    )

    ## Second split: 90% training, 10% validation (of the 80%)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp,
        test_size=0.1,
        random_state=42,
        stratify=y_temp.argmax(axis=1)
    )

    return x_train, x_val, x_test, y_train, y_val, y_test
