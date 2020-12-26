import requests
import shutil

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import keras
from utils import preprocess_infer


def download_image(img_name, url=None):

    if url is None:
        url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSq2yVX_FxsRNF6VFgyxFFqaxpf6aZ3sbjWaONlneQHX768CTSEtA"
    else:
        r = requests.get(url, stream=True)

        if r.status_code == 200:
            r.raw.decode_content = True
            with open(img_name, "wb") as f:
                shutil.copyfileobj(r.raw, f)


def predict(img_path, model_path):
    img = Image.open(img_path)
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocess_infer(img)
    img = img.reshape(1, 32, 32, 1)

    model = keras.models.load_model("path/to/location")
    return model.predict_classes(img)


if __name__ == "__main__":

    url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSq2yVX_FxsRNF6VFgyxFFqaxpf6aZ3sbjWaONlneQHX768CTSEtA"
    model_path = "./saved/traffic_signs_model.h5"
    img_name = "image.jpeg"

    download_image(img_name, url)

    prediction = predict(img_name, model_path)

    print("predicted sign: " + str(predict))
