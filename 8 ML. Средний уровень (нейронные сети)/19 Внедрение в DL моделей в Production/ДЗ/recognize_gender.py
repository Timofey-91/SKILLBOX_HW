import sys
import json
import requests
import numpy as np
import dlib
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from tensorflow.keras.preprocessing import image
import cv2

# # запуск сервера
# tensorflow_model_server --rest_api_port=8501 --model_name=checkpoint_best --model_base_path="/mnt/d/my_project_gender/checkpoint_best"

def prepare_input(img_fpath):
    flag_chel = 0
    img = cv2.imread(img_fpath)
    dets = __DETECTOR.run(img, 1)
    print(dets)
    if len(dets[0]) == 1:
        for d in dets[0]:
            img = img[max(0, d.top()): min(d.bottom(), img.shape[1]),
                    max(0, d.left()): min(d.right(), img.shape[0])]
            cv2.imwrite(f'{img_fpath[:-4]}.png', img)
            img = image.load_img(f'{img_fpath[:-4]}.png', target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = utils.preprocess_input(img, version=2) 
    else:
        flag_chel = 1
        print('Лицо человека не распознано. На фото может быть несколько людей. Должно быть одно лицо.')

    return img, flag_chel


# Чтение изображения с диска
# Путь к изображению -- первый аргумент командной строки
img_fpath = sys.argv[1]

# Подготовка входных данных для нейронной сети (вырезаем лицо)
__DETECTOR = dlib.get_frontal_face_detector()

inp, flag_chel = prepare_input(img_fpath)

print(flag_chel)

if flag_chel == 0:
    # Подготовка данных для HTTP запроса
    request_data = json.dumps({
        "signature_name": "serving_default",
        "instances": inp.tolist()
    })
    headers = {"content-type": "application/json"}

    # HTTP запрос на сервер
    json_response = requests.post(
        'http://localhost:8501/v1/models/saved_model/versions/1:predict',
        data=request_data, headers=headers)

    gender_mapping = {0: 'Male', 1: 'Female'}
    # Обработка JSON ответа
    predictions = json.loads(json_response.text)['predictions']
    print(predictions)
    list_pred = []
    for p in predictions:
        list_pred.append(p[0])
    predicted_gender = round(max(list_pred))
    pred_str = gender_mapping[predicted_gender]

    # Печать результата распознавания
    print('Определяем пол по фотографии:')
    print(pred_str)