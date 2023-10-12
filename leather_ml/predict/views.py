import tensorflow as tf
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.csrf import requires_csrf_token
import os
import time
import numpy as np

import json
from tensorflow.python.ops.numpy_ops import np_config
import requests
np_config.enable_numpy_behavior()
# Create your views here.
path = ""


def index(request):
    return render(request, 'index.html')


@csrf_exempt
def upload(request):
    if request.method == "POST":
        uploaded_file = request.FILES['pic']
        # objtype_name = request.POST["obj"]
        fs = FileSystemStorage()
        filename = uploaded_file.name.replace(" ", "_")
        urlname = fs.save(filename, uploaded_file)
        url = fs.url(urlname)
        fig = url.rsplit("/",1)[0]
        print(fig)
        img_height = 180
        img_width = 180

        img = tf.keras.utils.load_img(
            url, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        test_image = [img_array]

        class_names = ['Folding_marks', 'Grain_off', 'Growth_marks',
                       'loose_grains', 'non_defective', 'pinhole']
        data = json.dumps({"signature_name": "serving_default",
                          "instances": test_image[0].tolist()})
        # print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

        headers = {"content-type": "application/json"}
        json_response = requests.post(
            'http://localhost:8502/v1/models/fashion_model:predict', data=data, headers=headers)
        predictions = json.loads(json_response.text)['predictions']

        inde = predictions[0].index(max(predictions[0]))
        print(class_names[inde])
        class_names[inde] = class_names[inde].replace("_"," ")
        score = tf.nn.softmax(predictions[0])
        content = "This image belongs to the class {} and with the confidence rate of {:.2f}%.".format(class_names[inde],100 * np.max(score))

        context = {"img": url,
                   "res": class_names[inde],
                   "content":content }

    return render(request,"result.html",context)
