import tensorflow as tf
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.csrf import requires_csrf_token
import os
import time
import numpy as np

import uuid
import json
import re
from django.contrib import messages
from tensorflow.python.ops.numpy_ops import np_config
import requests
np_config.enable_numpy_behavior()
# Create your views here.
path = ""


def index(request):
    return render(request, 'index.html')


def predict(data):
    headers = {"content-type": "application/json"}
    json_response = requests.post(
        'http://localhost:8502/v1/models/fashion_model:predict', data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']

    return predictions


@csrf_exempt
def upload(request):
    if request.method == "POST":
        u =uuid.uuid4()
        print(u, str(u))
        os.system("mkdir media/"+str(u))
        uploaded_file = request.FILES['pic']
        # objtype_name = request.POST["obj"]
        dat =[]
        fs = FileSystemStorage()
        filename = uploaded_file.name.replace(" ", "_")
        if (re.search(re.compile("([^\\s]+(\\.(?i)(jpeg|jpg|png|gif|bmp))$)"), filename)):
            temp = []
            # print(filename)
            urlname = fs.save(str(u)+"/"+filename, uploaded_file)
            url = fs.url(urlname)
            # fig = url.rsplit("/",1)[0]
            # print(fig)
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
            predictions = predict(data)
            inde = predictions[0].index(max(predictions[0]))
            print(class_names[inde])
            class_names[inde] = class_names[inde].replace("_", " ")
            res = class_names[inde]
            score = tf.nn.softmax(predictions[0])
            content = "This image belongs to the class {} and with the confidence rate of {:.2f}%.".format(
            class_names[inde], 100 * np.max(score))
            temp.append(url)
            temp.append(res)
            temp.append(content)
            dat.append(temp)
            context = {"dat":dat}
        elif (re.search(re.compile("([^\\s]+(\\.(?i)(zip|rar|.tar.gz))$)"), filename)):
            if filename.endwith("zip"):
                print("zip")
                os.system("mkdir media/"+u)
            if filename.endwith("rar"):
                print("rar")
            if filename.endwith(".tar.gz"):
                print(".tar.gz")
                
        else:
            messages.error(
                request, "Check the uploaded file must either be an image or a compressed folder containing image!")
            return render(request, "index.html")


    return render(request, "result.html", context)
