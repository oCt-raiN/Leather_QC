import json
from tensorflow.python.ops.numpy_ops import np_config
import requests
np_config.enable_numpy_behavior()
import tensorflow as tf

batch_size = 32
img_height = 180
img_width = 180

sunflower_url = "https://impactiva.com/wp-content/uploads/2017/07/Leather-defect-machinery-fold-mark-1.jpg"
sunflower_path = tf.keras.utils.get_file('random_pic', origin=sunflower_url)

sunflower_path = "/workspaces/Leather_QC/Leather_Defect_Classification/Folding_marks/Folding_marks_01_(5).jpg"

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
test_image = [img_array]




class_names =['Folding_marks', 'Grain_off', 'Growth_marks', 'loose_grains', 'non_defective', 'pinhole']
data = json.dumps({"signature_name": "serving_default", "instances": test_image[0].tolist()})
# print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))


headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8502/v1/models/fashion_model:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']

index = predictions[0].index(max(predictions[0]))
print(class_names[index])