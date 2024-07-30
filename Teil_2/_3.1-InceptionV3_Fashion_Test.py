#Testreihe - Erlerntes Basemodel von Zahlenerkennung an eigener Schrift testen
#!! - Schwarzer Hintergrund, weiße Zahl. Bild quadratisch (z.B. 150px*150px NICHT 300px*100px)

import numpy as np
from keras.api.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import keras.api.datasets.fashion_mnist as fashion_mnist
from keras.api.preprocessing.image import img_to_array, array_to_img

#Testpictureindex
testindex = 7

# Load the dataset
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train/255.0, x_test / 255.0

saved_model = load_model('Transfer Learning\\KI\\my_inceptionv3_fashion_model.h5')
print('Model successfully loaded')

def preprocess_image(image):
    image_rgb = np.stack([image] * 3, axis=-1)
    image_resized = img_to_array(array_to_img(image_rgb, scale=False).resize((299, 299)))
    image_resized = image_resized / 255.0
    return np.expand_dims(image_resized, axis=0)  # Add batch dimension

test_image = preprocess_image(x_train[testindex])

prediction = saved_model.predict(test_image)

predicted_class = np.argmax(prediction)

class_names = {
    0: 'T-shirt/top', 
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'}

predicted_item = class_names[predicted_class]

print(f'The predicted item is: {predicted_item}')

plt.imshow(x_test[testindex].squeeze(), cmap='gray')
plt.title(f'Predicted Class: {predicted_item}')
plt.show()
