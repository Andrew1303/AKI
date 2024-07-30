#Testreihe - Erlerntes Basemodel von Zahlenerkennung an eigener Schrift testen
#!! - Schwarzer Hintergrund, weiße Zahl. Bild quadratisch (z.B. 150px*150px NICHT 300px*100px)

import numpy as np
from keras.api.models import load_model
import matplotlib.pyplot as plt
import keras.api.datasets.fashion_mnist as fashion_mnist

#Testpictureindex
testindex = 7

# Load the dataset
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train/255.0, x_test / 255.0

saved_model = load_model('Transfer Learning\\KI\\my_mnist_fashion_model.h5')
# saved_model = load_model('Transfer Learning\\KI\\my_fashion_model.h5')

def preprocess_image(image):
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

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
