#Testreihe - Erlerntes Basemodel von Zahlenerkennung an eigener Schrift testen
#!! - Schwarzer Hintergrund, weiße Zahl. Bild quadratisch (z.B. 150px*150px NICHT 300px*100px)

import numpy as np
from keras.api.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

saved_model = load_model('Transfer Learning\\KI\\my_mnist_model.h5')
print('Model successfully loaded: ', saved_model)

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

image_path = 'Transfer Learning\\KI\\test0.png' #

test_image = preprocess_image(image_path)

print(test_image.shape)

prediction = saved_model.predict(test_image)

predicted_class = np.argmax(prediction)

print(f'The model predicts this image is a: {predicted_class}')

plt.imshow(test_image.squeeze(), cmap='gray')
plt.title(f'Predicted Class: {predicted_class}')
plt.show()
