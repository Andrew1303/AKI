#Basemddel einer KI - Erkennung von Zahlen

from keras.api.models import Sequential
from keras.api.layers import Flatten, Dense, Dropout
import keras.api.datasets.fashion_mnist as fashion_mnist
from keras.api.datasets import fashion_mnist
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
print(len(x_train))
# # Trim the dataset to a small amount of pictures
x_train, y_train = x_train[:1000], y_train[:1000]
x_test, y_test = x_test[:1000], y_test[:1000]

model = Sequential([
    Flatten(input_shape=(28, 28)),  #Inputformat ist 28x28
    Dense(512, activation='relu'),  #Dense - Fully Connected - Jeder Inputwert jedes Pixels wird mit einem Neuronen Perzeptron - Activation relu - beliebteste
    Dropout(0.2),                   #Dropout - Setzt zufällige 20% der Werte auf 0
    Dense(10, activation='softmax') #Fully Connected Layer - Gibt die Möglichen Klassen aus - Aktivierungsfunktion softmax so normiert, dass die Summe der Perzeptoren am ende 1 ergeben
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# adam = adaptive moments

fit_history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

print('Performance on test data: ', model.evaluate(x_test, y_test, 4))

model.save('Transfer Learning/KI/my_fashion_model.h5')

plt.figure(1, figsize = (8,8))
plt.subplot(221)
plt.plot(fit_history.history['accuracy'])
plt.plot(fit_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])

plt.subplot(222)
plt.plot(fit_history.history['loss'])
plt.plot(fit_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])

plt.show()
print('finished')