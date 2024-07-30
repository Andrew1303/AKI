#Basemddel einer KI - Erkennung von Zahlen
import keras
from keras.api.models import Sequential
from keras.api.layers import Flatten, Dense, Dropout, InputLayer, Input
import keras.api.datasets.mnist as mnist
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test / 255.0

modell = Sequential([
    Flatten(),  #Inputformat ist 28x28
    Dense(512, activation='relu'),  #Dense - Fully Connected - Jeder Inputwert jedes Pixels wird mit einem Neuronen Perzeptron - Activation relu - beliebteste
    Dropout(0.2),                   #Dropout - Setzt zufällige 20% der Werte auf 0
    Dense(10, activation='softmax') #Fully Connected Layer - Gibt die Möglichen Klassen aus - Aktivierungsfunktion softmax so normiert, dass die Summe der Perzeptoren am ende 1 ergeben
])

modell.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# adam adaptive moments

fit_history = modell.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

print('Performance on test data: ', modell.evaluate(x_test, y_test, 4))

# modell.save('my_mnist_model.h5') 
modell.save('Transfer Learning/KI/my_mnist_model.h5')
# modell.save('my_pure_mnist_model.h5')

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