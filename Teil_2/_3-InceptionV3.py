import keras
import keras.api.datasets.fashion_mnist as fashion_mnist
import matplotlib.pyplot as plt
from keras.api.applications import InceptionV3
from keras.api.preprocessing.image import img_to_array, array_to_img
import numpy as np
# --------------- Gewünschtes vortrainiertes Modell laden: ---------------
inception_model = InceptionV3(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

# --------------- Zusammenfassung des Modells printen ---------------
inception_model.summary()

# --------------- Training auf Fashion -----------------

# Load the dataset
# Veränderung der Größe der Bilder für das Inputformat von InceptionV3
def preprocess_images(images):
    images_rgb = np.stack([images] * 3, axis=-1)
    images_resized = np.array([img_to_array(array_to_img(img, scale=False).resize((299, 299))) for img in images_rgb])
    return images_resized

# Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train[:200], x_test[:200]
y_train, y_test = y_train[:200], y_test[:200]
x_train, x_test = preprocess_images(x_train), preprocess_images(x_test)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Compile the model
# model.compile(optimizer=Adam(learning_rate=low_lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
inception_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the new_model on the new dataset
fit_history = inception_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the new_model on the test data
performance = inception_model.evaluate(x_test, y_test, batch_size=4)
print('Performance on test data:', performance)

inception_model.save('Transfer Learning/KI/my_inceptionv3_fashion_model.h5')

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