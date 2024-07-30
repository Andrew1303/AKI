import pandas as pd
from keras.api.models import load_model
import keras.api.datasets.fashion_mnist as fashion_mnist
import matplotlib.pyplot as plt
from keras.api.optimizers import Adam

# Load the dataset
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train/255.0, x_test / 255.0

# Trim dataset to a smaller amount of pictures
# x_train, y_train = x_train[:1000], y_train[:1000]
# x_test, y_test = x_test[:1000], y_test[:1000]

# Load the pretrained model
model = load_model('Transfer Learning\\KI\\my_mnist_model.h5')

# Set any layers to trainable = False
for i in range(0):
    model.layers[i].trainable = False

# Compile the model with a low learning rate
low_lr = 1e-5  # Define a low learning rate
# model.compile(optimizer=Adam(learning_rate=low_lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the new_model on the new dataset
fit_history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the new_model on the test data
performance = model.evaluate(x_test, y_test, batch_size=4)
print('Performance on test data:', performance)

model.save('Transfer Learning/KI/my_mnist_fashion_model.h5')

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