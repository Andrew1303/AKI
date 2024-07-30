import keras
from keras.api.layers import Dense
from keras.api.models import Model

# Load the InceptionV3 model
inception_model = keras.applications.InceptionV3()

# # --------------- Layer modifizieren -------------------

# --|| Layer am Ende hinzufügen ||--
# Output von letztem Layer ermitteln
last_layer_output = inception_model.layers[-1].output

# Neuen Layer definieren mit der Verbindung zum Output des vorherigen 
new_output = Dense(10, activation='relu')(last_layer_output)

# Neues Modell mit angehängtem Layer erstellen
new_model = Model(inputs=inception_model.input, outputs=new_output)

# Kompilieren des neuen Modells
new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# SUMMARY (Einkommentieren)
##new_model.summary()
##inception_model.summary()