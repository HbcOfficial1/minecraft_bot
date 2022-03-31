from scipy.io.wavfile import read
import os
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sounddevice as sd
import pickle as pkl
import librosa as librosa
import librosa.display
from tensorflow.keras import backend as K

def preproces_audio(y, n_fft=2048, hop_length=512, sr=48000):
    spectrogram_librosa = np.abs(librosa.stft(
        y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hann')) ** 2
    spectrogram_librosa_db = librosa.power_to_db(spectrogram_librosa, ref=np.max)
    spectrogram_librosa_db = spectrogram_librosa_db / (-80)
    spectrogram_librosa_db *= 2
    spectrogram_librosa_db -= 1
    return spectrogram_librosa_db

def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

data = []
y = []
undersample = 1


for f_name in os.listdir('0'):
    data.append(read(os.path.join('0', f_name))[1][::undersample])
    y.append(0)

for f_name in os.listdir('1'):
    data.append(read(os.path.join('1', f_name))[1][::undersample])
    y.append(1)

#sd.play(data[-2], samplerate=1500)
#sd.wait()

X = np.array(data)
X = np.array([preproces_audio(i) for i in X])
X = np.expand_dims(X, axis=-1)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)

input_shape = (1025, 94, 1)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=input_shape))
model.add(keras.layers.AvgPool2D((4, 2), strides=(3, 1)))
model.add(keras.layers.Conv2D(16, (3, 3), activation='relu'))
model.add(keras.layers.AvgPool2D((4, 2), strides=(3, 1)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.AvgPool2D((3, 2), strides=(2, 1)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.AvgPool2D((3, 2), strides=(2, 1)))
model.add(keras.layers.Conv2D(16, (3, 3), activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.MaxPool2D((3, 3), strides=(3, 3)))
model.add(keras.layers.Conv2D(16, (3, 3), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(6, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
             optimizer=keras.optimizers.Adam(),
             metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test),
                    batch_size=32, class_weight={0: 0.35, 1: 1})

for i in list(history.history.keys()):
    plt.plot(history.history[i], label=i)

plt.legend(loc='best')
plt.show()

model.save('model_CNN_4.pkl')