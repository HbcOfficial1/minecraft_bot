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
from pynput.mouse import Button, Controller
from time import sleep, time
import librosa as librosa
from scipy.io.wavfile import write

def preproces_audio(y, n_fft=2048, hop_length=512, sr=48000):
    spectrogram_librosa = np.abs(librosa.stft(
        y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hann')) ** 2
    spectrogram_librosa_db = librosa.power_to_db(spectrogram_librosa, ref=np.max)
    spectrogram_librosa_db = spectrogram_librosa_db / (-80)
    spectrogram_librosa_db *= 2
    spectrogram_librosa_db -= 1
    return spectrogram_librosa_db


mouse = Controller()
duration = 1
sr = 48000
#print(sd.query_devices()); input()

sd.default.device = 12, None

# Trudne próbki od 1
no_rec = 173
record = False
only_one = True

model = keras.models.load_model('model_CNN_4.pkl')


with sd.InputStream(samplerate=sr):
    sleep(1)
    print('Zaczynam działanie')
    while True:
        my_rec = sd.rec(duration * sr, channels=1, dtype='float64', samplerate=sr, blocking=True).flatten()
        p_my_rec = preproces_audio(my_rec)
        p_my_rec = np.expand_dims(p_my_rec, axis=0)
        p_my_rec = np.expand_dims(p_my_rec, axis=-1)

        start = time()
        pred = model.predict(p_my_rec)[0][0]
        print(pred, 'czas obliczeń:', round(time() - start, 2), 's')
        if pred > 0.5:
            mouse.click(Button.right, 1)
            sleep(1)
            mouse.click(Button.right, 1)
            sleep(1.5)

        if pred > 0.2 and record:
            write(os.path.join('trudne_dane', 'a' + str(no_rec) + '.wav'), sr, my_rec.flatten())
            no_rec += 1
