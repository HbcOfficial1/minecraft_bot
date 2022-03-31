import sys
import sounddevice as sd
from scipy.io.wavfile import read
import os
import numpy as np

"""
file = sys.argv[1]
rate, data = read(file)
sd.play(data, samplerate=rate)
sd.wait()
"""
c_dir = 'working_dir'
data_dir = np.sort(np.array(os.listdir(c_dir)))

for fname in data_dir:
    print('Aktualnie grany plik:', fname)
    rate, data = read(os.path.join(c_dir, fname))
    sd.play(data, samplerate=rate)
    sd.wait()
