import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import msvcrt as m
from time import sleep
import os
from pynput import keyboard
from scipy.io.wavfile import write
from pynput.mouse import Button, Controller

class CaptureKeys:
    def __init__(self):
        self.sr = 48000
        self.duration = 1
        pass

    def on_press(self, key):
        if key.char == 'm':
            print('nagrywam')
            my_rec = sd.rec(self.duration * self.sr, channels=1, dtype='float64', samplerate=self.sr, blocking=True)
            plt.plot(my_rec.flatten())
            plt.show()
            sd.play(my_rec)

    def on_release(self, key):
        print('{0} released'.format(
            key))
        if key == keyboard.Key.esc:
            # Stop listener
            return False

    # Collect events until released
    def main(self):
        with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join()

    def start_listener(self):
        keyboard.Listener.start
        self.main()


print(sd.query_devices()); input()
# samplerate = 48000

mouse = Controller()
duration = 1
sr = 48000

sd.default.device = 12, None # 12 - normalnie, do zaminy

no_rec = 100
only_one = True

with sd.InputStream(samplerate=sr):
    sleep(1)
    print('Zaczynam nagrywanie')
    with keyboard.Events() as events:
        while True:
            for event in events:
                if only_one:
                    if str(event.key)[1] == 'm':
                        print(f'Nagrywam nagranie numer {no_rec}')
                        my_rec = sd.rec(duration * sr, channels=1, dtype='float64', samplerate=sr,
                                        blocking=True)
                        sd.wait()

                        #mouse.click(Button.right, 1)
                        #sleep(0.6)
                        #mouse.click(Button.right, 1)

                        write(os.path.join('0', str(no_rec) + '.wav'), sr, my_rec.flatten())
                        no_rec += 1

                    only_one = False
                else:
                    only_one = True



