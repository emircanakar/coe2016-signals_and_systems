import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

# Ayarlar ve Parametreler 
SAMPLING_FREQ = 8000  # Standart Ornekleme Hizi
DURATION = 0.3  # Tonun Suresi (second)

# DTMF Frekans Haritalama
DTMF_KEYS = {
    '1': (697, 1209),
    '2': (697, 1336),
    '3': (697, 1477),
    'A': (697, 1633),
    '4': (770, 1209),
    '5': (770, 1336),
    '6': (770, 1477),
    'B': (770, 1633),
    '7': (852, 1209),
    '8': (852, 1336),
    '9': (852, 1477),
    'C': (852, 1633),
    '*': (941, 1209),
    '0': (941, 1336),
    '#': (941, 1477),
    'D': (941, 1633),
}


def generate_and_play(key):
    try:
        f_low, f_high = DTMF_KEYS[key]

        # Zaman Ekseni
        t = np.linspace(0, DURATION, int(SAMPLING_FREQ * DURATION), endpoint=False)

        # Sinyal Donusumu (iki sinusoid toplami)
        # Normallestime (0.5 kati) kirpilmayi onlemek icin
        signal = 0.5 * (np.sin(2 * np.pi * f_low * t) + np.sin(2 * np.pi * f_high * t))

        # Ses Ciktisi
        sd.play(signal, SAMPLING_FREQ)

        # Gorsellestirme (Zaman Alani)
        plt.figure("DTMF Visualization", figsize=(8, 4))
        plt.clf()  # Onceki Taslagi Temizle
        # Gorsel Netlik Icin Kucuk Bir Parca Goster (ilk 200 ornek gibi)
        plt.plot(t[:200], signal[:200])
        plt.title(f"Time Domain Signal for Key: {key} ({f_low}Hz + {f_high}Hz)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.draw()
        plt.pause(0.001)

    except Exception as e:
        messagebox.showerror("Error", str(e))


# GUI
root = tk.Tk()
root.title("DTMF Keypad")

# Sablon
key_list = [
    ['1', '2', '3', 'A'],
    ['4', '5', '6', 'B'],
    ['7', '8', '9', 'C'],
    ['*', '0', '#', 'D'],
]

for r, row in enumerate(key_list):
    for c, key in enumerate(row):
        btn = tk.Button(
            root,
            text=key,
            width=10,
            height=3,
            font=('Arial', 12, 'bold'),
            command=lambda k=key: generate_and_play(k),
        )
        btn.grid(row=r, column=c, padx=5, pady=5)

print("Keypad is running. Press any button to hear the tone and see the graph.")
root.mainloop()

