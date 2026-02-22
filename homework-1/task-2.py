import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

# 1. Configuration and Parameters [cite: 68, 69]
SAMPLING_FREQ = 8000  # Standard sampling rate
DURATION = 0.3  # Duration of the tone in seconds

# 2. DTMF Frequency Mapping
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

        # Time axis
        t = np.linspace(0, DURATION, int(SAMPLING_FREQ * DURATION), endpoint=False)

        # Signal Generation (Sum of two sinusoids) [cite: 48, 54, 73]
        # Normalization (multiply by 0.5) to avoid clipping
        signal = 0.5 * (np.sin(2 * np.pi * f_low * t) + np.sin(2 * np.pi * f_high * t))

        # Audio Output [cite: 56, 76]
        sd.play(signal, SAMPLING_FREQ)

        # Visualization (Time Domain) [cite: 55]
        plt.figure("DTMF Visualization", figsize=(8, 4))
        plt.clf()  # Clear previous plot
        # Showing only a small portion (e.g., first 200 samples) for visual clarity
        plt.plot(t[:200], signal[:200])
        plt.title(f"Time Domain Signal for Key: {key} ({f_low}Hz + {f_high}Hz)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.draw()
        plt.pause(0.001)

    except Exception as e:
        messagebox.showerror("Error", str(e))


# 3. GUI Design (Interactive Keypad) [cite: 52, 53]
root = tk.Tk()
root.title("DTMF Keypad")

# Layout configuration
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
