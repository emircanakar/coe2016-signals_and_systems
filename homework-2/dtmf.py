import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt

# Technical parameters from report
FS = 8000  # Sampling frequency
DURATION = 0.04  # 40 ms duration
TURKISH_LETTERS = [
    'A',
    'B',
    'C',
    'Ç',
    'D',
    'E',
    'F',
    'G',
    'Ğ',
    'H',
    'I',
    'İ',
    'J',
    'K',
    'L',
    'M',
    'N',
    'O',
    'Ö',
    'P',
    'R',
    'S',
    'Ş',
    'T',
    'U',
    'Ü',
    'V',
    'Y',
    'Z',
    ' ',
]

# Frequency mapping for 30 characters
freq_map = {ch: (400 + i * 20, 1200 + i * 25) for i, ch in enumerate(TURKISH_LETTERS)}


def encode_text(text):
    text = text.upper()
    full_signal = np.array([], dtype=np.float32)
    t = np.linspace(0, DURATION, int(FS * DURATION), False)
    for ch in text:
        if ch in freq_map:
            f_low, f_high = freq_map[ch]
            # Normalization formula from report
            tone = 0.5 * (
                np.sin(2 * np.pi * f_low * t) + np.sin(2 * np.pi * f_high * t)
            )
            full_signal = np.concatenate((full_signal, tone))
    return full_signal


def goertzel(signal, target_freq, fs):
    # Optimized frequency detection
    N = len(signal)
    k = int(0.5 + (N * target_freq) / fs)
    w = (2 * np.pi / N) * k
    coeff = 2 * np.cos(w)
    q1, q2 = 0, 0
    for x in signal:
        q0 = coeff * q1 - q2 + x
        q2, q1 = q1, q0
    return np.sqrt(q1**2 + q2**2 - q1 * q2 * coeff)


def decode_signal(signal):
    step = int(FS * DURATION)
    decoded_text = ""
    window = np.hamming(step)  # Hamming window to prevent leakage
    THRESHOLD = 5.0
    last_char = None
    for i in range(0, len(signal), step):
        chunk = signal[i : i + step]
        if len(chunk) < step:
            break
        chunk_windowed = chunk * window
        best_char, max_mag = "", 0
        for ch, (f1, f2) in freq_map.items():
            mag = goertzel(chunk_windowed, f1, FS) + goertzel(chunk_windowed, f2, FS)
            if mag > max_mag:
                max_mag, best_char = mag, ch
        if max_mag > THRESHOLD:
            if best_char != last_char:
                decoded_text += best_char
                last_char = best_char
        else:
            last_char = None
    return decoded_text


class DTMFApp:
    def __init__(self, root):
        self.root = root
        self.root.title("İSTÜN DTMF - Group 14")
        tk.Label(root, text="Enter Turkish Text:").pack(pady=5)
        self.entry = tk.Entry(root, width=40)
        self.entry.pack(pady=5)
        tk.Button(root, text="Encode, Play & Show Graphs", command=self.process).pack(
            pady=5
        )
        self.output_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
        self.output_label.pack(pady=10)

    def show_graphs(self, signal):
        # Taking first character chunk for visualization
        chunk_size = int(FS * DURATION)
        chunk = signal[:chunk_size]
        t = np.linspace(0, DURATION, chunk_size)

        plt.figure(figsize=(10, 6))

        # Plot 1: Time Domain
        plt.subplot(2, 1, 1)
        plt.plot(t, chunk)
        plt.title("Time Domain Signal (Single Character)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        # Plot 2: Frequency Spectrum (FFT)
        plt.subplot(2, 1, 2)
        fft_data = np.abs(np.fft.fft(chunk * np.hamming(chunk_size)))
        freqs = np.fft.fftfreq(chunk_size, 1 / FS)
        plt.plot(freqs[: chunk_size // 2], fft_data[: chunk_size // 2])
        plt.title("Frequency Spectrum (FFT)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def process(self):
        text = self.entry.get()
        if not text:
            return
        signal = encode_text(text)
        sd.play(signal, FS)
        write("output_group14.wav", FS, (signal * 32767).astype(np.int16))
        self.output_label.config(text=f"Decoded: {decode_signal(signal)}")
        self.show_graphs(signal)  # Display report graphs


if __name__ == "__main__":
    root = tk.Tk()
    app = DTMFApp(root)
    root.mainloop()
