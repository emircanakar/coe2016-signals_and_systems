import numpy as np
import matplotlib.pyplot as plt

# Parametreler
f0 = 68  # f0 = 10+55+03
f1 = f0
f2 = f0 / 2
f3 = 10 * f0
fs = 10000  # (Nyquist > 1360 Hz)

# Her grafikte en az 3 periyot göstermek için sure (T = 1/f)
# En uzun periyot f2'de oldugu icin (1/34), sureyi ona gore ayarlıyoruz.
duration = 3 / f2
t = np.arange(0, duration, 1 / fs)

# Sinyallerin Olusturulması
s1 = np.sin(2 * np.pi * f1 * t)
s2 = np.sin(2 * np.pi * f2 * t)
s3 = np.sin(2 * np.pi * f3 * t)
sum_signal = s1 + s2 + s3

# Gorsellestirme
fig1, axs = plt.subplots(3, 1, figsize=(10, 8))
fig1.suptitle(f'Task 1: Sinusodial Signals (f0 = {f0} Hz)')

axs[0].plot(t, s1)
axs[0].set_title(f'f1 = {f1} Hz')
axs[0].set_xlim([0, 3 / f1])

axs[1].plot(t, s2)
axs[1].set_title(f'f2 = {f2} Hz')
axs[1].set_xlim([0, 3 / f2])

axs[2].plot(t, s3)
axs[2].set_title(f'f3 = {f3} Hz')
axs[2].set_xlim([0, 3 / f3])

for ax in axs:
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')

plt.tight_layout()

# Toplam Sinyal Grafigi
plt.figure(figsize=(10, 4))
plt.plot(t, sum_signal, color='orange')
plt.title('Sum of Three Signals')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim([0, duration])
plt.grid(True)

plt.show()
