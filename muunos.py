import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file = pd.read_csv('data.csv')

x_akseli = file['Aika']
y_akseli = file['Signaali']

fig, ax = plt.subplots()
ax.plot(x_akseli, y_akseli, color='g', label="Alkuperäineen signaali")
ax.set_xlabel('Aika')
ax.set_ylabel('Signaali')
ax.legend()
plt.savefig('signaali.png')

fourier_muunnos = np.fft.fft(y_akseli)
taaajuudet = np.fft.fftfreq(len(y_akseli), x_akseli[1] - x_akseli[0])

tehospektri = np.abs(fourier_muunnos)**2

fig, ax = plt.subplots()
ax.plot(taaajuudet[:len(taaajuudet)//2], tehospektri[:len(tehospektri)//2], color='b', label="Tehospektri")
ax.set_xlabel('signaali')
ax.set_ylabel('Teho')
ax.legend()
plt.savefig('tehospektri.png')
print("Onnistui")

