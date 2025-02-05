import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, welch
import matplotlib.pyplot as plt


data = pd.read_csv('fixeddata.csv')


data = data.dropna().reset_index(drop=True)
print(data.head())

time = data['Time (s)']
acc_x = data['Linear Acceleration x (m/s^2)']
acc_y = data['Linear Acceleration y (m/s^2)']
acc_z = data['Linear Acceleration z (m/s^2)']

lat = data['Latitude (°)']
lon = data['Longitude (°)']
speed = data['Velocity (m/s)']  # in m/s

#Valitaan mitattava akseli
acc = acc_z 

#Suodatetaan data
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

fs = 50  #Valitaan taajuus
filtered_acc = butter_lowpass_filter(acc, cutoff=3, fs=fs)

#Askelmäärä suodatetusta datasta
peaks, _ = find_peaks(filtered_acc, distance=fs*0.5)  # Minimum distance between steps
step_count_filtered = len(peaks)

#Askelmäärä Fourier muunnoksella
f, Pxx = welch(acc, fs, nperseg=1024)
dominant_freq = f[np.argmax(Pxx)]
step_count_fft = int(dominant_freq * len(time) / fs)

#Keskinopeus
average_speed = speed.mean() * 3.6  #km/h

#Kokonaismatka metreinä
distance = speed.sum() / fs

# askelpituus
step_length = distance / step_count_filtered if step_count_filtered > 0 else np.nan

#Luodaan kaavio
plt.figure(figsize=(14, 10))

#Askelmäärä
plt.subplot(3, 1, 1)
plt.plot(time, filtered_acc, label='Filtered Acceleration (Z-axis)')
plt.plot(time[peaks], filtered_acc[peaks], 'r.', markersize=8, label='Detected Steps')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Filtered Acceleration with Step Detection')
plt.legend()

#Tehospektri
plt.subplot(3, 1, 2)
plt.semilogy(f, Pxx)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Power Spectral Density of Acceleration (Z-axis)')

#Kartta
plt.subplot(3, 1, 3)
plt.plot(lon, lat, color='blue', marker='o', markersize=2)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('GPS Route Map')

plt.tight_layout()
plt.show()

print(f"Step Count (Filtered Acceleration): {step_count_filtered}")
print(f"Step Count (FFT Analysis): {step_count_fft}")
print(f"Average Speed (km/h): {average_speed:.2f}")
print(f"Total Distance (meters): {distance:.2f}")
print(f"Step Length (meters): {step_length:.2f}")
