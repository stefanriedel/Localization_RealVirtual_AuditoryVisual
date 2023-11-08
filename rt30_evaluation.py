import numpy as np
import matplotlib.pyplot as plt

OCTAVE_BANDS = True
if OCTAVE_BANDS:
    rt30_data = np.loadtxt(
        'Utility/RT30_Omni_ExpStudio_octave_bands.txt')
else:
    rt30_data = np.loadtxt(
        'Utility/RT30_Omni_ExpStudio.txt')


plt.semilogx(rt30_data[:, 0], rt30_data[:, 5])
plt.xlim(100, 8000)
plt.xlabel('Frequency in Hz')

plt.ylim(0, 0.5)
plt.ylabel('RT30 in seconds')
plt.show()

idx_low = np.where(rt30_data[:, 0] >= 200)[0][0]
idx_high = np.where(rt30_data[:, 0] >= 8000)[0][0]

# Compute average rt30
avg_rt30 = np.mean(rt30_data[idx_low:idx_high+1, 5])

print(avg_rt30)
