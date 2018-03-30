import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

dat = pd.read_csv('Control_2.csv')
print(dat['Step'].values)


def iv_curve_func(V, k, g, Vr, Vs):
    return k*(np.exp((1.-g)*(V-Vr)/Vs)-np.exp(-g*(V-Vr)/Vs))


initial = [96e-12, 0.53, 4.6, 36.5]
# bounds2 = ([0, 0, -10, -50], [1000e-12, 10, 10, 50])
popt, pcov = curve_fit(iv_curve_func, p0=initial, xdata=dat['Step'].values,
                       ydata=dat['Current'].values, method='trf')
print(popt)
err = np.diag(pcov)
fig = plt.figure()
fig.subplots_adjust(top=0.971,
                    bottom=0.091,
                    left=0.067,
                    right=0.986,
                    hspace=0.253,
                    wspace=0.2)
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(dat['Step'].values, dat['Current'].values, 'x', color='k', antialiased=True)
ax1.plot(dat['Step'].values, iv_curve_func(dat['Step'].values, *popt), color='r', antialiased=True)
ax1.grid(color='k', linestyle='--', alpha=0.2)
ax1.axhline(0, color='k', linewidth='0.5')
ax1.axvline(0, color='k', linewidth='0.5')
plt.xlabel('Voltage (mV)')
plt.ylabel('Current (pA)')

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(dat['Step'].values, dat['Current'].values-iv_curve_func(dat['Step'].values, *popt),
         '.', color='k', antialiased=True)
ax2.grid(color='k', linestyle='--', alpha=0.2)
ax2.axhline(0, color='k', linewidth='0.5')
ax2.axvline(0, color='k', linewidth='0.5')
plt.xlabel('Voltage (mV)')
plt.ylabel('Im - If (pA)')
fig_manager = plt.get_current_fig_manager()
fig_manager.window.showMaximized()
# plt.tight_layout()
plt.show()
