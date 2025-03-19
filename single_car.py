import pandas as pd
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

p = 1

df = pd.read_csv("csv/saved_info_"+str(p)+".csv", sep="[;,]", engine="python", header=None)

df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(0)  # Erstatter eventuelle NaN med 0

tidspunkter = df.iloc[:, 0].values  # Første kolonne = tid
tetthet_data = df.iloc[:, 3:].values  # Resten = tetthet

# Sjekk at tetthet_data ikke er tomt
if tetthet_data.size == 0:
    raise ValueError("Tetthet-data er tom! Sjekk at CSV-filen er riktig.")

# Anta jevn posisjonsfordeling basert på antall kolonner
antall_posisjoner = tetthet_data.shape[1]
posisjoner = np.linspace(0, 1000, antall_posisjoner)  # 1000m vei

tetthet_interp = interp.RectBivariateSpline(tidspunkter, posisjoner, tetthet_data)

v_max = 22.2  # Maks hastighet (m/s)
u_max = 0.2  # Setter maks tetthet til den høyeste i datasettet

def hastighet(t, x):
    u = tetthet_interp(t, x)[0, 0]  # Henter interpolert tetthet
    v = v_max * (1 - (u / u_max)**p)
    if v > v_max:
        v = v_max
    return v

x = -1000
t = tidspunkter[0]
dt = 0.01
T = 150

posisjon_historikk = [x]
tid_historikk = [t]
hastighet_historikk = [hastighet(t, x)]

for _ in range(int(T / dt)):
    v = hastighet(t, x)
    
    x += v * dt
    t += dt
    posisjon_historikk.append(x)
    tid_historikk.append(t)
    hastighet_historikk.append(v)

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

axs[0].plot(tid_historikk, hastighet_historikk, color='r', label="Fart")
axs[0].set_ylabel("Hastighet (m/s)")
axs[0].set_title(f"Hastighet og Posisjon over Tid med modell {p = }")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(tid_historikk, posisjon_historikk, color='b', label="Posisjon")
axs[1].set_xlabel("Tid (s)")
axs[1].set_ylabel("Posisjon (m)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
