import pandas as pd
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

# === 1. Les CSV og håndter separatorproblemer ===
df = pd.read_csv("csv/saved_info_1.csv", sep="[;,]", engine="python", header=None)

# Konverter alle verdier til numerisk (håndterer eventuelle feil)
df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(0)  # Erstatter NaN med 0

# === 2. Ekstraher tid, posisjon og tetthet ===
tidspunkter = df.iloc[:, 0].values  # Første kolonne = tid
tetthet_data = df.iloc[:, 1:].values  # Resten = tetthet

# Definer posisjoner basert på kolonneantall
antall_posisjoner = tetthet_data.shape[1]
posisjoner = np.linspace(0, 1000, antall_posisjoner)  # 1000m vei

# === 3. Interpolasjon av tetthet ===
tetthet_interp = interp.RectBivariateSpline(tidspunkter, posisjoner, tetthet_data)

# === 4. Trafikkmodellparametere ===
v_max = 30  # Maks hastighet (m/s)
rho_max = np.max(tetthet_data)  # Setter maks tetthet til høyeste verdi

# Funksjon for bilhastighet basert på tetthet
def hastighet(t, x):
    rho = tetthet_interp(t, x)[0, 0]  # Interpolert tetthet
    return max(v_max * (1 - rho / rho_max), 0)  # Sikrer ikke-negativ hastighet

# === 5. Simulere bilens bevegelse og finne posisjon som funksjon av tid ===
startposisjon = 50  # Startposisjon for bilene
dt = 0.1  # Tidsskritt
T = 50  # Simuleringstid

# Funksjon som returnerer posisjonene til en bil som starter ved tid t_start
def finn_posisjon_over_tid(t_start):
    x = startposisjon
    t = t_start
    tid_historikk = [t]
    posisjon_historikk = [x]
    
    # Simulering over tid
    while t <= T:
        v = hastighet(t, x)
        x += v * dt
        t += dt
        tid_historikk.append(t)
        posisjon_historikk.append(x)
    
    return tid_historikk, posisjon_historikk

# === 6. Beregn posisjoner for biler som starter på forskjellige tidspunkter ===
tid_historikk_for_biler = {}
for start_tid in tidspunkter:
    tid_historikk, posisjon_historikk = finn_posisjon_over_tid(start_tid)
    tid_historikk_for_biler[start_tid] = (tid_historikk, posisjon_historikk)

# === 7. Plot resultatene ===
plt.figure(figsize=(8, 6))

for start_tid, (tid_historikk, posisjon_historikk) in tid_historikk_for_biler.items():
    plt.plot(tid_historikk, posisjon_historikk, label=f"Start t={start_tid:.1f}")

plt.xlabel("Tid (s)")
plt.ylabel("Posisjon (m)")
plt.title("Posisjoner til biler som starter på ulike tidspunkter")
plt.legend(loc="upper left", fontsize=8)
plt.grid(True)
plt.show()
