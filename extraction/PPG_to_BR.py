#VTPPG con aggiunta di cubic spline interpolation, level-dependent soft thresholding
# ESTRAZIONE DEI FRAME DAL VIDEO

import cv2
import os
import glob
import numpy as np
import pywt
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import scipy
import scipy.signal as sig
from scipy.interpolate import CubicSpline 
import copy


video_path = "percorso_video.mp4" #inserire il percorso al video di interesse
output_folder = "percorso_cartella" #inserire il percorso alla cartella in cui si vogliono salvare i frame estratti dal video
folder_path = "percorso_cartella/*.jpg" #inserire la stessa cartella della riga precedente prima di /*.jpg
percorso_csv = "percorso_PPG.csv" #inserire il file .csv in cui salvare il segnale respiratorio ricavato
START = 0 #inserire il frame da cui si intente partire con l'estrazione del ppg
END = None #inserire il numero dell'ultimo frame di interesse
#se si è interessati al video intero START = 0 END = None
#se si vogliono scartare gli ultimi x frame settare END = -x

def extract_frames(video_path, output_folder, frame_interval=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Errore: impossibile aprire il video.")
        return

    frame_count = 0
    saved_count = 0
    timestamps = [] # Array per salvare i tempi reali

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Estraiamo il tempo reale del frame
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamps.append(timestamp_ms)

            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    
    # Salviamo i timestamp per usarli dopo il taglio
    df_times = pd.DataFrame({"timestamp_ms": timestamps})
    df_times.to_csv(os.path.join(output_folder, "timestamps_reali.csv"), index=False)
    
    print(f"Estratti {saved_count} frame in {output_folder}")

#extract_frames(video_path, output_folder) 

# ELABORAZIONE ROI E NORMALIZZAZIONE

all_files = sorted(glob.glob(folder_path))
files = all_files[START:END]

# Controllo di sicurezza: verifica che ci siano immagini!
if len(files) == 0:
    raise ValueError("Nessuna immagine trovata dopo il taglio! Controlla il percorso o avvia extract_frames().")

avg_color_roi = np.zeros((len(files), 3))
somma = np.zeros(3)

for i, file in enumerate(files):
    img = cv2.imread(file)
    
    if i == 0:
        dim_1, dim_2, _ = img.shape
        color = ("b", "g", "r")
        
    img = img[dim_1//2-50:dim_1//2+50, dim_2//2-50:dim_2//2+50, :]
    
    for j, col in enumerate(color):
        canale_flat = img[:, :, j].flatten()
        limite_basso = np.percentile(canale_flat, 5)
        limite_alto = np.percentile(canale_flat, 95)
        mask = (canale_flat >= limite_basso) & (canale_flat <= limite_alto)
        pixel_validi = canale_flat[mask]
        
        media_pulita = np.mean(pixel_validi)
        avg_color_roi[i, j] = media_pulita
        somma[j] += media_pulita
        
    if i % 50 == 0:
        print(f"Elaborato frame {i}/{len(files)}")

# Normalizzazione Zero-Mean
avg_color_roi = avg_color_roi - (somma / len(files))

percorso_tempi = os.path.join(output_folder, "timestamps_reali.csv")
if os.path.exists(percorso_tempi):
    # Carichiamo i tempi reali e applichiamo lo stesso taglio dei file !!
    df_times = pd.read_csv(percorso_tempi)
    tempi_grezzi = df_times['timestamp_ms'].values[START:END]
else:
    print("ATTENZIONE: File timestamp non trovato. Fallback ai tempi teorici. Avvia extract_frames!")
    fps_target = 30.0
    passo_ms = 1000.0 / fps_target
    tempi_grezzi = np.arange(0, len(files) * passo_ms, passo_ms)[:len(files)]

# Griglia di campionamento uniforme a 30 Hz
fps_target = 30.0
passo_ms = 1000.0 / fps_target
tempi_30hz = np.arange(tempi_grezzi[0], tempi_grezzi[-1], passo_ms)

avg_color_roi_interp = np.zeros((len(tempi_30hz), 3))
for c in range(3):
    # Usiamo CubicSpline per l'interpolazione
    cs = CubicSpline(tempi_grezzi, avg_color_roi[:, c])
    avg_color_roi_interp[:, c] = cs(tempi_30hz)

# Sovrascriviamo avg_color_roi per lasciare tutto il resto del codice inalterato
avg_color_roi = avg_color_roi_interp 
# --- FINE AGGIUNTA ---


# WAVELET DENOISING
def wavelet_denoising_bandpass(signal, wavelet_type='db8', level=7):
    # Clipping temporale - #Clipping 5 - 95% al segnale
    low_lim = np.percentile(signal, 5)
    high_lim = np.percentile(signal, 95)
    signal_clipped = np.clip(signal, low_lim, high_lim)
    
    # Scomposizione
    coeffs = pywt.wavedec(signal_clipped, wavelet_type, level=level)
    
    # Calcolo Sigma 
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    
    # MODIFICA: LEVEL-DEPENDENT SOFT THRESHOLDING
    soft_thr_normal = sigma * 2.0  # Soglia per i livelli del battito cardiaco
    soft_thr_huge = sigma * 200   # Soglia gigante per attenuare le altre bande
    soft_thr_lowf = sigma * 2000
 # Evita soglie nulle
    if soft_thr_normal == 0.0:
      soft_thr_normal = 1e-8 
      soft_thr_huge = 1e-7

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        # 1. Attenuazione forte su cA5 (0 - 0.46 Hz)
        coeffs[0] = pywt.threshold(coeffs[0], value=soft_thr_lowf, mode='soft')
        
        # 2. Soft Thresholding normale sui livelli centrali cD5, cD4, cD3 (banda cardiaca utile)
        for i in (1,2):
          coeffs[i] = pywt.threshold(coeffs[i], value=soft_thr_normal, mode='soft')
          
        # 3. Attenuazione forte su cD2 e cD1 (alte frequenze, > 3.75 Hz)
        coeffs[-2] = pywt.threshold(coeffs[-2], value=soft_thr_huge, mode='soft')
        coeffs[-1] = pywt.threshold(coeffs[-1], value=soft_thr_huge, mode='soft')
        coeffs[-3] = pywt.threshold(coeffs[-3], value=soft_thr_huge, mode='soft')
        coeffs[-4] = pywt.threshold(coeffs[-4], value=soft_thr_huge, mode='soft')
        coeffs[-5] = pywt.threshold(coeffs[-5], value=soft_thr_huge, mode='soft')
        
    # Ricostruzione
    denoised_signal = pywt.waverec(coeffs, wavelet_type)

    # Restituisce il segnale troncato alla lunghezza originale
    return denoised_signal[:len(signal)]


# --- 1. FUNZIONE DI NORMALIZZAZIONE ---
def normalize_minus1_plus1(signal):
    s_min = np.min(signal)
    s_max = np.max(signal)
    if s_max - s_min == 0:
        return np.zeros_like(signal)
    # Formula: 2 * (x - min) / (max - min) - 1
    return 2 * ((signal - s_min) / (s_max - s_min)) - 1


# --- 2. ESTRAZIONE, PULIZIA E NORMALIZZAZIONE ---
segnale_blu_invertito = avg_color_roi[:, 0] * -1
segnale_verde_invertito = avg_color_roi[:, 1] * -1
segnale_rosso_invertito = avg_color_roi[:, 2] * -1

segnali_grezzi = [segnale_blu_invertito, segnale_verde_invertito, segnale_rosso_invertito]
segnali_puliti_norm = []
segnali_grezzi_norm = []

for segnale in segnali_grezzi:
    # Applichiamo Wavelet
    pulito = wavelet_denoising_bandpass(signal=segnale)
    
    # Applichiamo la normalizzazione [-1, 1] a entrambi per coerenza
    segnali_puliti_norm.append(normalize_minus1_plus1(pulito))
    segnali_grezzi_norm.append(normalize_minus1_plus1(segnale))


# --- 3. SALVATAGGIO CSV (DATI NORMALIZZATI) ---
df_output = pd.DataFrame({
    "blue": segnali_puliti_norm[0],
    "green": segnali_puliti_norm[1],
    "red": segnali_puliti_norm[2]
})


df_output.to_csv(percorso_csv, index=True)
print(f"\n--- FATTO! Dati normalizzati salvati in: {percorso_csv} ---")


# --- 4. CREAZIONE DEI GRAFICI ---
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
colori_plot = ['blue', 'green', 'red']
nomi_canali = ['Blu', 'Verde', 'Rosso']

for i in range(3):
    ax1 = axes[i]             
    ax2 = ax1.twinx()         

    # Tracciamo il segnale GREZZO NORMALIZZATO
    l1 = ax1.plot(segnali_grezzi_norm[i], color='gray', alpha=0.3, label=f'Grezzo {nomi_canali[i]} (Norm)')
    ax1.set_ylabel('Intensità Norm.', color='gray', fontsize=10)
    ax1.set_ylim(-1.1, 1.1) # Fissiamo il range visivo

    # Tracciamo il segnale PULITO NORMALIZZATO
    l2 = ax2.plot(segnali_puliti_norm[i], color=colori_plot[i], linewidth=2, label=f'Onda PPG {nomi_canali[i]} (Norm)')
    ax2.set_ylabel(f'Variazione {nomi_canali[i]}', color=colori_plot[i], fontsize=10)
    ax2.set_ylim(-1.1, 1.1) # Fissiamo il range visivo

    ax1.set_title(f"Analisi Canale {nomi_canali[i]} (Range [-1, 1])", fontsize=12, fontweight='bold', color=colori_plot[i])
    
    linee = l1 + l2
    labels = [l.get_label() for l in linee]
    ax2.legend(linee, labels, loc='upper right')

axes[-1].set_xlabel("Numero del Campione Interpolato", fontsize=12)
plt.tight_layout()
plt.show()
