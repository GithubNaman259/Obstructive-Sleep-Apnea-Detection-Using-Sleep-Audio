import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from tqdm import tqdm

# =====================================================
#                 PATH SETTINGS
# =====================================================
BASE_DIR = r"A:\ML project\final_preprocessed_events"
OUTPUT_DIR = r"A:\ML project\FEATURES"

MFCC_DIR = os.path.join(OUTPUT_DIR, "MFCC_png")
LFCC_DIR = os.path.join(OUTPUT_DIR, "LFCC_png")
LOGMEL_DIR = os.path.join(OUTPUT_DIR, "LOGMEL_png")
CQCC_DIR = os.path.join(OUTPUT_DIR, "CQCC_png")
NPY_DIR = os.path.join(OUTPUT_DIR, "numpy")

for d in [MFCC_DIR, LFCC_DIR, LOGMEL_DIR, CQCC_DIR, NPY_DIR]:
    os.makedirs(d, exist_ok=True)

# =====================================================
#              FEATURE SETTINGS
# =====================================================
target_sr = 16000
n_mfcc = 40
n_fft = 512
hop = 256
n_mels = 64
cqt_bins = 84
fixed_frames = 400   # ALL FEATURES WILL BE PADDED TO THIS SHAPE

# =====================================================
#               HELPER FUNCTIONS
# =====================================================

def pad_frames(mat, max_frames=fixed_frames):
    """Pads or crops feature matrices to fixed length."""
    if mat.shape[1] < max_frames:
        pad = max_frames - mat.shape[1]
        return np.pad(mat, ((0, 0), (0, pad)), mode="constant")
    return mat[:, :max_frames]


def extract_lfcc(audio, sr):
    """Extract LFCC using linear filterbanks."""
    S = np.abs(librosa.stft(y=audio, n_fft=n_fft, hop_length=hop)) ** 2

    n_filters = 40
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    linear_fb = np.linspace(0, sr/2, n_filters + 2)
    fb = np.zeros((n_filters, len(freqs)))

    for i in range(1, n_filters + 1):
        left, center, right = linear_fb[i-1], linear_fb[i], linear_fb[i+1]

        fb[i-1] = np.maximum(0, np.minimum(
            (freqs - left) / (center - left),
            (right - freqs) / (right - center)
        ))

    filtered = np.dot(fb, S)
    filtered = np.where(filtered == 0, 1e-8, filtered)

    log_fb = np.log(filtered)
    lfcc = dct(log_fb, type=2, axis=0, norm='ortho')[:n_mfcc]
    return lfcc


def extract_cqcc(audio, sr):
    """Extract Constant-Q Cepstral Coefficients."""
    CQT = np.abs(librosa.cqt(y=audio, sr=sr, n_bins=cqt_bins))
    log_CQT = np.log(CQT + 1e-8)
    cqcc = dct(log_CQT, type=2, axis=0, norm='ortho')[:n_mfcc]
    return cqcc


# =====================================================
#                 MAIN EXTRACTION
# =====================================================

X_mfcc, X_lfcc, X_logmel, X_cqcc, Y = [], [], [], [], []

print("\n🔍 Scanning preprocessed patient folders...")

for patient in os.listdir(BASE_DIR):
    patient_path = os.path.join(BASE_DIR, patient)
    if not os.path.isdir(patient_path):
        continue

    print(f"\n📁 Processing: {patient}")

    wav_files = [f for f in os.listdir(patient_path) if f.endswith(".wav")]

    for wav_file in tqdm(wav_files):
        label = wav_file.split("_")[1].replace(".wav", "")
        full_path = os.path.join(patient_path, wav_file)

        # LOAD AUDIO
        audio, sr = librosa.load(full_path, sr=target_sr)

        # =============== FEATURES ===============
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        logmel = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=audio, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
            ) + 1e-8
        )
        lfcc = extract_lfcc(audio, sr)
        cqcc = extract_cqcc(audio, sr)

        # =============== FIX SHAPE ===============
        mfcc = pad_frames(mfcc)
        logmel = pad_frames(logmel)
        lfcc = pad_frames(lfcc)
        cqcc = pad_frames(cqcc)

        # =============== SAVE PNG (MFCC only) ===============
        plt.figure(figsize=(6,4))
        librosa.display.specshow(mfcc, x_axis="time")
        plt.title(f"MFCC - {wav_file}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(MFCC_DIR, wav_file.replace(".wav", ".png")))
        plt.close()

        # =============== SAVE ARRAYS ===============
        X_mfcc.append(mfcc)
        X_lfcc.append(lfcc)
        X_logmel.append(logmel)
        X_cqcc.append(cqcc)
        Y.append(label)

# =====================================================
#              SAVE FINAL DATASETS
# =====================================================
np.save(os.path.join(NPY_DIR, "X_mfcc.npy"), np.array(X_mfcc))
np.save(os.path.join(NPY_DIR, "X_lfcc.npy"), np.array(X_lfcc))
np.save(os.path.join(NPY_DIR, "X_logmel.npy"), np.array(X_logmel))
np.save(os.path.join(NPY_DIR, "X_cqcc.npy"), np.array(X_cqcc))
np.save(os.path.join(NPY_DIR, "y_labels.npy"), np.array(Y))

print("\n🎉 FEATURE EXTRACTION COMPLETE!")
print(f"Total clips processed: {len(Y)}")
