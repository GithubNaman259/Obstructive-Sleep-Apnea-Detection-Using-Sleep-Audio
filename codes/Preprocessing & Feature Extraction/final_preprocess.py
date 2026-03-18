import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

# ---------- SETTINGS ----------
base_dir = r"A:\\ML project\\APSAA"             # main dataset folder (contains 35 patient folders)
output_dir = r"A:\\ML project\\final_preprocessed_events"  # where preprocessed clips will be saved
target_sr = 16000      # resample rate
fixed_length = 10.0    # seconds

os.makedirs(output_dir, exist_ok=True)

# ---------- FUNCTIONS ----------
def normalize_audio(sig):
    """Scale waveform to [-1,1]."""
    return sig / np.max(np.abs(sig)) if np.max(np.abs(sig)) > 0 else sig

def pad_or_crop(sig, sr, target_sec):
    """Make fixed length by padding or cropping."""
    target_len = int(target_sec * sr)
    if len(sig) < target_len:
        pad = target_len - len(sig)
        sig = np.pad(sig, (0, pad), mode="constant")
    else:
        sig = sig[:target_len]
    return sig

# ---------- MAIN LOOP ----------
for patient_folder in os.listdir(base_dir):
    patient_path = os.path.join(base_dir, patient_folder)
    if not os.path.isdir(patient_path):
        continue

    # Find .wav and annotation file
    wav_file = None
    anno_file = None
    for f in os.listdir(patient_path):
        if f.endswith(".wav"):
            wav_file = os.path.join(patient_path, f)
        elif f.endswith("_Annotations.csv"):
            anno_file = os.path.join(patient_path, f)

    if not wav_file or not anno_file:
        print(f"⚠️ Skipping {patient_folder}: Missing files")
        continue

    print(f"\n📂 Processing patient: {patient_folder}")
    print(f"   WAV:  {os.path.basename(wav_file)}")
    print(f"   ANNO: {os.path.basename(anno_file)}")

    # ---------- LOAD ----------
    audio, sr = librosa.load(wav_file, sr=target_sr)
    ann = pd.read_csv(anno_file)
    ann["Start_Time"] = pd.to_timedelta(ann["Start_Time"]).dt.total_seconds()

    print(f"   Audio length: {len(audio)/sr:.1f}s, Events: {len(ann)}")

    # ---------- PROCESS EVENTS ----------
    patient_out_dir = os.path.join(output_dir, patient_folder)
    os.makedirs(patient_out_dir, exist_ok=True)

    for i, row in ann.iterrows():
        label = row["Event_Name"]
        start = row["Start_Time"]
        dur = row["Duration"]

        start_sample = int(start * sr)
        end_sample = int((start + dur) * sr)

        if end_sample > len(audio):
            print(f"⚠️ Skipping event {i+1}: out of range ({label})")
            continue

        clip = audio[start_sample:end_sample]
        clip = normalize_audio(clip)
        clip = pad_or_crop(clip, sr, fixed_length)

        out_wav = os.path.join(patient_out_dir, f"{i+1:03d}_{label}.wav")
        sf.write(out_wav, clip, sr)
        print(f"💾 Saved {out_wav} ({label})")

print("\n🎯 All patients processed successfully!")
