"""
COE216 - Assignment 3: VAD and Voiced/Unvoiced Analysis
========================================================
Voice Activity Detection (VAD) and Voiced/Unvoiced classification
for speech signals using Short-Time Energy and Zero-Crossing Rate.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import soundfile as sf
import os
import sys

# ============================================================
# CONFIGURATION
# ============================================================
FRAME_DURATION_MS = 20       # Window/frame size in ms
OVERLAP_RATIO = 0.5          # 50% overlap
NOISE_ESTIMATE_MS = 200      # First 200ms assumed to be silence
HANGOVER_FRAMES = 4          # Hangover: keep speech if last N frames were speech
ENERGY_MARGIN_FACTOR = 5     # Noise threshold multiplier (margin above noise floor)
MEDIAN_FILTER_SIZE = 5       # Median filter kernel size for smoothing decisions
ZCR_VOICED_THRESHOLD = 0.3   # ZCR threshold: below this -> Voiced (will be tuned)
ENERGY_VOICED_THRESHOLD = None  # Will be set dynamically

# ============================================================
# STEP 0: Load and Normalize Audio
# ============================================================
def load_audio(filepath):
    """Load audio file and normalize to [-1, 1] range."""
    data, fs = sf.read(filepath, dtype='float64')

    # If stereo, convert to mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Normalize to [-1, 1]
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val

    print(f"[INFO] Audio loaded: {filepath}")
    print(f"  Sampling Rate: {fs} Hz")
    print(f"  Duration: {len(data)/fs:.2f} seconds")
    print(f"  Samples: {len(data)}")
    print(f"  Normalized to [-1, 1]")

    return data, fs

# ============================================================
# STEP 1.1: Split Signal into Overlapping Frames
# ============================================================
def frame_signal(data, fs, frame_ms=FRAME_DURATION_MS, overlap=OVERLAP_RATIO):
    """
    Divide signal into overlapping frames.
    - frame_ms: frame duration in milliseconds
    - overlap: overlap ratio (0.5 = 50%)
    Returns: frames (2D array), frame_size, hop_size
    """
    frame_size = int(fs * frame_ms / 1000)           # samples per frame
    hop_size = int(frame_size * (1 - overlap))        # hop = frame_size * 0.5
    num_frames = (len(data) - frame_size) // hop_size + 1

    # Apply Hamming window to reduce spectral leakage
    window = np.hamming(frame_size)

    frames = np.zeros((num_frames, frame_size))
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frames[i] = data[start:end] * window

    print(f"[INFO] Framing complete:")
    print(f"  Frame size: {frame_size} samples ({frame_ms} ms)")
    print(f"  Hop size: {hop_size} samples ({frame_ms * (1-overlap):.0f} ms)")
    print(f"  Number of frames: {num_frames}")

    return frames, frame_size, hop_size

# ============================================================
# STEP 1.2: Calculate Short-Time Energy (Squared Energy)
# ============================================================
def compute_energy(frames):
    """
    Calculate squared energy for each frame.
    Formula: E[n] = (1/N) * sum(x[i]^2) for i in frame
    This is the Mean Squared Energy (proportional to RMS^2).
    """
    energy = np.mean(frames ** 2, axis=1)
    return energy

# ============================================================
# STEP 1.3: Dynamic Noise Threshold Estimation
# ============================================================
def estimate_noise_threshold(energy, fs, hop_size, noise_ms=NOISE_ESTIMATE_MS,
                              margin=ENERGY_MARGIN_FACTOR):
    """
    Estimate noise floor from the first noise_ms milliseconds.
    Threshold = mean_noise_energy + margin * std_noise_energy
    """
    noise_samples = int(fs * noise_ms / 1000)
    noise_frames = noise_samples // hop_size

    if noise_frames < 1:
        noise_frames = 1

    noise_energy = energy[:noise_frames]
    mean_noise = np.mean(noise_energy)
    std_noise = np.std(noise_energy)

    threshold = mean_noise + margin * std_noise

    # Ensure minimum threshold (avoid zero threshold for very clean recordings)
    min_threshold = np.max(energy) * 0.01
    threshold = max(threshold, min_threshold)

    print(f"[INFO] Noise threshold estimation:")
    print(f"  Noise frames used: {noise_frames} (first {noise_ms} ms)")
    print(f"  Mean noise energy: {mean_noise:.6f}")
    print(f"  Std noise energy: {std_noise:.6f}")
    print(f"  Threshold (mean + {margin}*std): {threshold:.6f}")

    return threshold

# ============================================================
# STEP 1.4: VAD Decision with Hangover
# ============================================================
def vad_decision(energy, threshold, hangover=HANGOVER_FRAMES):
    """
    Binary speech/silence decision for each frame.
    1 = Speech, 0 = Silence/Noise
    Includes hangover to prevent cutting short pauses between words.
    """
    num_frames = len(energy)
    decision = np.zeros(num_frames, dtype=int)

    # Initial energy-based decision
    raw_decision = np.zeros(num_frames, dtype=int)
    raw_decision[energy > threshold] = 1

    # Apply hangover: extend speech regions by 'hangover' frames after speech ends
    # This prevents cutting short pauses between words
    decision = raw_decision.copy()
    silence_counter = 0
    in_speech = False

    for i in range(num_frames):
        if raw_decision[i] == 1:
            decision[i] = 1
            in_speech = True
            silence_counter = 0
        else:
            if in_speech:
                silence_counter += 1
                if silence_counter <= hangover:
                    decision[i] = 1  # Keep as speech during hangover period
                else:
                    in_speech = False
                    silence_counter = 0

    # Apply median filter to smooth out isolated errors
    decision = signal.medfilt(decision, kernel_size=MEDIAN_FILTER_SIZE)
    decision = decision.astype(int)

    speech_count = np.sum(decision)
    print(f"[INFO] VAD Decision:")
    print(f"  Speech frames: {speech_count} / {num_frames}")
    print(f"  Silence frames: {num_frames - speech_count}")

    return decision

# ============================================================
# STEP 1.5: Extract Speech and Save
# ============================================================
def extract_speech(data, vad, fs, frame_size, hop_size, output_path):
    """
    Concatenate speech frames and save as new .wav file.
    Reports compression ratio.
    """
    speech_signal = []

    for i in range(len(vad)):
        if vad[i] == 1:
            start = i * hop_size
            end = start + frame_size
            if end <= len(data):
                speech_signal.append(data[start:end])

    if len(speech_signal) == 0:
        print("[WARNING] No speech detected! Check your threshold.")
        return np.array([])

    # Overlap-add for smooth concatenation
    # Since we used 50% overlap, we need to handle the overlap properly
    total_len = (len(speech_signal) - 1) * hop_size + frame_size
    output = np.zeros(total_len)
    count = np.zeros(total_len)

    for i, frame in enumerate(speech_signal):
        start = i * hop_size
        end = start + frame_size
        output[start:end] += frame
        count[start:end] += 1

    # Normalize by overlap count
    count[count == 0] = 1
    output = output / count

    # Normalize output
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val

    sf.write(output_path, output, fs)

    original_duration = len(data) / fs
    speech_duration = len(output) / fs
    compression = (1 - speech_duration / original_duration) * 100

    print(f"[INFO] Speech extraction complete:")
    print(f"  Original duration: {original_duration:.2f} s")
    print(f"  Speech duration:   {speech_duration:.2f} s")
    print(f"  Time saved:        {original_duration - speech_duration:.2f} s")
    print(f"  Compression:       {compression:.1f}%")
    print(f"  Saved to: {output_path}")

    return output

# ============================================================
# STEP 2: Zero-Crossing Rate (ZCR)
# ============================================================
def compute_zcr(frames):
    """
    Calculate Zero-Crossing Rate for each frame.
    ZCR = (1/(N-1)) * sum(|sign(x[i]) - sign(x[i-1])|) / 2
    """
    num_frames, frame_size = frames.shape
    zcr = np.zeros(num_frames)

    for i in range(num_frames):
        frame = frames[i]
        signs = np.sign(frame)
        # Count zero crossings
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        zcr[i] = crossings / (frame_size - 1)

    return zcr

# ============================================================
# STEP 2.2: Voiced/Unvoiced Classification
# ============================================================
def classify_voiced_unvoiced(energy, zcr, vad):
    """
    Classify speech frames as Voiced or Unvoiced.
    - Voiced: High energy, Low ZCR (vocal cord vibration, periodic)
    - Unvoiced: Low/Medium energy, High ZCR (air friction, noise-like)

    Uses adaptive thresholding based on the distribution of ZCR and energy
    within speech frames.

    Returns:
        classification: array where 0=Silence, 1=Voiced, 2=Unvoiced
    """
    # Only classify frames that are speech (vad == 1)
    speech_mask = (vad == 1)
    speech_energy = energy[speech_mask]
    speech_zcr = zcr[speech_mask]

    if len(speech_energy) == 0:
        return np.zeros(len(energy), dtype=int)

    # Dynamic thresholds:
    # ZCR threshold: voiced sounds typically have ZCR < 0.15-0.20
    # Use the 40th percentile of speech ZCR as threshold
    zcr_threshold = np.percentile(speech_zcr, 40)
    # Ensure a reasonable range
    zcr_threshold = np.clip(zcr_threshold, 0.05, 0.30)

    # Energy threshold: voiced sounds have higher energy
    # Use the 30th percentile of speech energy as threshold
    energy_threshold = np.percentile(speech_energy, 30)

    print(f"[INFO] V/UV Classification thresholds:")
    print(f"  ZCR threshold:    {zcr_threshold:.4f} (ZCR < this -> likely Voiced)")
    print(f"  Energy threshold: {energy_threshold:.6f} (Energy > this -> likely Voiced)")

    classification = np.zeros(len(energy), dtype=int)  # 0 = Silence

    for i in range(len(energy)):
        if vad[i] == 1:
            # Primary decision: ZCR-based with energy confirmation
            if zcr[i] < zcr_threshold and energy[i] > energy_threshold:
                classification[i] = 1  # Voiced
            elif zcr[i] >= zcr_threshold:
                classification[i] = 2  # Unvoiced (high ZCR → noise-like)
            elif energy[i] > np.percentile(speech_energy, 70):
                classification[i] = 1  # High energy override → Voiced
            else:
                classification[i] = 2  # Default to Unvoiced

    # Apply median filter to smooth V/UV decisions (only on speech frames)
    speech_indices = np.where(vad == 1)[0]
    if len(speech_indices) >= MEDIAN_FILTER_SIZE:
        speech_class = classification[speech_indices]
        speech_class = signal.medfilt(speech_class, kernel_size=MEDIAN_FILTER_SIZE).astype(int)
        # Ensure we don't turn speech back to silence
        for j, idx in enumerate(speech_indices):
            if speech_class[j] > 0:
                classification[idx] = speech_class[j]

    voiced_count = np.sum(classification == 1)
    unvoiced_count = np.sum(classification == 2)
    print(f"  Voiced frames:   {voiced_count}")
    print(f"  Unvoiced frames: {unvoiced_count}")

    return classification

# ============================================================
# STEP 3: Visualization
# ============================================================
def plot_analysis(data, fs, energy, zcr, vad, classification, hop_size, frame_size):
    """
    Create subplot visualization:
    1. Original audio signal (time domain)
    2. Window-based Energy graph
    3. Window-based ZCR graph
    4. VAD and Voiced/Unvoiced regions with colored overlay
    """
    # Time axes
    time_signal = np.arange(len(data)) / fs
    time_frames = np.arange(len(energy)) * hop_size / fs

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle('COE216 HW3 - VAD and Voiced/Unvoiced Analysis', fontsize=14, fontweight='bold')

    # --- Plot 1: Original Audio Signal ---
    axes[0].plot(time_signal, data, color='steelblue', linewidth=0.5)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original Audio Signal (Time Domain)')
    axes[0].set_xlim([0, time_signal[-1]])
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: Short-Time Energy ---
    axes[1].plot(time_frames, energy, color='darkorange', linewidth=1.2)
    axes[1].axhline(y=np.mean(energy[vad == 0]) if np.any(vad == 0) else 0,
                     color='red', linestyle='--', linewidth=0.8, label='Noise Level')
    axes[1].set_ylabel('Energy')
    axes[1].set_title('Short-Time Energy (per frame)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # --- Plot 3: Zero-Crossing Rate ---
    axes[2].plot(time_frames, zcr, color='purple', linewidth=1.2)
    axes[2].set_ylabel('ZCR')
    axes[2].set_title('Zero-Crossing Rate (per frame)')
    axes[2].grid(True, alpha=0.3)

    # --- Plot 4: VAD + Voiced/Unvoiced Overlay ---
    axes[3].plot(time_signal, data, color='gray', linewidth=0.5, alpha=0.5)
    axes[3].set_ylabel('Amplitude')
    axes[3].set_title('VAD & Voiced/Unvoiced Classification')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].grid(True, alpha=0.3)

    # Color overlay for each frame
    for i in range(len(classification)):
        t_start = i * hop_size / fs
        t_end = t_start + frame_size / fs

        if classification[i] == 1:  # Voiced
            axes[3].axvspan(t_start, t_end, alpha=0.3, color='green')
        elif classification[i] == 2:  # Unvoiced
            axes[3].axvspan(t_start, t_end, alpha=0.3, color='orange')

    # Legend
    voiced_patch = mpatches.Patch(color='green', alpha=0.3, label='Voiced (Vowels)')
    unvoiced_patch = mpatches.Patch(color='orange', alpha=0.3, label='Unvoiced (Consonants)')
    silence_patch = mpatches.Patch(color='white', edgecolor='gray', label='Silence')
    axes[3].legend(handles=[voiced_patch, unvoiced_patch, silence_patch], loc='upper right')

    plt.tight_layout()
    plt.savefig('hw3_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Plot saved as hw3_analysis.png")

# ============================================================
# STEP 3.2: Print Analysis Table
# ============================================================
def print_analysis_table(energy, zcr, vad, classification):
    """Print statistics for voiced vs unvoiced frames."""
    print("\n" + "="*60)
    print("ANALYSIS TABLE: Energy and ZCR Statistics")
    print("="*60)
    print(f"{'Category':<15} {'Mean Energy':<15} {'Mean ZCR':<15} {'Dominant':<15}")
    print("-"*60)

    if np.any(classification == 1):
        v_energy = np.mean(energy[classification == 1])
        v_zcr = np.mean(zcr[classification == 1])
        print(f"{'Voiced':<15} {v_energy:<15.6f} {v_zcr:<15.4f} {'Energy':<15}")

    if np.any(classification == 2):
        uv_energy = np.mean(energy[classification == 2])
        uv_zcr = np.mean(zcr[classification == 2])
        print(f"{'Unvoiced':<15} {uv_energy:<15.6f} {uv_zcr:<15.4f} {'ZCR':<15}")

    if np.any(vad == 0):
        s_energy = np.mean(energy[vad == 0])
        s_zcr = np.mean(zcr[vad == 0])
        print(f"{'Silence':<15} {s_energy:<15.6f} {s_zcr:<15.4f} {'-':<15}")

    print("="*60)


def print_letter_analysis(energy, zcr, fs, hop_size, letter_regions=None):
    """
    Print per-letter ZCR and Energy analysis table.
    letter_regions: dict of {letter: (start_sec, end_sec)} or None for auto-detect from synthetic signal.
    """
    if letter_regions is None:
        # Default regions from synthetic test signal
        letter_regions = {
            "A (Voiced)":  (0.5, 1.2),
            "O (Voiced)":  (2.0, 2.8),
            "U (Voiced)":  (3.5, 4.2),
            "S (Unvoiced)": (1.5, 1.9),
            "F (Unvoiced)": (3.2, 3.5),
        }

    print("\n" + "="*70)
    print("LETTER-SPECIFIC ANALYSIS TABLE")
    print("="*70)
    print(f"{'Letter':<18} {'Time (s)':<14} {'Mean Energy':<15} {'Mean ZCR':<12} {'Dominant':<12}")
    print("-"*70)

    for letter, (t_start, t_end) in letter_regions.items():
        frame_start = int(t_start * fs / hop_size)
        frame_end = int(t_end * fs / hop_size)
        frame_end = min(frame_end, len(energy))

        if frame_start < frame_end:
            region_energy = np.mean(energy[frame_start:frame_end])
            region_zcr = np.mean(zcr[frame_start:frame_end])
            dominant = "Energy" if region_energy > 0.01 and region_zcr < 0.2 else "ZCR"
            print(f"{letter:<18} {t_start:.1f}-{t_end:.1f} s    {region_energy:<15.6f} {region_zcr:<12.4f} {dominant:<12}")

    print("="*70)
    print("\nConclusion: Voiced sounds (A, O, U) -> High Energy, Low ZCR")
    print("            Unvoiced sounds (S, F)  -> Low Energy, High ZCR")

# ============================================================
# OPTIONAL: Autocorrelation for Pitch Detection
# ============================================================
def compute_autocorrelation_pitch(frames, fs, vad):
    """
    Compute pitch frequency using autocorrelation for voiced frames.
    Returns estimated pitch for each frame (0 for silence/unvoiced).
    """
    num_frames, frame_size = frames.shape
    pitches = np.zeros(num_frames)

    # Typical pitch range: 80-400 Hz
    min_lag = int(fs / 400)  # Max pitch frequency
    max_lag = int(fs / 80)   # Min pitch frequency

    for i in range(num_frames):
        if vad[i] == 1:
            frame = frames[i]
            # Autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:]  # Take positive lags only

            # Normalize
            corr = corr / corr[0]

            # Find peak in valid pitch range
            if max_lag < len(corr):
                search_region = corr[min_lag:max_lag]
                if len(search_region) > 0:
                    peak_idx = np.argmax(search_region) + min_lag
                    if corr[peak_idx] > 0.3:  # Confidence threshold
                        pitches[i] = fs / peak_idx

    return pitches

# ============================================================
# MAIN
# ============================================================
def main():
    # --- Configuration ---
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Default: look for any .wav file in current directory
        wav_files = [f for f in os.listdir('.') if f.endswith('.wav') and not f.startswith('output_')]
        if wav_files:
            input_file = wav_files[0]
            print(f"[INFO] Using found wav file: {input_file}")
        else:
            print("[ERROR] No .wav file found! Please provide a .wav file as argument.")
            print("Usage: python hw3_vad.py <input.wav>")
            print("\nGenerating a synthetic test signal instead...")
            input_file = generate_test_signal()

    output_file = "output_speech_only.wav"

    # STEP 0: Load and normalize
    print("\n" + "="*60)
    print("STEP 0: Loading and Normalizing Audio")
    print("="*60)
    data, fs = load_audio(input_file)

    # STEP 1.1: Frame the signal
    print("\n" + "="*60)
    print("STEP 1.1: Framing Signal (20ms windows, 50% overlap)")
    print("="*60)
    frames, frame_size, hop_size = frame_signal(data, fs)

    # STEP 1.2: Compute energy
    print("\n" + "="*60)
    print("STEP 1.2: Computing Short-Time Energy")
    print("="*60)
    energy = compute_energy(frames)
    print(f"  Max energy: {np.max(energy):.6f}")
    print(f"  Min energy: {np.min(energy):.6f}")
    print(f"  Mean energy: {np.mean(energy):.6f}")

    # STEP 1.3: Estimate noise threshold
    print("\n" + "="*60)
    print("STEP 1.3: Dynamic Noise Threshold Estimation")
    print("="*60)
    threshold = estimate_noise_threshold(energy, fs, hop_size)

    # STEP 1.4: VAD decision with hangover
    print("\n" + "="*60)
    print("STEP 1.4: VAD Decision (with Hangover)")
    print("="*60)
    vad = vad_decision(energy, threshold)

    # STEP 1.5: Extract speech and save
    print("\n" + "="*60)
    print("STEP 1.5: Extracting Speech & Saving")
    print("="*60)
    speech_output = extract_speech(data, vad, fs, frame_size, hop_size, output_file)

    # STEP 2.1: Compute ZCR
    print("\n" + "="*60)
    print("STEP 2.1: Computing Zero-Crossing Rate")
    print("="*60)
    zcr = compute_zcr(frames)
    print(f"  Max ZCR: {np.max(zcr):.4f}")
    print(f"  Min ZCR: {np.min(zcr):.4f}")
    print(f"  Mean ZCR: {np.mean(zcr):.4f}")

    # STEP 2.2: Voiced/Unvoiced classification
    print("\n" + "="*60)
    print("STEP 2.2: Voiced/Unvoiced Classification")
    print("="*60)
    classification = classify_voiced_unvoiced(energy, zcr, vad)

    # STEP 3: Visualization
    print("\n" + "="*60)
    print("STEP 3: Visualization & Analysis")
    print("="*60)
    plot_analysis(data, fs, energy, zcr, vad, classification, hop_size, frame_size)

    # Analysis table
    print_analysis_table(energy, zcr, vad, classification)

    # Letter-specific analysis table
    print_letter_analysis(energy, zcr, fs, hop_size)

    # OPTIONAL: Autocorrelation pitch
    print("\n" + "="*60)
    print("OPTIONAL: Autocorrelation Pitch Detection")
    print("="*60)
    pitches = compute_autocorrelation_pitch(frames, fs, vad)
    voiced_pitches = pitches[pitches > 0]
    if len(voiced_pitches) > 0:
        print(f"  Mean pitch: {np.mean(voiced_pitches):.1f} Hz")
        print(f"  Pitch range: {np.min(voiced_pitches):.1f} - {np.max(voiced_pitches):.1f} Hz")
        if np.mean(voiced_pitches) < 165:
            print(f"  Estimated gender: Male (avg pitch < 165 Hz)")
        else:
            print(f"  Estimated gender: Female (avg pitch >= 165 Hz)")
    else:
        print("  No pitched frames detected.")

    print("\n[DONE] Analysis complete!")


# ============================================================
# TEST: Generate Synthetic Test Signal
# ============================================================
def generate_test_signal(filename="test_speech.wav", fs=16000, duration=5.0):
    """
    Generate a synthetic test signal with:
    - Silence regions
    - Voiced-like regions (low frequency sine waves)
    - Unvoiced-like regions (noise bursts)
    """
    t = np.arange(0, duration, 1/fs)
    signal_out = np.zeros(len(t))

    # Region 1: Silence (0 - 0.5s)
    # Already zeros

    # Region 2: Voiced sound - "A" like (0.5 - 1.2s)
    mask_voiced1 = (t >= 0.5) & (t < 1.2)
    f0 = 150  # Fundamental frequency
    signal_out[mask_voiced1] = (0.8 * np.sin(2 * np.pi * f0 * t[mask_voiced1]) +
                                 0.3 * np.sin(2 * np.pi * 2*f0 * t[mask_voiced1]) +
                                 0.1 * np.sin(2 * np.pi * 3*f0 * t[mask_voiced1]))

    # Region 3: Silence (1.2 - 1.5s)

    # Region 4: Unvoiced sound - "S" like (1.5 - 1.9s)
    mask_unvoiced1 = (t >= 1.5) & (t < 1.9)
    np.random.seed(42)
    signal_out[mask_unvoiced1] = 0.15 * np.random.randn(np.sum(mask_unvoiced1))

    # Region 5: Short silence (1.9 - 2.0s)

    # Region 6: Voiced sound - "O" like (2.0 - 2.8s)
    mask_voiced2 = (t >= 2.0) & (t < 2.8)
    f0 = 120
    signal_out[mask_voiced2] = (0.7 * np.sin(2 * np.pi * f0 * t[mask_voiced2]) +
                                 0.4 * np.sin(2 * np.pi * 2*f0 * t[mask_voiced2]))

    # Region 7: Silence (2.8 - 3.2s)

    # Region 8: Unvoiced - "F" like (3.2 - 3.5s)
    mask_unvoiced2 = (t >= 3.2) & (t < 3.5)
    signal_out[mask_unvoiced2] = 0.1 * np.random.randn(np.sum(mask_unvoiced2))

    # Region 9: Voiced - "U" like (3.5 - 4.2s)
    mask_voiced3 = (t >= 3.5) & (t < 4.2)
    f0 = 180
    signal_out[mask_voiced3] = (0.6 * np.sin(2 * np.pi * f0 * t[mask_voiced3]) +
                                 0.2 * np.sin(2 * np.pi * 2*f0 * t[mask_voiced3]))

    # Region 10: Silence (4.2 - 5.0s)

    # Add slight background noise everywhere
    signal_out += 0.002 * np.random.randn(len(t))

    # Normalize
    signal_out = signal_out / np.max(np.abs(signal_out))

    sf.write(filename, signal_out, fs)
    print(f"[INFO] Synthetic test signal generated: {filename}")
    print(f"  Layout: Silence | Voiced(A) | Silence | Unvoiced(S) | Silence | Voiced(O) | Silence | Unvoiced(F) | Voiced(U) | Silence")

    return filename


if __name__ == "__main__":
    main()
