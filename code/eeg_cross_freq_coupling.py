"""
Cross-Frequency EEG Coupling as a Real-Time Learning Metric
Theta-Alpha Phase-Amplitude Coupling for Knowledge Acquisition Tracking
"""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

def bandpass_filter(signal, lowcut, highcut, fs=256, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)

def phase_amplitude_coupling(eeg, fs=256):
    """Compute Theta-Alpha PAC using Modulation Index."""
    theta = bandpass_filter(eeg, 4, 8, fs)
    alpha = bandpass_filter(eeg, 8, 13, fs)
    theta_phase = np.angle(hilbert(theta))
    alpha_amp   = np.abs(hilbert(alpha))
    n_bins = 18
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amp = np.array([
        alpha_amp[(theta_phase >= phase_bins[i]) & (theta_phase < phase_bins[i+1])].mean()
        for i in range(n_bins)
    ])
    mean_amp /= (mean_amp.sum() + 1e-10)
    MI = np.sum(mean_amp * np.log(mean_amp * n_bins + 1e-10)) / np.log(n_bins)
    return float(MI)

def sliding_window_pac(eeg, fs=256, window_sec=2.0, step_sec=0.5):
    """PAC trajectory across a session."""
    win = int(window_sec * fs)
    step = int(step_sec * fs)
    indices = range(0, len(eeg) - win, step)
    pac_series = [phase_amplitude_coupling(eeg[i:i+win], fs) for i in indices]
    times = [i / fs for i in indices]
    return np.array(times), np.array(pac_series)

def simulate_learning_session(duration_sec=300, fs=256, seed=42):
    np.random.seed(seed)
    n = duration_sec * fs
    t = np.linspace(0, duration_sec, n)
    theta_strength = 0.5 + 0.5 * (t / duration_sec)
    alpha_strength = 0.3 + 0.4 * (t / duration_sec)
    signal = (theta_strength * np.sin(2 * np.pi * 6 * t) +
              alpha_strength * np.sin(2 * np.pi * 10 * t) +
              0.3 * np.random.randn(n))
    return signal

if __name__ == "__main__":
    print("Simulating 5-minute EEG learning session...")
    eeg = simulate_learning_session(duration_sec=300)
    times, pac = sliding_window_pac(eeg)
    print(f"  Samples   : {len(eeg)}")
    print(f"  PAC windows: {len(pac)}")
    print(f"  Mean PAC  : {pac.mean():.4f}")
    print(f"  Max PAC   : {pac.max():.4f}  at t={times[pac.argmax()]:.1f}s")
    print("EEG Coupling analysis complete.")
