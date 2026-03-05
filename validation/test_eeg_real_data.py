"""
Real Data Test — P4/P10: EEG Cognitive Load Validation
Dataset: PhysioNet Mental Arithmetic EEG (Zyma et al., 2019) — eegmat
_1 = background/rest, _2 = mental arithmetic (serial subtraction, cognitive load)
This is directly relevant to cognitive load measurement used in P10.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib.request, os, io, json, warnings
import mne
from scipy import signal, stats

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
np.random.seed(42)

BASE_URL = "https://physionet.org/files/eegmat/1.0.0"
OUT      = os.path.join(os.path.dirname(__file__), "figures_eeg_real")
CACHE    = os.path.join(os.path.dirname(__file__), "eeg_cache")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

print("=" * 60)
print("  REAL DATA TEST: EEG Cognitive Load — Mental Arithmetic")
print("  Dataset: PhysioNet eegmat (Zyma et al., 2019)")
print("    _1 = rest/background  |  _2 = serial subtraction task")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD EDF FILES
# ─────────────────────────────────────────────────────────────────────────────
# eegmat: Subject00–Subject35, _1 = rest, _2 = arithmetic task
SUBJECTS   = [f"Subject{i:02d}" for i in range(10)]  # first 10
RUNS       = ["_1", "_2"]
RUNS_LABEL = {"_1": "Rest (background)", "_2": "Mental arithmetic (load)"}

def download_edf(subject, run):
    fname = f"{subject}{run}.edf"
    local = os.path.join(CACHE, fname)
    if os.path.exists(local):
        return local
    url = f"{BASE_URL}/{fname}"
    try:
        urllib.request.urlretrieve(url, local)
        return local
    except Exception as e:
        print(f"    WARN: could not download {fname}: {e}")
        return None

print(f"\n[1] Downloading EDF files for {len(SUBJECTS)} subjects × 2 conditions (rest + arithmetic)...")
downloaded = {run: 0 for run in RUNS}
for subj in SUBJECTS:
    for run in RUNS:
        path = download_edf(subj, run)
        if path:
            downloaded[run] += 1
print(f"    Downloaded: rest={downloaded['_1']}, arithmetic={downloaded['_2']}")

# ─────────────────────────────────────────────────────────────────────────────
# BAND POWER COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
THETA_BAND = (4, 8)      # Hz
ALPHA_BAND = (8, 13)     # Hz
BETA_BAND  = (13, 30)    # Hz

# Frontal channels for theta, parietal/occipital for alpha
FRONTAL_CANDS  = ["FC5","FC3","FC1","FCz","FC2","FC4","FC6","F3","Fz","F4","AF3","AF4"]
PARIETAL_CANDS = ["P3","Pz","P4","PO3","PO4","POz","CP1","CP2","CPz"]

def compute_band_power(data, sfreq, fmin, fmax):
    """Compute mean band power using Welch's method."""
    nperseg = min(int(sfreq * 2), data.shape[-1])
    f, psd = signal.welch(data, fs=sfreq, nperseg=nperseg)
    idx = np.logical_and(f >= fmin, f <= fmax)
    return float(np.mean(psd[:, idx]))

def extract_features_from_edf(filepath):
    """Read EDF, pick channels, compute theta/alpha/beta powers and ratios."""
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    ch_names_upper = [ch.upper() for ch in raw.ch_names]
    
    # Select frontal channels
    frontal_idx  = [i for i, c in enumerate(ch_names_upper) if c in [f.upper() for f in FRONTAL_CANDS]]
    parietal_idx = [i for i, c in enumerate(ch_names_upper) if c in [p.upper() for p in PARIETAL_CANDS]]
    
    if not frontal_idx:
        frontal_idx = list(range(min(8, len(raw.ch_names))))
    if not parietal_idx:
        parietal_idx = list(range(8, min(16, len(raw.ch_names))))
    
    data = raw.get_data()  # (n_channels, n_times)
    
    frontal_data  = data[frontal_idx,  :]
    parietal_data = data[parietal_idx, :]
    
    # Apply bandpass filter
    for ch_data in [frontal_data, parietal_data]:
        pass   # Welch handles this
    
    theta_front = compute_band_power(frontal_data,  sfreq, *THETA_BAND)
    alpha_front = compute_band_power(frontal_data,  sfreq, *ALPHA_BAND)
    alpha_par   = compute_band_power(parietal_data, sfreq, *ALPHA_BAND)
    beta_front  = compute_band_power(frontal_data,  sfreq, *BETA_BAND)
    
    # Log-transform (EEG power is log-normally distributed)
    log_theta = np.log10(max(1e-30, theta_front))
    log_alpha = np.log10(max(1e-30, alpha_par))
    log_beta  = np.log10(max(1e-30, beta_front))
    
    ta_ratio = (theta_front + 1e-30) / (alpha_par + 1e-30)  # key cognitive load proxy
    
    return {
        "theta_frontal_log":  log_theta,
        "alpha_parietal_log": log_alpha,
        "beta_frontal_log":   log_beta,
        "theta_alpha_ratio":  float(ta_ratio),
        "n_frontal_ch":       len(frontal_idx),
        "n_parietal_ch":      len(parietal_idx),
        "sfreq":              float(sfreq),
        "duration_s":         float(data.shape[1] / sfreq),
    }

print(f"\n[2] Extracting EEG band features (theta/alpha/beta)...")
records = []
for subj in SUBJECTS:
    for run in RUNS:
        local = os.path.join(CACHE, f"{subj}{run}.edf")
        if not os.path.exists(local):
            continue
        try:
            feats = extract_features_from_edf(local)
            feats["subject"]   = subj
            feats["run"]       = run
            feats["condition"] = RUNS_LABEL[run]
            records.append(feats)
        except Exception as e:
            print(f"    WARN: {subj}{run}: {e}")

df = pd.DataFrame(records)
print(f"    Extracted: {len(df)} recordings ({df.groupby('run').size().to_dict()})")

# ─────────────────────────────────────────────────
print(f"\n[3] Statistical analysis: rest vs MENTAL ARITHMETIC (cognitive load)...")
rest = df[df["run"] == "_1"]
task = df[df["run"] == "_2"]

n_matched = min(len(rest), len(task))
metrics = ["theta_frontal_log", "alpha_parietal_log", "theta_alpha_ratio"]
metric_labels = ["Frontal θ power (log)", "Parietal α power (log)", "θ/α Ratio"]

stats_results = {}
print(f"\n    {'Metric':30s}  {'Rest (mean±SD)':20s}  {'Task (mean±SD)':20s}  t        p")
print("    " + "-"*85)
for m, lab in zip(metrics, metric_labels):
    r_vals = rest[m].values[:n_matched]
    t_vals = task[m].values[:n_matched]
    t_stat, p_val = stats.ttest_rel(r_vals[:len(t_vals)], t_vals[:len(r_vals)])
    d = (t_vals.mean() - r_vals.mean()) / max(1e-6, np.sqrt(
        (r_vals.std()**2 + t_vals.std()**2)/2))
    stats_results[m] = {"rest_mean": float(r_vals.mean()), "rest_std":  float(r_vals.std()),
                        "task_mean": float(t_vals.mean()), "task_std":  float(t_vals.std()),
                        "t": float(t_stat), "p": float(p_val), "cohens_d": float(d)}
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"    {lab:30s}  {r_vals.mean():+.3f}±{r_vals.std():.3f}        "
          f"{t_vals.mean():+.3f}±{t_vals.std():.3f}      {t_stat:+.2f}  {p_val:.4f} {sig}")

# Compare real theta/alpha ratio to what P10 simulation assumed
sim_theta_alpha_rest = 2.83   # P10 recalibrated baseline (theta_base=14/alpha_base=5 µV², PhysioNet eegmat Zyma 2019)
real_ta_rest = rest["theta_alpha_ratio"].mean()
real_ta_task = task["theta_alpha_ratio"].mean()
print(f"\n    P10 simulation assumed baseline θ/α = {sim_theta_alpha_rest:.2f}")
print(f"    Real EEG baseline θ/α = {real_ta_rest:.3f} (rest)")
print(f"    Real EEG task     θ/α = {real_ta_task:.3f} (arithmetic task)")
print(f"    Simulation calibration error: {abs(real_ta_rest - sim_theta_alpha_rest):.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[4] Generating figures...")
COLORS = ["#2196F3","#E91E63","#4CAF50","#FF9800","#9C27B0","#00BCD4"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Real EEG Validation: PhysioNet eegmat (Zyma et al., 2019)\n"
             f"Subject00-09, Rest vs Mental Arithmetic (n={len(df)} recordings)", fontsize=12, fontweight='bold')

# (0,0) Theta/alpha ratio: rest vs task per subject
subjects_matched = list(set(rest["subject"]) & set(task["subject"]))
ta_rest_per_subj = [rest[rest["subject"]==s]["theta_alpha_ratio"].values[0] for s in subjects_matched]
ta_task_per_subj = [task[task["subject"]==s]["theta_alpha_ratio"].values[0] for s in subjects_matched]
x = range(len(subjects_matched))
axes[0,0].plot(x, ta_rest_per_subj, 'o--', color=COLORS[0], lw=2, label="Rest", markersize=7)
axes[0,0].plot(x, ta_task_per_subj, 's-',  color=COLORS[1], lw=2, label="Mental arithmetic", markersize=7)
axes[0,0].axhline(sim_theta_alpha_rest, color='gray', lw=1.5, linestyle=':', label=f"P10 sim baseline={sim_theta_alpha_rest:.2f}")
axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(subjects_matched, rotation=45, fontsize=8)
axes[0,0].set_title("θ/α Ratio per Subject\n(Real EEG)"); axes[0,0].set_ylabel("θ/α ratio")
axes[0,0].legend(fontsize=8); axes[0,0].grid(alpha=0.3)

# (0,1) Box plot of all 3 metrics
df_plot = df[["condition", "theta_frontal_log", "alpha_parietal_log", "theta_alpha_ratio"]].copy()
data_rest = [rest["theta_frontal_log"].values, rest["alpha_parietal_log"].values, rest["theta_alpha_ratio"].values]
data_task = [task["theta_frontal_log"].values, task["alpha_parietal_log"].values, task["theta_alpha_ratio"].values]
positions_rest = [1,3,5]
positions_task = [1.7,3.7,5.7]
bp1 = axes[0,1].boxplot(data_rest, positions=positions_rest, widths=0.55,
                         patch_artist=True, boxprops=dict(facecolor=COLORS[0], alpha=0.7))
bp2 = axes[0,1].boxplot(data_task, positions=positions_task, widths=0.55,
                         patch_artist=True, boxprops=dict(facecolor=COLORS[1], alpha=0.7))
axes[0,1].set_xticks([1.35, 3.35, 5.35])
axes[0,1].set_xticklabels(["Frontal θ\n(log)", "Parietal α\n(log)", "θ/α ratio"])
axes[0,1].set_title("EEG Band Powers: Rest vs Mental Arithmetic Task")
from matplotlib.patches import Patch
axes[0,1].legend([Patch(facecolor=COLORS[0], alpha=0.7), Patch(facecolor=COLORS[1], alpha=0.7)],
                  ["Rest", "Mental arithmetic"], fontsize=9)
axes[0,1].grid(axis='y', alpha=0.3)

# (0,2) Simulation vs real: θ/α calibration
cats = ["P10 sim\nbaseline", "P10 sim\nloaded", "Real\nrest", "Real\ntask"]
means = [sim_theta_alpha_rest, 1.2, real_ta_rest, real_ta_task]
cols = [COLORS[2]]*2 + [COLORS[0]]*2
bars = axes[0,2].bar(cats, means, color=cols, alpha=0.85, edgecolor='white')
axes[0,2].set_title("Simulation Assumption vs Real EEG\nθ/α Ratio Calibration")
axes[0,2].set_ylabel("θ/α ratio")
for bar, m in zip(bars, means):
    axes[0,2].text(bar.get_x()+bar.get_width()/2, m+0.01, f"{m:.2f}", ha='center', fontsize=10, fontweight='bold')
axes[0,2].grid(axis='y', alpha=0.3)

# (1,0) PSD of one subject: rest vs task
try:
    raw_rest = mne.io.read_raw_edf(os.path.join(CACHE, "Subject00_1.edf"), preload=True, verbose=False)
    raw_task = mne.io.read_raw_edf(os.path.join(CACHE, "Subject00_2.edf"), preload=True, verbose=False)
    sfreq = raw_rest.info["sfreq"]
    front_ch = min(4, len(raw_rest.ch_names)-1)
    data_r = raw_rest.get_data()[front_ch, :]
    data_t = raw_task.get_data()[front_ch, :]
    f_r, psd_r = signal.welch(data_r, fs=sfreq, nperseg=int(sfreq*2))
    f_t, psd_t = signal.welch(data_t, fs=sfreq, nperseg=int(sfreq*2))
    mask = f_r <= 40
    axes[1,0].semilogy(f_r[mask], psd_r[mask], color=COLORS[0], lw=2, label="Rest")
    axes[1,0].semilogy(f_t[mask], psd_t[mask], color=COLORS[1], lw=2, linestyle='--', label="Task")
    axes[1,0].axvspan(*THETA_BAND, alpha=0.12, color=COLORS[4], label="θ (4-8 Hz)")
    axes[1,0].axvspan(*ALPHA_BAND, alpha=0.12, color=COLORS[2], label="α (8-13 Hz)")
    axes[1,0].set_title("PSD: Subject 00 — Rest vs Mental Arithmetic Task")
    axes[1,0].set_xlabel("Frequency (Hz)"); axes[1,0].set_ylabel("PSD (log)")
    axes[1,0].legend(fontsize=8); axes[1,0].grid(alpha=0.3)
except Exception as e:
    axes[1,0].text(0.5, 0.5, f"PSD unavailable\n{e}", ha='center', va='center', transform=axes[1,0].transAxes)

# (1,1) Effect sizes
ds = [stats_results[m]["cohens_d"] for m in metrics]
colors_d = [COLORS[2] if abs(d) >= 0.8 else COLORS[3] if abs(d) >= 0.5 else COLORS[0] for d in ds]
bars2 = axes[1,1].bar(metric_labels, ds, color=colors_d, alpha=0.85, edgecolor='white')
for bar, d in zip(bars2, ds):
    axes[1,1].text(bar.get_x()+bar.get_width()/2, d + (0.02 if d>=0 else -0.05),
                   f"d={d:.2f}", ha='center', fontsize=9, fontweight='bold')
axes[1,1].axhline(0.5, color='gray', lw=1, linestyle='--', label="Medium (0.5)")
axes[1,1].axhline(0.8, color='gray', lw=1.5, linestyle='-',  label="Large (0.8)")
axes[1,1].axhline(0,   color='black', lw=0.8)
axes[1,1].set_title("Cohen's d: Task vs Rest\n(real EEG effect sizes)")
axes[1,1].set_ylabel("Cohen's d"); axes[1,1].legend(fontsize=8); axes[1,1].grid(axis='y', alpha=0.3)
axes[1,1].tick_params(axis='x', labelsize=8)

# (1,2) Summary text
ta_direction = "↑" if real_ta_task > real_ta_rest else "↓"
p_ta = stats_results["theta_alpha_ratio"]["p"]
summary_text = (
    f"REAL EEG FINDINGS\n"
    f"─────────────────────────────\n"
    f"Subjects:    Subject00-09 (n={len(subjects_matched)})\n"
    f"Rest θ/α:    {real_ta_rest:.3f} ± {rest['theta_alpha_ratio'].std():.3f}\n"
    f"Task θ/α:    {real_ta_task:.3f} ± {task['theta_alpha_ratio'].std():.3f}\n"
    f"Direction:   {ta_direction} during task (expected ↑)\n"
    f"p-value:     {p_ta:.4f}\n\n"
    f"SIMULATION CALIBRATION\n"
    f"─────────────────────────────\n"
    f"P10 assumed rest θ/α = {sim_theta_alpha_rest:.2f}\n"
    f"Real rest θ/α = {real_ta_rest:.3f}\n"
    f"Error: {real_ta_rest - sim_theta_alpha_rest:+.3f}\n\n"
    f"Theta↑ during task: {'YES' if stats_results['theta_frontal_log']['t'] > 0 else 'NO'} "
    f"(p={stats_results['theta_frontal_log']['p']:.4f})\n"
    f"Alpha↓ during task: {'YES' if stats_results['alpha_parietal_log']['t'] < 0 else 'NO'} "
    f"(p={stats_results['alpha_parietal_log']['p']:.4f})"
)
axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.8))
axes[1,2].axis("off"); axes[1,2].set_title("Validation Summary", fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUT}/eeg_real_data_validation.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"    Figure saved: {OUT}/eeg_real_data_validation.png")

# Save results JSON
out_json = {
    "dataset": "PhysioNet eegmat (Zyma et al., 2019)",
    "subjects": SUBJECTS,
    "n_recordings": len(df),
    "conditions": RUNS_LABEL,
    "real_eeg": {
        "rest_ta_mean": round(real_ta_rest, 4),
        "rest_ta_std":  round(rest["theta_alpha_ratio"].std(), 4),
        "task_ta_mean": round(real_ta_task, 4),
        "task_ta_std":  round(task["theta_alpha_ratio"].std(), 4),
        "task_vs_rest_p": round(p_ta, 4),
        "task_vs_rest_d": round(stats_results["theta_alpha_ratio"]["cohens_d"], 4),
    },
    "simulation_calibration": {
        "p10_assumed_rest_ta": sim_theta_alpha_rest,
        "real_rest_ta":        round(real_ta_rest, 4),
        "calibration_error":   round(real_ta_rest - sim_theta_alpha_rest, 4),
    },
    "band_stats": stats_results,
}
with open(f"{OUT}/eeg_real_results.json", "w") as f:
    json.dump(out_json, f, indent=2)

print(f"\n{'='*60}")
print(f"  EEG REAL DATA SUMMARY")
print(f"  Rest θ/α: {real_ta_rest:.3f} | Task θ/α: {real_ta_task:.3f}")
print(f"  Task direction: {ta_direction} (expected ↑ under load)")
print(f"  P-value: {p_ta:.4f}")
print(f"  Sim calibration error: {real_ta_rest - sim_theta_alpha_rest:+.3f}")
print(f"{'='*60}")
