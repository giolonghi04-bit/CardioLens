#Spo2_ML
# For model training:
#   python spo2_ML.py metadata.csv --output dataset.csv
#   python spo2_ML.py dataset.csv --train --alpha 1.0
# To obtain a prediction from the trained model:
#   python spo2_ML.py new_video.mp4 --predict 


# ===========================================================================
# IMPORTS
# ===========================================================================
import argparse
import warnings
import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import pywt


# ===========================================================================
# MODULE 1 — PREPROCESSING
# ===========================================================================

def _extract_clipped_means(video_path, roi_size=100):
   
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"An error occured in opening: {video_path}")

    raw_colors, raw_times_ms = [], []
    dim1 = dim2 = half = None
    half_roi = roi_size // 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        if dim1 is None:
            dim1, dim2, _ = frame.shape
            half = half_roi

        roi = frame[dim1 // 2 - half: dim1 // 2 + half,
                    dim2 // 2 - half: dim2 // 2 + half, :]

        colors = np.zeros(3)
        for j in range(3): 
            flat = roi[:, :, j].flatten().astype(np.float64)
            lo, hi = np.percentile(flat, 5), np.percentile(flat, 95)
            mask = (flat >= lo) & (flat <= hi)
            valid = flat[mask]
            colors[j] = valid.mean() if valid.size > 0 else flat.mean()

        raw_colors.append(colors)
        raw_times_ms.append(t_ms)

    fps_meta = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    raw_colors = np.array(raw_colors)
    raw_times_ms = np.array(raw_times_ms, dtype=np.float64)

    if len(raw_times_ms) < 20:
        raise ValueError(f"Frame number too low: {video_path}")

    if np.allclose(raw_times_ms, raw_times_ms[0]):
        raw_times_ms = np.arange(len(raw_colors)) * (1000.0 / fps_meta)

    return raw_colors, raw_times_ms, fps_meta


def _wavelet_denoising_vtppg(signal, wavelet_type='db8', level=5):
   
    low_lim = np.percentile(signal, 5)
    high_lim = np.percentile(signal, 95)
    signal_clipped = np.clip(signal, low_lim, high_lim)

    coeffs = pywt.wavedec(signal_clipped, wavelet_type, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    soft_thr_normal = sigma * 2.0
    soft_thr_huge = sigma * 20.0
    if soft_thr_normal == 0.0:
        soft_thr_normal = 1e-8
        soft_thr_huge = 1e-7

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        coeffs[0] = pywt.threshold(coeffs[0], value=soft_thr_huge, mode='soft')
        for i in range(1, 4):
            coeffs[i] = pywt.threshold(coeffs[i], value=soft_thr_normal, mode='soft')
        coeffs[-2] = pywt.threshold(coeffs[-2], value=soft_thr_huge, mode='soft')
        coeffs[-1] = pywt.threshold(coeffs[-1], value=soft_thr_huge, mode='soft')

    denoised = pywt.waverec(coeffs, wavelet_type)
    return denoised[:len(signal)]


def _normalize_m1_p1(signal):
    s_min, s_max = np.min(signal), np.max(signal)
    if s_max - s_min == 0:
        return np.zeros_like(signal)
    return 2.0 * ((signal - s_min) / (s_max - s_min)) - 1.0


def vtppg_preprocess_video(video_path, roi_size=100,
                            trim_start_sec=2.0, trim_end_sec=2.0,
                            target_fps=30.0):

    raw_colors, raw_times_ms, _fps_meta = _extract_clipped_means(video_path, roi_size=roi_size)

    # video cutting
    t0 = raw_times_ms[0] + trim_start_sec * 1000.0
    t1 = raw_times_ms[-1] - trim_end_sec * 1000.0
    keep = (raw_times_ms >= t0) & (raw_times_ms <= t1)
    if keep.sum() < 20:
        raise ValueError(f"Frame number too low: {video_path}")

    times_ms = raw_times_ms[keep]
    colors = raw_colors[keep]

    order = np.argsort(times_ms)
    times_ms, colors = times_ms[order], colors[order]
    _, uniq_idx = np.unique(times_ms, return_index=True)
    times_ms, colors = times_ms[uniq_idx], colors[uniq_idx]

    # zero-mean normalization 
    colors = colors - colors.mean(axis=0, keepdims=True)

    # CubicSpline interpolation
    step_ms = 1000.0 / target_fps
    grid_ms = np.arange(times_ms[0], times_ms[-1], step_ms)
    interp = np.zeros((len(grid_ms), 3))
    for c in range(3):
        cs = CubicSpline(times_ms, colors[:, c])
        interp[:, c] = cs(grid_ms)

    # signal inversion in amplitude + wavelet denoising + normalization in [-1,1]
    clean_signals = np.zeros_like(interp)
    for c in range(3):
        inverted = interp[:, c] * -1.0
        denoised = _wavelet_denoising_vtppg(inverted)
        clean_signals[:, c] = _normalize_m1_p1(denoised)

    return clean_signals, target_fps


# ===========================================================================
# MODULE 2 — HR extraction
# ===========================================================================

def _limit_signal_2sigma(signal):
    # Signal clipping
    thr_hi = np.mean(signal) + 2 * np.std(signal)
    thr_lo = np.mean(signal) - 2 * np.std(signal)
    s = signal.copy()
    s[s > thr_hi] = thr_hi
    s[s < thr_lo] = thr_lo
    return s


def _compute_stats_peaks(i_peaks):
    # Median and 10°-90° percentiles of RR intervals, per channel
    medians, p10s, p90s = np.zeros(3), np.zeros(3), np.zeros(3)
    for i in range(3):
        peaks = i_peaks[i]
        if len(peaks) < 2:
            continue
        intervals = [peaks[j + 1] - peaks[j] for j in range(len(peaks) - 1)]
        medians[i] = np.median(intervals)
        p10s[i] = np.percentile(intervals, 10)
        p90s[i] = np.percentile(intervals, 90)
    return medians, p10s, p90s


def _clean_peaks(peaks, median_i, p10_i, p90_i, alpha_1=1.7, alpha_2=0.6):
   # Elimination of abnoarmal peaks
    keep = [True] * len(peaks)
    intervals = []
    j = 1
    while j < len(peaks) - 1:
        prev_diff = peaks[j] - peaks[j - 1]
        next_diff = peaks[j + 1] - peaks[j]
        cond1 = prev_diff < alpha_2 * median_i or prev_diff > alpha_1 * median_i
        cond2 = (
            (prev_diff < p10_i and next_diff > p90_i) or
            (next_diff < p10_i and prev_diff > p90_i)
        )
        if cond1 or cond2:
            intervals.append((peaks[j - 1], peaks[j + 1]))
            keep[j] = False
            keep[j + 1] = False
            j += 2
        else:
            j += 1
    new_peaks = [p for p, k in zip(peaks, keep) if k]
    deleted = [p for p, k in zip(peaks, keep) if not k]
    return new_peaks, deleted, intervals


def _merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def estimate_hr_hyperbola_cut(signals_3ch, fps):
 
    fr = int(round(fps))
    n = signals_3ch.shape[0]

    # Raw estimation
    fixed_freq = np.zeros(3)
    for i in range(3):
        peaks, _ = find_peaks(signals_3ch[:, i], height=-0.25, distance=0.39 * fr)
        if n > 1 and len(peaks) > 0:
            fixed_freq[i] = len(peaks) / (n - 1) * fr * 60
        else:
            fixed_freq[i] = 70.0  # physiological fallback

    # estimation refining with hyperbolic method
    d02, ad, h2 = 0.037, 25, -0.2
    i_peaks2 = []
    for i in range(3):
        freq_i = fixed_freq[i] if fixed_freq[i] > 0 else 70.0
        d2 = d02 + ad / freq_i
        peaks, _ = find_peaks(signals_3ch[:, i], height=h2, distance=d2 * fr)
        i_peaks2.append(list(peaks))

    # Cutting tecnique
    medians, p10s, p90s = _compute_stats_peaks(i_peaks2)
    new_peaks2 = []
    all_intervals2 = []
    for i in range(3):
        med_i = medians[i] if medians[i] > 0 else fr
        new_p, _, intervals_i = _clean_peaks(
            i_peaks2[i], med_i, p10s[i], p90s[i], alpha_1=1.7, alpha_2=0.6
        )
        intervals_i = _merge_intervals(intervals_i)
        new_peaks2.append(new_p)
        all_intervals2.append(intervals_i if intervals_i else [[0, 0]])

    # Final HR computing and quality estimation for the red channel
    removed = sum(end - start for start, end in all_intervals2[2])
    effective_len = max(n - removed, 1)
    n_peaks = len(new_peaks2[2])

    if effective_len > 1 and n_peaks > 0:
        hr_bpm = n_peaks / (effective_len - 1) * fr * 60
    else:
        hr_bpm = float(fixed_freq[2])

    quality = 10 * (effective_len / n) ** 2

    return float(hr_bpm), float(quality)


# ===========================================================================
# MODULE 3 — Features extraction (Peaks, AC, RMS, mean, std per B/G/R)
# ===========================================================================

def extract_tonmoy_features(clean_signals, fps, window_sec=10, step_sec=5):
    
    n = clean_signals.shape[0]
    win = int(window_sec * fps)
    step = int(step_sec * fps)

    rows = []
    start = 0
    while start + win <= n:
        end = start + win
        win_sig = clean_signals[start:end, :]

        feat = {}
        for ci, cname in enumerate(CHANNEL_NAMES):
            sig_w = win_sig[:, ci]
            feat[f'peak_{cname}'] = float(np.max(sig_w))
            feat[f'ac_{cname}']   = float((np.percentile(sig_w, 95) - np.percentile(sig_w, 5)) / 2.0)
            feat[f'rms_{cname}']  = float(np.sqrt(np.mean(sig_w ** 2)))
            feat[f'mean_{cname}'] = float(np.mean(sig_w))
            feat[f'std_{cname}']  = float(np.std(sig_w))

        hr_input = np.zeros_like(win_sig)
        for c in range(3):
            hr_input[:, c] = _normalize_m1_p1(_limit_signal_2sigma(win_sig[:, c]))

        hr_bpm, quality = estimate_hr_hyperbola_cut(hr_input, fps)
        feat['hr_bpm'] = hr_bpm
        feat['signal_quality'] = quality
        feat['window_start_sec'] = start / fps

        rows.append(feat)
        start += step

    return pd.DataFrame(rows)


def build_feature_table_for_video(video_path, roi_size=100,
                                   trim_start_sec=2.0, trim_end_sec=2.0,
                                   target_fps=30.0, window_sec=10, step_sec=5):

    clean_signals, fps = vtppg_preprocess_video(
        video_path, roi_size=roi_size,
        trim_start_sec=trim_start_sec, trim_end_sec=trim_end_sec,
        target_fps=target_fps,
    )
    feats = extract_tonmoy_features(clean_signals, fps, window_sec, step_sec)
    feats['video_path'] = str(video_path)
    return feats


# ===========================================================================
# MODULE 4 — Device normalization
# ===========================================================================

def compute_device_stats(df, feature_cols=FEATURE_COLS, device_col='device_id'):
    stats = {}
    for device, group in df.groupby(device_col):
        stats[device] = {}
        for col in feature_cols:
            mu = group[col].mean()
            sd = group[col].std()
            stats[device][col] = (mu, sd if sd > 1e-8 else 1.0)
    stats['__global__'] = {}
    for col in feature_cols:
        mu = df[col].mean()
        sd = df[col].std()
        stats['__global__'][col] = (mu, sd if sd > 1e-8 else 1.0)
    return stats


def apply_device_stats(df, stats, feature_cols=FEATURE_COLS, device_col='device_id'):
    df = df.copy()
    devices = df[device_col].values if device_col in df.columns else ['__global__'] * len(df)
    for col in feature_cols:
        new_vals = np.zeros(len(df))
        col_vals = df[col].values.astype(float)
        for i, dev in enumerate(devices):
            dev_key = dev if dev in stats else '__global__'
            mu, sd = stats[dev_key][col]
            new_vals[i] = (col_vals[i] - mu) / sd
        df[col] = new_vals
    return df


# ===========================================================================
# MODULE 5 — RIDGE REGRESSION (sensitivity: 1%)
# ===========================================================================

def build_ridge_model(alpha=1.0):

    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('model', Ridge(alpha=alpha)),
    ])


def _round_to_sensitivity(values, sensitivity=1.0):
    
    return np.round(np.asarray(values) / sensitivity) * sensitivity


# ===========================================================================
# MODULE 6 — LOSO validation + Metrics + Bland-Altman
# ===========================================================================

def bland_altman_stats(ref, pred):
    diff = np.asarray(pred) - np.asarray(ref)
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    return bias, sd, bias - 1.96 * sd, bias + 1.96 * sd


def evaluate_loso(dataset, feature_cols=FEATURE_COLS,
                   normalize_device=True, alpha=1.0, sensitivity=1.0):

    df = dataset.dropna(subset=feature_cols + ['spo2_ref']).copy()

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df[col].abs() > 1e6, col] = np.nan
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    n_subjects = df['subject_id'].nunique()
    if n_subjects < 3:
        raise ValueError(f"A minimun of 3 subjects is required for LOSO, only: {n_subjects} were found")

    logo = LeaveOneGroupOut()
    groups = df['subject_id'].values
    window_preds = np.full(len(df), np.nan)

    for train_idx, test_idx in logo.split(df, df['spo2_ref'].values, groups):
        train_df = df.iloc[train_idx].copy()
        test_df  = df.iloc[test_idx].copy()

        if normalize_device:
            stats = compute_device_stats(train_df, feature_cols)
            train_df = apply_device_stats(train_df, stats, feature_cols)
            test_df  = apply_device_stats(test_df, stats, feature_cols)

        model = build_ridge_model(alpha=alpha)
        model.fit(train_df[feature_cols].values, train_df['spo2_ref'].values)
        window_preds[test_idx] = model.predict(test_df[feature_cols].values)

    df['spo2_pred'] = _round_to_sensitivity(window_preds, sensitivity)

    video_level = df.groupby('video_path').agg(
        subject_id     = ('subject_id', 'first'),
        device_id      = ('device_id', 'first'),
        spo2_ref       = ('spo2_ref', 'first'),
        spo2_pred      = ('spo2_pred', 'mean'),
        signal_quality = ('signal_quality', 'mean'),
        n_finestre     = ('spo2_pred', 'size'),
    ).reset_index()
    video_level['spo2_pred'] = _round_to_sensitivity(video_level['spo2_pred'].values, sensitivity)

    ref  = video_level['spo2_ref'].values.astype(float)
    pred = video_level['spo2_pred'].values.astype(float)

    mae  = float(np.mean(np.abs(pred - ref)))
    mse  = float(np.mean((pred - ref) ** 2))
    rmse = float(np.sqrt(mse))
    bias, sd, loa_low, loa_high = bland_altman_stats(ref, pred)

    print("\n=== LOSO results (Ridge Regression) ===")
    print(f"Evaluated videos : {len(video_level)}  |  Subjects: {n_subjects}")
    print(f"MAE            : {mae:.3f} SpO2 points")
    print(f"MSE            : {mse:.3f}")
    print(f"RMSE           : {rmse:.3f} SpO2 points")
    print(f"Bias           : {bias:+.3f}")
    print(f"LoA            : [{loa_low:+.3f}, {loa_high:+.3f}]  (Bland-Altman 95%)")
    print("=" * 60)

    metrics = dict(mae=mae, mse=mse, rmse=rmse,
                   bias=bias, loa_low=loa_low, loa_high=loa_high)
    return video_level, metrics


def plot_bland_altman(video_level, out_path='bland_altman_tonmoy.png'):
    ref  = video_level['spo2_ref'].values.astype(float)
    pred = video_level['spo2_pred'].values.astype(float)
    mean_v = (ref + pred) / 2
    diff   = pred - ref
    bias, sd, loa_low, loa_high = bland_altman_stats(ref, pred)

    plt.figure(figsize=(7, 5))
    plt.scatter(mean_v, diff, alpha=0.75, zorder=3)
    plt.axhline(bias,     color='black', linestyle='-',  label=f'Bias {bias:+.2f}')
    plt.axhline(loa_high, color='red',   linestyle='--', label='Limits of agreement 95%')
    plt.axhline(loa_low,  color='red',   linestyle='--')
    plt.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    plt.xlabel('Average (reference, predicted) [%SpO2]')
    plt.ylabel('Difference (predicted - reference) [%SpO2]')
    plt.title('Bland-Altman -- Ridge Regression SpO2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Bland-Altman plot saved in: {out_path}")


# ===========================================================================
# MODULE 7 — Final Model
# ===========================================================================

def train_final_model(dataset, feature_cols=FEATURE_COLS,
                       normalize_device=True, alpha=1.0,
                       out_path='spo2_model_tonmoy.joblib'):
    df = dataset.dropna(subset=feature_cols + ['spo2_ref']).copy()
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df[col].abs() > 1e6, col] = np.nan
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    device_stats = None
    if normalize_device:
        device_stats = compute_device_stats(df, feature_cols)
        df = apply_device_stats(df, device_stats, feature_cols)

    model = build_ridge_model(alpha=alpha)
    model.fit(df[feature_cols].values, df['spo2_ref'].values)

    joblib.dump({
        'model'            : model,
        'feature_cols'     : feature_cols,
        'normalize_device' : normalize_device,
        'device_stats'     : device_stats,
        'alpha'            : alpha,
    }, out_path)
    print(f"Final model saved in: {out_path}  ({len(df)} finestre, {df['subject_id'].nunique()} soggetti)")
    return model


# ===========================================================================
# MODULE 8 — BUILD DATASET
# ===========================================================================

REQUIRED_META_COLS = {'video_path', 'subject_id', 'device_id', 'spo2_ref'}


def build_dataset(metadata_csv, output_csv='dataset_tonmoy.csv',
                   roi_size=100, trim_start_sec=2.0, trim_end_sec=2.0,
                   window_sec=10, step_sec=5):
    meta = pd.read_csv(metadata_csv)
    missing = REQUIRED_META_COLS - set(meta.columns)
    if missing:
        raise ValueError(f"Missing data in the CSV metadata file: {missing}")

    all_feats = []
    n_ok, n_skip = 0, 0

    for _, row in meta.iterrows():
        try:
            feats = build_feature_table_for_video(
                row['video_path'], roi_size=roi_size,
                trim_start_sec=trim_start_sec, trim_end_sec=trim_end_sec,
                window_sec=window_sec, step_sec=step_sec,
            )
        except Exception as e:
            print(f"[SKIP] {row['video_path']}: {e}")
            n_skip += 1
            continue

        if feats.empty:
            print(f"[SKIP] {row['video_path']}: no extracted window")
            n_skip += 1
            continue

        feats['subject_id'] = row['subject_id']
        feats['device_id']  = row['device_id']
        feats['spo2_ref']   = float(row['spo2_ref'])
        all_feats.append(feats)
        n_ok += 1
        print(f"[OK]   {row['video_path']}: {len(feats)} windows")

    if not all_feats:
        raise RuntimeError("No video was successfully processed")

    dataset = pd.concat(all_feats, ignore_index=True)
    dataset.to_csv(output_csv, index=False)
    print(f"\nDataset salvato: {output_csv}")
    print(f"Video OK: {n_ok} | saltati: {n_skip}")
    print(f"Finestre totali: {len(dataset)} | soggetti: {dataset['subject_id'].nunique()}")
    return dataset


# ===========================================================================
# MODULE 9 — Prediction on a new video
# ===========================================================================

def predict_spo2(video_path, model_path='spo2_model_tonmoy.joblib',
                  device_id=None, min_quality=0.0, sensitivity=1.0):
    bundle = joblib.load(model_path)
    model          = bundle['model']
    feature_cols   = bundle['feature_cols']
    normalize_dev  = bundle['normalize_device']
    device_stats   = bundle['device_stats']

    feats = build_feature_table_for_video(video_path)

    feats = feats.replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        feats.loc[feats[col].abs() > 1e6, col] = np.nan
    feats = feats.dropna(subset=feature_cols)

    if device_id is not None:
        feats['device_id'] = device_id

    n_tot = len(feats)
    feats_ok = feats[feats['signal_quality'] >= min_quality].copy()
    if feats_ok.empty:
        feats_ok = feats.copy()
        print("[WARN] No window succedeed the quality test, all windows will be used")

    if normalize_dev and device_stats is not None:
        if device_id is not None and device_id not in device_stats:
            print(f"[WARN] device_id '{device_id}' not seen in training phase, global normalization will be used")
        feats_ok = apply_device_stats(feats_ok, device_stats, feature_cols)

    X = feats_ok[feature_cols].values
    preds = _round_to_sensitivity(model.predict(X), sensitivity)

    result = {
        'spo2_pred'         : round(float(np.mean(preds)), 1),
        'std_tra_finestre'  : round(float(np.std(preds)), 2) if len(preds) > 1 else 0.0,
        'n_finestre_totali' : n_tot,
        'n_finestre_usate'  : len(preds),
        'qualita_media'     : round(float(feats_ok['signal_quality'].mean()), 2),
    }
    return result


# ===========================================================================
# MAIN CLI
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Pipeline SpO2 -- preprocessing VTPPG + HR iperbole/taglio + Ridge Regression"
    )
    parser.add_argument('input',
        help="metadata.csv (build) | dataset.csv (train) | video.mp4 (predict)")
    parser.add_argument('--output',     default='dataset_tonmoy.csv')
    parser.add_argument('--train',      action='store_true',
        help="Modalita' addestramento+valutazione LOSO")
    parser.add_argument('--predict',    action='store_true',
        help="Modalita' predizione su nuovo video")
    parser.add_argument('--model',      default='spo2_model_tonmoy.joblib')
    parser.add_argument('--alpha',      type=float, default=1.0,
        help="Parametro di regolarizzazione Ridge (default=1.0)")
    parser.add_argument('--sensitivity', type=float, default=1.0,
        help="Sensibilita' di arrotondamento output SpO2 in punti %% (default=1.0)")
    parser.add_argument('--no-device-norm', action='store_true')
    parser.add_argument('--device-id',  default=None,
        help="es. honor_x8b (solo per --predict)")
    parser.add_argument('--roi-size',   type=int, default=100,
        help="lato della ROI quadrata centrale in pixel (default=100, come VTPPG)")
    parser.add_argument('--trim-start-sec', type=float, default=2.0)
    parser.add_argument('--trim-end-sec',   type=float, default=2.0)
    parser.add_argument('--window-sec', type=float, default=10.0)
    parser.add_argument('--step-sec',   type=float, default=5.0)
    args = parser.parse_args()

    if args.predict:
        import json
        res = predict_spo2(args.input, model_path=args.model,
                            device_id=args.device_id, sensitivity=args.sensitivity)
        print(json.dumps(res, indent=2, ensure_ascii=False))

    elif args.train:
        dataset = pd.read_csv(args.input)
        normalize = not args.no_device_norm
        video_level, metrics = evaluate_loso(
            dataset, normalize_device=normalize, alpha=args.alpha,
            sensitivity=args.sensitivity,
        )
        plot_bland_altman(video_level)
        video_level.to_csv('loso_predictions_tonmoy.csv', index=False)
        train_final_model(
            dataset, normalize_device=normalize,
            alpha=args.alpha, out_path=args.model
        )

    else:
        build_dataset(
            args.input, output_csv=args.output,
            roi_size=args.roi_size,
            trim_start_sec=args.trim_start_sec, trim_end_sec=args.trim_end_sec,
            window_sec=args.window_sec, step_sec=args.step_sec,
        )
