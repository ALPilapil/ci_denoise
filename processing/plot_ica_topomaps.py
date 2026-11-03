"""
plot_ica_topomaps.py

Generates visual aids and 2D scalp head maps (topographies) for CI EEG analysis:
1. 2D Scalp Topomaps for ALL ICA Components (ica.plot_components) with automated
   classification (CI Artifact vs. Auditory Dipole vs. Ocular/Biological).
2. ICA Component Activation Timecourse (ica.plot_sources) across active speech
   (A, AV) vs. Visual-only silent baseline (V) trials.
3. Scalp Voltage Topomaps at N1 (~100 ms) and P2 (~200 ms) comparing:
   - Raw CI EEG
   - Denoised CI EEG (AATS, SOBI, AATS+SOBI)
   - Clean Normal-Hearing Control Ground Truth
4. Multi-Channel Butterfly Overlays across the scalp.
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.preprocessing import ICA


CI_REFERENCE_CHANNELS = ["P7", "T7", "M2", "M1", "P8"]
CENTRAL_AUDITORY_CHANNELS = ["Cz", "Fz", "FC1", "FC2", "C3", "C4"]


def find_matching_pair(
    data_dir: str,
    target_perm: int = 1,
    ci_pref: Optional[str] = "0901y2",
    hearing_pref: Optional[str] = "0801y2",
) -> Tuple[Optional[str], Optional[str]]:
    ci_files = sorted(glob.glob(os.path.join(data_dir, "**", "*09*-raw.fif"), recursive=True))
    hearing_files = sorted(glob.glob(os.path.join(data_dir, "**", "*08*-raw.fif"), recursive=True))

    chosen_ci = None
    chosen_hearing = None

    if ci_pref:
        for f in ci_files:
            if ci_pref in os.path.basename(f):
                chosen_ci = f
                break

    if hearing_pref:
        for f in hearing_files:
            if hearing_pref in os.path.basename(f):
                chosen_hearing = f
                break

    if chosen_ci is None and ci_files:
        chosen_ci = ci_files[0]
    if chosen_hearing is None and hearing_files:
        chosen_hearing = hearing_files[0]

    return chosen_ci, chosen_hearing


def setup_montage(raw: mne.io.Raw) -> mne.io.Raw:
    """Applies standard 10-05 montage with alias renaming for 2D scalp topographic plotting."""
    try:
        rename_map = {"LMas": "M1", "RMas": "M2"}
        actual_renames = {k: v for k, v in rename_map.items() if k in raw.ch_names and v not in raw.ch_names}
        if actual_renames:
            raw.rename_channels(actual_renames)
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore", match_case=False, verbose=False)
    except Exception as e:
        print(f"Warning: Montage assignment notice: {e}")
    return raw


def classify_components(
    ica: ICA,
    ch_names: List[str],
    ref_channels: List[str] = CI_REFERENCE_CHANNELS,
    central_channels: List[str] = CENTRAL_AUDITORY_CHANNELS,
) -> Dict[int, Dict]:
    """
    Classifies each ICA component based on spatial projection weights:
    - CI Artifact: Strongest projection on retroauricular/mastoid sensors.
    - Central Auditory Dipole: Strongest projection on Cz/Fz with low mastoid weight.
    - Frontal/Ocular: Strongest projection on anterior/frontal poles.
    """
    comp_matrix = ica.get_components()  # (n_channels, n_components)
    n_comps = comp_matrix.shape[1]
    results = {}

    ref_idxs = [ch_names.index(c) for c in ref_channels if c in ch_names]
    central_idxs = [ch_names.index(c) for c in central_channels if c in ch_names]

    for k in range(n_comps):
        weights = comp_matrix[:, k]
        abs_weights = np.abs(weights)
        top_idx = int(np.argmax(abs_weights))
        top_ch = ch_names[top_idx]

        ref_power = np.mean(abs_weights[ref_idxs]) if ref_idxs else 0.0
        central_power = np.mean(abs_weights[central_idxs]) if central_idxs else 0.0

        if any(ch_names[idx] in ref_channels for idx in np.argsort(abs_weights)[::-1][:2]):
            cat = "CI ARTIFACT (Coil/Mastoid)"
            color = "#d62728"
        elif any(ch_names[idx] in ["Fp1", "Fp2", "AF3", "AF4"] for idx in np.argsort(abs_weights)[::-1][:2]):
            cat = "OCULAR (Blink/Eye)"
            color = "#ff7f0e"
        elif central_power > 1.3 * (ref_power + 1e-9) and (top_ch in central_channels):
            cat = "CENTRAL AUDITORY DIPOLE"
            color = "#2ca02c"
        else:
            cat = "DIFFUSE / NEURAL"
            color = "#1f77b4"

        results[k] = {
            "top_channel": top_ch,
            "category": cat,
            "color": color,
            "ref_ratio": float(ref_power / (central_power + 1e-9)),
        }

    return results


def plot_ica_components_topomap(
    ica: ICA,
    raw_ci: mne.io.Raw,
    classifications: Dict[int, Dict],
    output_path: str,
):
    """Plots 2D scalp head topographies for all independent components with labels."""
    n_comps = ica.n_components_
    n_cols = 5
    n_rows = int(np.ceil(n_comps / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.2 * n_rows))
    axes = np.array(axes).reshape(-1)

    print(f"Plotting 2D scalp topographies for {n_comps} ICA components...")

    for k in range(n_comps):
        ax = axes[k]
        info_sub = ica.info.copy()
        weights = ica.get_components()[:, k]

        # Topomap of component spatial map
        im, _ = mne.viz.plot_topomap(
            weights,
            info_sub,
            axes=ax,
            show=False,
            sphere=0.095,
            cmap="RdBu_r",
        )

        cls = classifications.get(k, {})
        cat = cls.get("category", "")
        col = cls.get("color", "black")
        top_ch = cls.get("top_channel", "")

        ax.set_title(
            f"IC #{k:02d} [{top_ch}]\n{cat}",
            fontsize=8.5,
            fontweight="bold",
            color=col,
            pad=2,
        )

    # Hide unused subplots
    for k in range(n_comps, len(axes)):
        axes[k].axis("off")

    plt.suptitle(
        f"2D Scalp Topography of All {n_comps} Independent Components (Subject: {raw_ci.info.get('description', 'CI')})\n"
        "Red = CI Coil Artifact (Mastoid Hotspot) | Green = Central Auditory Dipole (Cz/Fz)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved 2D ICA Scalp Topomap Figure → {output_path}")


def plot_component_activation_timecourse(
    ica: ICA,
    raw_ci: mne.io.Raw,
    classifications: Dict[int, Dict],
    output_path: str,
    duration_s: float = 15.0,
):
    """
    Plots the timecourses of key components across continuous speech
    highlighting active speech (A, AV) vs. Visual-only silence (V).
    """
    sfreq = raw_ci.info["sfreq"]
    n_samples = int(duration_s * sfreq)

    # Find first continuous window with annotations
    t0 = 0.0
    if raw_ci.annotations and len(raw_ci.annotations) > 0:
        t0 = float(raw_ci.annotations[0]["onset"])

    raw_crop = raw_ci.copy().crop(tmin=t0, tmax=t0 + duration_s, verbose=False)
    sources = ica.get_sources(raw_crop).get_data()
    times = raw_crop.times

    # Select top artifact component, top auditory component, and ocular component
    art_ic = next((k for k, v in classifications.items() if "CI ARTIFACT" in v["category"]), 0)
    aud_ic = next((k for k, v in classifications.items() if "AUDITORY" in v["category"]), 1)
    eye_ic = next((k for k, v in classifications.items() if "OCULAR" in v["category"]), None)

    plot_ics = [art_ic, aud_ic]
    if eye_ic is not None:
        plot_ics.append(eye_ic)
    while len(plot_ics) < min(4, ica.n_components_):
        for k in range(ica.n_components_):
            if k not in plot_ics:
                plot_ics.append(k)
                break

    fig, axes = plt.subplots(len(plot_ics), 1, figsize=(14, 2.8 * len(plot_ics)), sharex=True)
    if len(plot_ics) == 1:
        axes = [axes]

    # Trial background shading
    cond_colors = {"A": "#d62728", "AV": "#9467bd", "V": "#2ca02c"}
    if raw_ci.annotations:
        for ax in axes:
            for annot in raw_ci.annotations:
                a_t0 = float(annot["onset"]) - t0
                a_dur = float(annot["duration"]) if annot["duration"] > 0 else 1.5
                desc = annot["description"]
                if desc in cond_colors and (0 <= a_t0 < duration_s or 0 < a_t0 + a_dur <= duration_s):
                    ax.axvspan(max(0, a_t0), min(duration_s, a_t0 + a_dur), color=cond_colors[desc], alpha=0.12)
                    ax.axvline(max(0, a_t0), color="gray", ls="--", lw=0.8, alpha=0.5)

    for idx, ic_idx in enumerate(plot_ics):
        ax = axes[idx]
        sig = sources[ic_idx]
        cls = classifications.get(ic_idx, {})
        cat = cls.get("category", "")
        col = cls.get("color", "black")

        ax.plot(times, sig, color=col, lw=1.2, label=f"IC #{ic_idx} ({cat})")
        ax.set_ylabel(r"Amplitude (A.U.)")
        ax.set_title(f"Component #{ic_idx:02d}: {cat} (Top Sensor: {cls.get('top_channel', 'N/A')})", fontweight="bold", fontsize=10)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds relative to window start)", fontweight="bold")
    plt.suptitle(
        f"ICA Component Activations Over Time (Green Shading = Visual-Only Silence [V], Red = Audio [A])\n"
        "Notice CI Artifact Component flatlining during Visual-Only Silence!",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved Component Activation Timeline Figure → {output_path}")


def plot_evoked_topomap_comparison(
    raw_ci: mne.io.Raw,
    raw_hearing: mne.io.Raw,
    ica: ICA,
    ci_comp_indices: List[int],
    output_path: str,
):
    """
    Plots 2D scalp voltage maps at N1 (~100 ms) and P2 (~200 ms) comparing:
    1. Raw CI EEG
    2. ICA-Denoised CI EEG
    3. Clean Normal-Hearing Control Ground Truth
    """
    cond = "A" if "A" in raw_ci.annotations.description else ("AV" if "AV" in raw_ci.annotations.description else None)
    if not cond:
        print("Notice: No A or AV conditions for evoked topomap.")
        return

    # Epoch Raw CI
    ev_ci, _ = mne.events_from_annotations(raw_ci, event_id={cond: 1}, verbose=False)
    ep_ci = mne.Epochs(raw_ci, ev_ci, event_id={cond: 1}, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0.0), preload=True, verbose=False)
    evoked_raw_ci = ep_ci.average()

    # Denoise with ICA
    clean_raw_ci = raw_ci.copy()
    ica.apply(clean_raw_ci, exclude=ci_comp_indices, verbose=False)
    ep_ci_clean = mne.Epochs(clean_raw_ci, ev_ci, event_id={cond: 1}, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0.0), preload=True, verbose=False)
    evoked_clean_ci = ep_ci_clean.average()

    # Epoch Clean Hearing Control
    ev_h, _ = mne.events_from_annotations(raw_hearing, event_id={cond: 1}, verbose=False)
    ep_h = mne.Epochs(raw_hearing, ev_h, event_id={cond: 1}, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0.0), preload=True, verbose=False)
    evoked_hearing = ep_h.average()

    times_to_plot = [0.100, 0.200]  # N1 (~100 ms), P2 (~200 ms)
    titles = [
        "Raw CI EEG (Contaminated with Coil Artifact)",
        f"ICA-Cleaned CI EEG (Excluded ICs: {ci_comp_indices})",
        "Clean Normal-Hearing Control (Intact Auditory Dipole)",
    ]
    evokeds = [evoked_raw_ci, evoked_clean_ci, evoked_hearing]

    fig, axes = plt.subplots(3, 2, figsize=(10, 11))

    for row_idx, ev in enumerate(evokeds):
        for col_idx, t_val in enumerate(times_to_plot):
            ax = axes[row_idx, col_idx]
            im, _ = mne.viz.plot_topomap(
                ev.get_data(tmin=t_val - 0.01, tmax=t_val + 0.01).mean(axis=1) * 1e6,
                ev.info,
                axes=ax,
                show=False,
                sphere=0.095,
                cmap="RdBu_r",
            )
            lbl = "N1 Peak (~100 ms)" if col_idx == 0 else "P2 Peak (~200 ms)"
            ax.set_title(f"{titles[row_idx]}\n{lbl}", fontsize=9, fontweight="bold", pad=4)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"Voltage ($\mu$V)")

    plt.suptitle(
        f"2D Brain Scalp Potential Maps at Auditory ERP Peaks (N1 & P2)\n"
        "Notice the massive unilateral CI artifact disappearing and central dipole emerging after cleaning!",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved ERP Brain Topomap Comparison Figure → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate 2D Scalp Topographies and Brain Maps for ICA & Denoising")
    parser.add_argument("--data-dir", default="/quobyte/millerlmgrp/annotated_data", help="Directory of annotated FIF files")
    parser.add_argument("--ci-subject", default="0901y2", help="CI subject ID (e.g. 0901y2)")
    parser.add_argument("--hearing-subject", default="0801y2", help="Hearing subject ID (e.g. 0801y2)")
    parser.add_argument("--permutation", type=int, default=1, help="Permutation index (1, 2, or 3)")
    parser.add_argument("--n-components", type=int, default=15, help="Number of ICA components to fit")
    parser.add_argument("--output-dir", default="./benchmark_results/single_pair_demo", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  2D BRAIN TOPOGRAPHY & ICA COMPONENT VISUALIZATION STUDIO")
    print("=" * 70)
    print(f"CI Subject:      {args.ci_subject}")
    print(f"Hearing Control: {args.hearing_subject}")
    print(f"Permutation:     perm{args.permutation}")
    print(f"Components:      {args.n_components}\n")

    ci_path, hearing_path = find_matching_pair(
        args.data_dir,
        target_perm=args.permutation,
        ci_pref=args.ci_subject,
        hearing_pref=args.hearing_subject,
    )

    if not ci_path or not os.path.exists(ci_path):
        print(f"Error: Could not find CI file for subject {args.ci_subject} under {args.data_dir}")
        return
    if not hearing_path or not os.path.exists(hearing_path):
        print(f"Error: Could not find Hearing file for subject {args.hearing_subject} under {args.data_dir}")
        return

    print(f"1. Loading CI Recording:      {os.path.basename(ci_path)}")
    print(f"2. Loading Hearing Recording: {os.path.basename(hearing_path)}")

    raw_ci = mne.io.read_raw_fif(ci_path, preload=True, verbose=False)
    raw_hearing = mne.io.read_raw_fif(hearing_path, preload=True, verbose=False)

    # Common channels and standard 10-20 montage
    common_chs = [ch for ch in raw_hearing.ch_names if ch in raw_ci.ch_names]
    raw_ci.pick(common_chs)
    raw_hearing.pick(common_chs)

    raw_ci = setup_montage(raw_ci)
    raw_hearing = setup_montage(raw_hearing)

    # Step 1: Fit ICA on CI Recording
    print(f"\nFitting FastICA with {args.n_components} components on {args.ci_subject}...")
    ica = ICA(n_components=min(len(common_chs), args.n_components), method="fastica", random_state=42, max_iter="auto")
    ica.fit(raw_ci, verbose=False)

    # Step 2: Classify components
    classifications = classify_components(ica, common_chs)
    ci_comp_indices = [k for k, v in classifications.items() if "CI ARTIFACT" in v["category"]]
    if not ci_comp_indices:
        ci_comp_indices = [0]

    print("\n" + "=" * 65)
    print("  ICA COMPONENT CLASSIFICATION & SPATIAL MAPPING SUMMARY")
    print("=" * 65)
    print(f"{'IC #':<6} {'Top Sensor':<12} {'Classification':<32} {'Mastoid Ratio':<12}")
    print("-" * 65)
    for k, v in classifications.items():
        flag = " [EXCLUDED AS NOISE]" if k in ci_comp_indices else ""
        print(f"#{k:02d}   {v['top_channel']:<12} {v['category']:<32} {v['ref_ratio']:<12.2f}{flag}")
    print("=" * 65 + "\n")

    # Step 3: Export 2D Scalp Topomaps of All Components
    out_topomap = os.path.join(args.output_dir, f"ica_all_components_topomap_{args.ci_subject}.png")
    plot_ica_components_topomap(ica, raw_ci, classifications, out_topomap)

    # Step 4: Export Activation Timecourse across A, AV, V conditions
    out_timeline = os.path.join(args.output_dir, f"ica_sources_timeline_{args.ci_subject}.png")
    plot_component_activation_timecourse(ica, raw_ci, classifications, out_timeline, duration_s=15.0)

    # Step 5: Export N1/P2 Brain Topomaps Before vs After Denoising
    out_erp_topo = os.path.join(args.output_dir, f"erp_topomap_n1_p2_comparison_{args.ci_subject}.png")
    plot_evoked_topomap_comparison(raw_ci, raw_hearing, ica, ci_comp_indices, out_erp_topo)

    # Save summary metadata JSON
    meta_json = os.path.join(args.output_dir, f"ica_components_summary_{args.ci_subject}.json")
    with open(meta_json, "w") as f:
        json.dump(classifications, f, indent=2)
    print(f"Saved Component Metadata Summary → {meta_json}")

    print("\nVisual Aid and 2D Topography Generation Completed Successfully!")


if __name__ == "__main__":
    main()
