---
name: visualization-styling
description: Enforces the minimal, publication-ready design philosophy of the cidenoise EEG Waveform Comparison App.
always_on: true
---

# Electrophysiology Visualization & UI Guidelines

All visualizers, interactive HTML dashboards, and plots created for `cidenoise` MUST adhere to the clean, minimal aesthetic established in `visualization/waveform_viewer.html`:

1. **Light, Minimal Theme:**
   - Background: Pure white `#ffffff` with subtle `#fafafa` or `#f5f5f5` panel backgrounds.
   - Dividers & Borders: Thin 1px `#e0e0e0` or `#dcdcdc`.
   - Never use dark gaming themes, heavy drop shadows, or fluorescent neon accents.

2. **Typography & Colors:**
   - Font: Standard system sans-serif (`-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`), 12–13px base size.
   - Text Colors: Primary `#111111`, secondary `#666666`, captions `#777777`.
   - Data Palette:
     * Clean Ground Truth: `#1f77b4` (blue).
     * Raw Contaminated CI: `#777777` (neutral gray).
     * SOBI Denoised: `#2ca02c` (forest green).
     * FastICA Denoised: `#ff7f0e` (orange).
     * CI Coil Artifact: `#ef4444` / `#b91c1c` (red).

3. **Topographies & Waveforms:**
   - Topographies: Render on white canvas with `#111111` 1.5px scalp contour, nose and ear guides, and standard electrode dots.
   - Waveforms: White plot background, 1px `#e0e0e0` zero-lines, and clean unobtrusive legends.
