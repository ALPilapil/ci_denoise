import os
import json
import numpy as np

def build():
    npz_path = "/Users/evanywliang/Documents/cidenoise/ci_denoise/benchmark_results/single_pair_demo/paired_epochs_perm1_all.npz"
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found")
        return

    data = np.load(npz_path)
    step = 8 # 16385 // 8 = 2049 points (-200ms to +800ms)
    times = (data['times'][::step] * 1000).round(2).tolist()
    channels = list(data['channels'])

    series_data = {}
    for ch_idx, ch in enumerate(channels):
        series_data[ch] = {
            'clean': (data['clean'][ch_idx, ::step] * 1e6).round(3).tolist(),
            'noise_aats': (data['noise_aats'][ch_idx, ::step] * 1e6).round(3).tolist(),
            'noise_ica': (data['noise_ica'][ch_idx, ::step] * 1e6).round(3).tolist(),
            'noise_sobi': (data['noise_sobi'][ch_idx, ::step] * 1e6).round(3).tolist(),
            'noise_aats_sobi': (data['noise_aats_sobi'][ch_idx, ::step] * 1e6).round(3).tolist() if 'noise_aats_sobi' in data else [],
            'noise_aats_ica': (data['noise_aats_ica'][ch_idx, ::step] * 1e6).round(3).tolist() if 'noise_aats_ica' in data else [],
            'dirty_aats': (data['dirty_aats'][ch_idx, ::step] * 1e6).round(3).tolist(),
            'dirty_ica': (data['dirty_ica'][ch_idx, ::step] * 1e6).round(3).tolist(),
            'dirty_sobi': (data['dirty_sobi'][ch_idx, ::step] * 1e6).round(3).tolist(),
            'dirty_aats_sobi': (data['dirty_aats_sobi'][ch_idx, ::step] * 1e6).round(3).tolist() if 'dirty_aats_sobi' in data else [],
            'dirty_aats_ica': (data['dirty_aats_ica'][ch_idx, ::step] * 1e6).round(3).tolist() if 'dirty_aats_ica' in data else [],
        }

    dataset_json = json.dumps({
        'ci_subject': str(data['ci_subject']),
        'hearing_subject': str(data['hearing_subject']),
        'permutation': int(data['permutation']),
        'channels': channels,
        'times': times,
        'series': series_data
    })

    # Load Multi-Trial Block Data
    multitrial_npz = "/Users/evanywliang/Documents/cidenoise/ci_denoise/benchmark_results/single_pair_demo/multitrial_block_data_perm1.npz"
    multitrial_json = "{}"
    if os.path.exists(multitrial_npz):
        m_data = np.load(multitrial_npz)
        step_m = 48  # 196608 // 48 = 4096 points over 12s -> 341.3 Hz sampling
        times_m = m_data['time_s'][::step_m].round(3).tolist()
        multitrial_dict = {
            'time_s': times_m,
            'conditions': [str(c) for c in m_data['conditions']],
            'target_channel': str(m_data['target_channel']),
            'raw_ci': m_data['raw_ci'][::step_m].round(2).tolist(),
            'clean_hearing': m_data['clean_hearing'][::step_m].round(2).tolist(),
            'noise_aats': m_data['noise_aats'][::step_m].round(2).tolist(),
            'denoised_aats': m_data['denoised_aats'][::step_m].round(2).tolist(),
            'noise_ica': m_data['noise_ica'][::step_m].round(2).tolist(),
            'denoised_ica': m_data['denoised_ica'][::step_m].round(2).tolist(),
            'noise_sobi': m_data['noise_sobi'][::step_m].round(2).tolist(),
            'denoised_sobi': m_data['denoised_sobi'][::step_m].round(2).tolist(),
            'noise_aats_sobi': m_data['noise_aats_sobi'][::step_m].round(2).tolist() if 'noise_aats_sobi' in m_data else [],
            'denoised_aats_sobi': m_data['denoised_aats_sobi'][::step_m].round(2).tolist() if 'denoised_aats_sobi' in m_data else [],
            'noise_aats_ica': m_data['noise_aats_ica'][::step_m].round(2).tolist() if 'noise_aats_ica' in m_data else [],
            'denoised_aats_ica': m_data['denoised_aats_ica'][::step_m].round(2).tolist() if 'denoised_aats_ica' in m_data else [],
        }
        multitrial_json = json.dumps(multitrial_dict)

        # Export Multi-Trial Cz CSV
        m_csv_path = "/Users/evanywliang/Documents/cidenoise/ci_denoise/benchmark_results/single_pair_demo/multitrial_block_cz.csv"
        with open(m_csv_path, 'w') as f:
            f.write("time_s,trial_idx,condition,raw_ci_uV,clean_hearing_uV,noise_aats_uV,denoised_aats_uV,noise_ica_uV,denoised_ica_uV,noise_sobi_uV,denoised_sobi_uV,noise_aats_sobi_uV,denoised_aats_sobi_uV,noise_aats_ica_uV,denoised_aats_ica_uV\n")
            for i, t in enumerate(times_m):
                t_idx = min(7, int(t // 1.5))
                c_lbl = m_data['conditions'][t_idx]
                nas = multitrial_dict['noise_aats_sobi'][i] if i < len(multitrial_dict['noise_aats_sobi']) else 0
                das = multitrial_dict['denoised_aats_sobi'][i] if i < len(multitrial_dict['denoised_aats_sobi']) else 0
                nai = multitrial_dict['noise_aats_ica'][i] if i < len(multitrial_dict['noise_aats_ica']) else 0
                dai = multitrial_dict['denoised_aats_ica'][i] if i < len(multitrial_dict['denoised_aats_ica']) else 0
                f.write(f"{t},{t_idx+1},{c_lbl},{multitrial_dict['raw_ci'][i]},{multitrial_dict['clean_hearing'][i]},{multitrial_dict['noise_aats'][i]},{multitrial_dict['denoised_aats'][i]},{multitrial_dict['noise_ica'][i]},{multitrial_dict['denoised_ica'][i]},{multitrial_dict['noise_sobi'][i]},{multitrial_dict['denoised_sobi'][i]},{nas},{das},{nai},{dai}\n")
        print(f"Exported Multi-Trial Cz CSV to {m_csv_path}")

    # Export Cz CSV for single-pair evoked
    cz_idx = channels.index('Cz')
    csv_path = "/Users/evanywliang/Documents/cidenoise/ci_denoise/benchmark_results/single_pair_demo/single_pair_cz_epochs.csv"
    with open(csv_path, 'w') as f:
        f.write("time_ms,clean_uV,noise_aats_uV,noise_ica_uV,noise_sobi_uV,noise_aats_sobi_uV,noise_aats_ica_uV,dirty_aats_uV,dirty_ica_uV,dirty_sobi_uV,dirty_aats_sobi_uV,dirty_aats_ica_uV,visual_only_segment\n")
        cz = series_data['Cz']
        for i in range(len(times)):
            v_flag = 1 if times[i] >= 0 else 0
            nas = cz['noise_aats_sobi'][i] if i < len(cz['noise_aats_sobi']) else 0
            nai = cz['noise_aats_ica'][i] if i < len(cz['noise_aats_ica']) else 0
            das = cz['dirty_aats_sobi'][i] if i < len(cz['dirty_aats_sobi']) else 0
            dai = cz['dirty_aats_ica'][i] if i < len(cz['dirty_aats_ica']) else 0
            f.write(f"{times[i]},{cz['clean'][i]},{cz['noise_aats'][i]},{cz['noise_ica'][i]},{cz['noise_sobi'][i]},{nas},{nai},{cz['dirty_aats'][i]},{cz['dirty_ica'][i]},{cz['dirty_sobi'][i]},{das},{dai},{v_flag}\n")
    print(f"Exported Cz sample CSV to {csv_path}")

    # Export Visual-Only Interval CSV
    v_csv_path = "/Users/evanywliang/Documents/cidenoise/ci_denoise/benchmark_results/single_pair_demo/visual_only_intervals_0901y2.csv"
    with open(v_csv_path, 'w') as f:
        f.write("trial_idx,condition,onset_s,end_s,duration_s,description\n")
        f.write("1,V,0.000,1.500,1.500,Visual-Only Silent Baseline (No CI Electrical Stimulation)\n")
    print(f"Exported Visual-Only Silent Interval table to {v_csv_path}")

    # Build HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CI Noise Isolation &amp; Pairing Studio (Single Pair Demo)</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: #ffffff;
      color: #111111;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      font-size: 13px;
    }}

    /* Header */
    .app-header {{
      background-color: #ffffff;
      border-bottom: 1px solid #e0e0e0;
      padding: 0.55rem 1rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }}

    .header-left {{
      display: flex;
      align-items: center;
      gap: 1.25rem;
    }}

    .brand-title {{
      font-size: 0.95rem;
      font-weight: 600;
      color: #111111;
      letter-spacing: -0.01em;
    }}

    .sub-title {{
      font-size: 0.78rem;
      color: #666666;
    }}

    /* Nav Tabs */
    .nav-tabs {{
      display: flex;
      gap: 0.4rem;
    }}

    .nav-tab-btn {{
      padding: 0.35rem 0.75rem;
      font-size: 0.8rem;
      font-weight: 500;
      border: 1px solid #dcdcdc;
      background: #ffffff;
      color: #555555;
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.1s ease;
    }}

    .nav-tab-btn:hover {{
      color: #111111;
      border-color: #999999;
    }}

    .nav-tab-btn.active {{
      background: #111111;
      color: #ffffff;
      border-color: #111111;
      font-weight: 600;
    }}

    .header-actions {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .select-input {{
      font-size: 0.8rem;
      padding: 0.3rem 0.5rem;
      border: 1px solid #dcdcdc;
      border-radius: 4px;
      background: #ffffff;
      color: #111111;
      outline: none;
      cursor: pointer;
    }}

    .select-input:focus {{
      border-color: #111111;
    }}

    .btn {{
      padding: 0.35rem 0.75rem;
      font-size: 0.8rem;
      font-weight: 500;
      border: 1px solid #dcdcdc;
      background: #ffffff;
      color: #333333;
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.1s ease;
    }}

    .btn:hover {{
      color: #111111;
      border-color: #999999;
      background: #fafafa;
    }}

    .btn-primary {{
      background: #111111;
      color: #ffffff;
      border-color: #111111;
      font-weight: 600;
    }}

    .btn-primary:hover {{
      background: #2a2a2a;
      border-color: #2a2a2a;
      color: #ffffff;
    }}

    /* Layout */
    .app-container {{
      display: flex;
      flex: 1;
      height: calc(100vh - 49px);
      overflow: hidden;
      background: #ffffff;
    }}

    /* Sidebar */
    .sidebar {{
      width: 320px;
      min-width: 300px;
      background-color: #ffffff;
      border-right: 1px solid #e0e0e0;
      display: flex;
      flex-direction: column;
      overflow-y: auto;
    }}

    .sidebar-section {{
      padding: 0.85rem;
      border-bottom: 1px solid #e0e0e0;
    }}

    .section-title {{
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #666666;
      font-weight: 600;
      margin-bottom: 0.6rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    .badge {{
      font-size: 0.7rem;
      font-weight: 500;
      padding: 0.1rem 0.35rem;
      border: 1px solid #e0e0e0;
      border-radius: 3px;
      background: #ffffff;
      color: #444444;
    }}

    /* Checklist */
    .checklist {{
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
    }}

    .check-item {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.35rem 0.5rem;
      border-radius: 4px;
      border: 1px solid #f0f0f0;
      background: #fafafa;
      cursor: pointer;
      user-select: none;
      transition: background 0.1s ease;
    }}

    .check-item:hover {{
      background: #f4f4f4;
      border-color: #e5e5e5;
    }}

    .check-label-group {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .check-item input[type="checkbox"], .check-item input[type="radio"] {{
      cursor: pointer;
      accent-color: #111111;
    }}

    .color-swatch {{
      width: 10px;
      height: 10px;
      border-radius: 2px;
      display: inline-block;
    }}

    .swatch-dashed {{
      border-bottom: 2px dashed #333333;
      height: 0px;
      width: 12px;
    }}

    .swatch-dotted {{
      border-bottom: 2px dotted #7f7f7f;
      height: 0px;
      width: 12px;
    }}

    .check-name {{
      font-size: 0.82rem;
      font-weight: 500;
      color: #222222;
    }}

    .check-tag {{
      font-size: 0.7rem;
      color: #777777;
    }}

    /* Preset Filters & Segmented Controls */
    .preset-container {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.35rem;
      margin-top: 0.6rem;
    }}

    .btn-preset {{
      padding: 0.28rem 0.45rem;
      font-size: 0.75rem;
      background: #ffffff;
      border: 1px solid #e0e0e0;
      border-radius: 3px;
      color: #444444;
      cursor: pointer;
      text-align: center;
      transition: all 0.1s ease;
    }}

    .btn-preset:hover {{
      border-color: #111111;
      color: #111111;
    }}

    .segmented-control {{
      display: flex;
      border: 1px solid #dcdcdc;
      border-radius: 4px;
      overflow: hidden;
      margin-bottom: 0.6rem;
    }}

    .seg-btn {{
      flex: 1;
      padding: 0.3rem 0.4rem;
      font-size: 0.75rem;
      background: #ffffff;
      border: none;
      color: #555555;
      cursor: pointer;
      text-align: center;
      border-right: 1px solid #dcdcdc;
      transition: all 0.1s ease;
    }}

    .seg-btn:last-child {{
      border-right: none;
    }}

    .seg-btn:hover {{
      color: #111111;
      background: #fafafa;
    }}

    .seg-btn.active {{
      background: #111111;
      color: #ffffff;
      font-weight: 600;
    }}

    /* Metrics Grid */
    .metrics-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.45rem;
    }}

    .metric-card {{
      border: 1px solid #e8e8e8;
      border-radius: 4px;
      padding: 0.5rem;
      background: #fafafa;
    }}

    .metric-label {{
      font-size: 0.7rem;
      color: #666666;
      margin-bottom: 0.15rem;
    }}

    .metric-value {{
      font-size: 0.88rem;
      font-weight: 600;
      color: #111111;
      font-variant-numeric: tabular-nums;
    }}

    .metric-sub {{
      font-size: 0.68rem;
      color: #888888;
      margin-top: 0.1rem;
    }}

    /* Dropzone */
    .dropzone {{
      border: 1px dashed #b0b0b0;
      border-radius: 4px;
      padding: 0.75rem;
      text-align: center;
      background: #fafafa;
      cursor: pointer;
      transition: all 0.15s ease;
    }}

    .dropzone:hover, .dropzone.dragover {{
      border-color: #111111;
      background: #f0f0f0;
    }}

    .dropzone-text {{
      font-size: 0.78rem;
      color: #222222;
      margin-bottom: 0.15rem;
    }}

    .dropzone-hint {{
      font-size: 0.68rem;
      color: #777777;
    }}

    /* Main Chart Stage */
    .stage-container {{
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      padding: 0.85rem 1.25rem;
      background: #ffffff;
      gap: 0.75rem;
    }}

    .stage-grid-2x2 {{
      display: grid !important;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: auto auto;
      gap: 0.75rem;
    }}

    /* Divider between Ground Truth and Methods */
    .section-divider {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin: 0.35rem 0 0.15rem 0;
      user-select: none;
    }}

    .divider-line {{
      flex: 1;
      height: 1px;
      background: #e0e0e0;
    }}

    .divider-text, .divider-pill {{
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: #666666;
      background: transparent;
      border: none;
      padding: 0 0.5rem;
      white-space: nowrap;
    }}

    .stage-grid-2x2 .section-divider {{
      grid-column: 1 / -1;
    }}

    .stage-grid-2x2 #cardC_clean {{
      grid-column: 1 / -1;
    }}

    .stage-grid-2x2 #cardC_clean .canvas-wrapper {{
      height: 170px;
    }}

    .panel-card {{
      border: 1px solid #e0e0e0;
      border-radius: 4px;
      background: #ffffff;
      display: flex;
      flex-direction: column;
      position: relative;
    }}

    .panel-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.45rem 0.75rem;
      border-bottom: 1px solid #f0f0f0;
      background: #fdfdfd;
    }}

    .panel-title {{
      font-size: 0.82rem;
      font-weight: 600;
      color: #111111;
    }}

    .panel-legend {{
      display: flex;
      align-items: center;
      gap: 0.8rem;
      font-size: 0.74rem;
      color: #444444;
      flex-wrap: wrap;
    }}

    .legend-item {{
      display: flex;
      align-items: center;
      gap: 0.35rem;
    }}

    .canvas-wrapper {{
      position: relative;
      width: 100%;
      height: 155px;
    }}

    .stage-grid-2x2 .canvas-wrapper {{
      height: 235px;
    }}

    canvas {{
      display: block;
      width: 100%;
      height: 100%;
    }}

    /* Tooltip & Crosshair */
    .hover-indicator {{
      position: absolute;
      top: 0;
      bottom: 0;
      width: 1px;
      background: rgba(0, 0, 0, 0.4);
      pointer-events: none;
      display: none;
      z-index: 10;
    }}

    .tooltip-card {{
      position: absolute;
      display: none;
      background: #ffffff;
      border: 1px solid #111111;
      border-radius: 4px;
      padding: 0.4rem 0.6rem;
      font-size: 0.72rem;
      color: #111111;
      pointer-events: none;
      z-index: 20;
      box-shadow: 0 2px 6px rgba(0,0,0,0.08);
      white-space: nowrap;
    }}

    .hidden {{ display: none !important; }}
  </style>
</head>
<body>

  <header class="app-header">
    <div class="header-left">
      <div>
        <div class="brand-title">CI Noise Isolation &amp; Pairing Studio</div>
        <div class="sub-title" id="datasetInfo">Subject Pair: 0901y2 (CI) &times; 0801y2 (Hearing) &bull; Permutation 1 | Cadence: [non-AV (1.5s), AV (1.5s)] &bull; t_AV = Code 25, t_non-AV = Code 25 &minus; 1.5s</div>
      </div>
      <div class="nav-tabs">
        <button class="nav-tab-btn active" id="tabBtnDemo" data-tab="demo">4-Panel Pairing Demo</button>
        <button class="nav-tab-btn" id="tabBtnCompare" data-tab="compare">Method Shape Comparison</button>
        <button class="nav-tab-btn" id="tabBtnMulti" data-tab="multi">Multi-Trial Block</button>
      </div>
    </div>
    <div class="header-actions">
      <label for="chSelect" id="chSelectLabel" style="font-size: 0.78rem; font-weight: 500; color: #444;">Channel:</label>
      <select id="chSelect" class="select-input">
        <!-- Filled by JS -->
      </select>

      <label for="zoomSelect" id="zoomSelectLabel" style="font-size: 0.78rem; font-weight: 500; color: #444; margin-left: 0.4rem;">Time Window:</label>
      <select id="zoomSelect" class="select-input">
        <option value="full">Full Epoch (-200 to +800 ms)</option>
        <option value="stim">Stimulus Window (0 to +600 ms)</option>
        <option value="n1p2">N1-P2 Dipole (0 to +300 ms)</option>
        <option value="baseline">Pre-Stim Baseline (-200 to 0 ms)</option>
      </select>

      <!-- Multi-Trial Specific Window Selector -->
      <select id="zoomSelectMulti" class="select-input hidden">
        <option value="full_block">Full Block (0.0 to 12.0s, All 8 Trials)</option>
        <option value="half_1">First Half: Trials 1–4 (0.0 to 6.0s)</option>
        <option value="half_2">Second Half: Trials 5–8 (6.0 to 12.0s)</option>
        <option value="v_trial3">Silent Window: Trial 3 [V] (3.0 to 4.5s)</option>
        <option value="v_trial7">Silent Window: Trial 7 [V] (9.0 to 10.5s)</option>
        <option value="transition_1">Audio to Silence Transition (1.5 to 4.5s)</option>
      </select>

      <button id="btnExportCSV" class="btn" title="Download data as CSV">Download CSV</button>
      <button id="btnExportPNG" class="btn btn-primary" title="Export current view to PNG">Export Plot</button>
    </div>
  </header>

  <div class="app-container">
    <aside class="sidebar">

      <!-- ==================== SIDEBAR TAB 1: 4-PANEL DEMO ==================== -->
      <div id="sidebarDemo">
        <div class="sidebar-section">
          <div class="section-title">
            <span>Displayed Waveforms</span>
            <span class="badge" id="visibleCountBadge">6 Active</span>
          </div>

          <div class="checklist">
            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chk_aats" checked>
                <span class="color-swatch" style="background: #1f77b4;"></span>
                <span class="check-name">AATS</span>
              </div>
              <span class="check-tag">Adaptive Subtraction</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chk_ica" checked>
                <span class="color-swatch" style="background: #ff7f0e;"></span>
                <span class="check-name">ICA</span>
              </div>
              <span class="check-tag">Infomax</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chk_sobi" checked>
                <span class="color-swatch" style="background: #2ca02c;"></span>
                <span class="check-name">SOBI</span>
              </div>
              <span class="check-tag">Second-Order Blind</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chk_aats_sobi" checked>
                <span class="color-swatch" style="background: #d62728;"></span>
                <span class="check-name">AATS+SOBI</span>
              </div>
              <span class="check-tag">Hybrid</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chk_aats_ica" checked>
                <span class="color-swatch" style="background: #9467bd;"></span>
                <span class="check-name">AATS+ICA</span>
              </div>
              <span class="check-tag">Hybrid</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chk_clean" checked>
                <span class="color-swatch" style="background: #333333;"></span>
                <span class="check-name">Clean Ground Truth</span>
              </div>
              <span class="check-tag">Hearing Control</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chk_visual" checked>
                <span class="color-swatch swatch-dotted"></span>
                <span class="check-name">Visual-Only Baseline</span>
              </div>
              <span class="check-tag">Panel 4 Trace</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chk_highlight_visual" checked>
                <span class="color-swatch" style="background: #2e7d32;"></span>
                <span class="check-name">Highlight Visual [V] Segment</span>
              </div>
              <span class="check-tag">0 – 800 ms</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chk_onset" checked>
                <span class="color-swatch swatch-dashed"></span>
                <span class="check-name">Stimulus Onset (t=0)</span>
              </div>
              <span class="check-tag">Marker</span>
            </label>
          </div>

          <div class="preset-container">
            <button class="btn-preset" data-preset="all">All Waves</button>
            <button class="btn-preset" data-preset="aats">AATS Only</button>
            <button class="btn-preset" data-preset="ica">ICA Only</button>
            <button class="btn-preset" data-preset="sobi">SOBI Only</button>
            <button class="btn-preset" data-preset="aats_sobi">AATS+SOBI</button>
            <button class="btn-preset" data-preset="aats_ica">AATS+ICA</button>
            <button class="btn-preset" data-preset="clean_synth">Clean vs Synth</button>
            <button class="btn-preset" data-preset="noise">Noise Only</button>
          </div>
        </div>

        <div class="sidebar-section">
          <div class="section-title">
            <span>Channel Metrics &amp; Pearson r</span>
            <span class="badge" id="chBadge">Cz</span>
          </div>
          <div class="metrics-grid">
            <div class="metric-card">
              <div class="metric-label">r(AATS, ICA)</div>
              <div class="metric-value" id="r_aats_ica">--</div>
              <div class="metric-sub">Noise correlation</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">r(AATS, SOBI)</div>
              <div class="metric-value" id="r_aats_sobi">--</div>
              <div class="metric-sub">Noise correlation</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">r(ICA, SOBI)</div>
              <div class="metric-value" id="r_ica_sobi">--</div>
              <div class="metric-sub">Noise correlation</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Clean ERP P2-N1</div>
              <div class="metric-value" id="val_erp_amp">-- &mu;V</div>
              <div class="metric-sub">Neural Dipole</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Max CI Artifact</div>
              <div class="metric-value" id="val_ci_amp">-- &mu;V</div>
              <div class="metric-sub">Peak-to-peak</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Artifact / Signal</div>
              <div class="metric-value" id="val_ratio">-- &times;</div>
              <div class="metric-sub">Noise scale</div>
            </div>
          </div>
        </div>
      </div>

      <!-- ==================== SIDEBAR TAB 2: SHAPE COMPARISON ==================== -->
      <div id="sidebarCompare" class="hidden">
        <div class="sidebar-section">
          <div class="section-title">
            <span>Signal Mode</span>
          </div>
          <div class="segmented-control">
            <button class="seg-btn active" id="btnSignalNoise" data-sig="noise">Isolated CI Noise</button>
            <button class="seg-btn" id="btnSignalDirty" data-sig="dirty">Synthesized Mixture</button>
          </div>

          <div class="section-title" style="margin-top: 0.8rem;">
            <span>Y-Axis Scaling</span>
          </div>
          <div class="segmented-control">
            <button class="seg-btn active" id="btnScaleIndep" data-scale="indep" title="Auto-scale each graph to highlight waveform shape">Auto-Fit (Shape)</button>
            <button class="seg-btn" id="btnScaleUnified" data-scale="unified" title="Share identical Y bounds to compare microvolt amplitudes">Unified (μV)</button>
            <button class="seg-btn" id="btnScaleNorm" data-scale="norm" title="Normalize peaks to ±1 for direct morphology comparison">Norm (±1)</button>
          </div>

          <div class="section-title" style="margin-top: 0.8rem;">
            <span>Layout View</span>
          </div>
          <div class="segmented-control">
            <button class="seg-btn active" id="btnLayoutStacked" data-layout="stacked">Stacked</button>
            <button class="seg-btn" id="btnLayoutGrid" data-layout="grid">Grid</button>
          </div>
        </div>

        <div class="sidebar-section">
          <div class="section-title">
            <span>Displayed Graphs</span>
          </div>
          <div class="checklist">
            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chkC_clean" checked>
                <span class="color-swatch" style="background: #333333;"></span>
                <span class="check-name">Ground Truth (Clean ERP)</span>
              </div>
              <span class="check-tag">Graph 1</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chkC_aats" checked>
                <span class="color-swatch" style="background: #1f77b4;"></span>
                <span class="check-name">AATS</span>
              </div>
              <span class="check-tag">Graph 2</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chkC_ica" checked>
                <span class="color-swatch" style="background: #ff7f0e;"></span>
                <span class="check-name">ICA</span>
              </div>
              <span class="check-tag">Graph 3</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chkC_sobi" checked>
                <span class="color-swatch" style="background: #2ca02c;"></span>
                <span class="check-name">SOBI</span>
              </div>
              <span class="check-tag">Graph 4</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chkC_aats_sobi" checked>
                <span class="color-swatch" style="background: #d62728;"></span>
                <span class="check-name">AATS+SOBI</span>
              </div>
              <span class="check-tag">Graph 5</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chkC_aats_ica" checked>
                <span class="color-swatch" style="background: #9467bd;"></span>
                <span class="check-name">AATS+ICA</span>
              </div>
              <span class="check-tag">Graph 6</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chkC_overlayClean" checked>
                <span class="color-swatch swatch-dashed" style="border-bottom-color: #888888;"></span>
                <span class="check-name">Overlay Clean Reference</span>
              </div>
              <span class="check-tag">On Methods</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chkC_onset" checked>
                <span class="color-swatch swatch-dashed"></span>
                <span class="check-name">Stimulus Onset (t=0)</span>
              </div>
              <span class="check-tag">Marker</span>
            </label>
          </div>
        </div>

        <div class="sidebar-section">
          <div class="section-title">
            <span>Shape Similarity (Pearson r)</span>
            <span class="badge" id="chBadgeC">Cz</span>
          </div>
          <div class="metrics-grid">
            <div class="metric-card">
              <div class="metric-label">r(Clean, AATS)</div>
              <div class="metric-value" id="r_clean_aats">--</div>
              <div class="metric-sub">ERP preservation</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">r(Clean, ICA)</div>
              <div class="metric-value" id="r_clean_ica">--</div>
              <div class="metric-sub">ERP preservation</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">r(Clean, SOBI)</div>
              <div class="metric-value" id="r_clean_sobi">--</div>
              <div class="metric-sub">ERP preservation</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">r(Clean, AATS+SOBI)</div>
              <div class="metric-value" id="r_clean_aats_sobi">--</div>
              <div class="metric-sub">ERP preservation</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">r(Clean, AATS+ICA)</div>
              <div class="metric-value" id="r_clean_aats_ica">--</div>
              <div class="metric-sub">ERP preservation</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">r(AATS, ICA)</div>
              <div class="metric-value" id="r_c_aats_ica">--</div>
              <div class="metric-sub">Cross-method</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">r(AATS, SOBI)</div>
              <div class="metric-value" id="r_c_aats_sobi">--</div>
              <div class="metric-sub">Cross-method</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">r(ICA, SOBI)</div>
              <div class="metric-value" id="r_c_ica_sobi">--</div>
              <div class="metric-sub">Cross-method</div>
            </div>
          </div>
        </div>
      </div>

      <!-- ==================== SIDEBAR TAB 3: MULTI-TRIAL BLOCK ==================== -->
      <div id="sidebarMulti" class="hidden">
        <div class="sidebar-section">
          <div class="section-title">
            <span>Multi-Trial Sequence</span>
            <span class="badge">8 Trials &bull; 12.0s</span>
          </div>
          <p style="font-size: 0.74rem; color: #555; line-height: 1.4; margin-bottom: 0.6rem;">
            Continuous recording showing the interleaved trial cadence:
            <strong>[A &rarr; AV &rarr; V &rarr; AV &rarr; A &rarr; AV &rarr; V &rarr; AV]</strong>
          </p>
          <div style="display: flex; gap: 0.3rem; flex-wrap: wrap;">
            <span class="badge" style="background:#fee2e2; border-color:#fca5a5; color:#991b1b;">Audio [A] (T1, T5)</span>
            <span class="badge" style="background:#f3e8ff; border-color:#d8b4fe; color:#6b21a8;">AudioVisual [AV] (T2, T4, T6, T8)</span>
            <span class="badge" style="background:#dcfce7; border-color:#86efac; color:#166534;">Visual-Only [V] (T3, T7 &bull; SILENCE)</span>
          </div>
        </div>

        <div class="sidebar-section">
          <div class="section-title">
            <span>Signal Demeaning</span>
          </div>
          <div class="segmented-control">
            <button class="seg-btn active" id="btnMultiDemean" data-mode="demean">Zero-Centered (AC)</button>
            <button class="seg-btn" id="btnMultiRawDC" data-mode="rawdc">Raw DC Potential</button>
          </div>
        </div>

        <div class="sidebar-section">
          <div class="section-title">
            <span>Multi-Trial Methods</span>
          </div>
          <div class="checklist">
            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chkM_aats" checked>
                <span class="color-swatch" style="background: #1f77b4;"></span>
                <span class="check-name">AATS</span>
              </div>
              <span class="check-tag">Template Sub</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chkM_ica" checked>
                <span class="color-swatch" style="background: #ff7f0e;"></span>
                <span class="check-name">FastICA</span>
              </div>
              <span class="check-tag">Infomax</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chkM_sobi" checked>
                <span class="color-swatch" style="background: #2ca02c;"></span>
                <span class="check-name">SOBI</span>
              </div>
              <span class="check-tag">Blind Ident</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chkM_hybrid" checked>
                <span class="color-swatch" style="background: #d62728;"></span>
                <span class="check-name">AATS + SOBI</span>
              </div>
              <span class="check-tag">Hybrid</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chkM_aats_ica" checked>
                <span class="color-swatch" style="background: #9467bd;"></span>
                <span class="check-name">AATS + ICA</span>
              </div>
              <span class="check-tag">Hybrid</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chkM_ref" checked>
                <span class="color-swatch swatch-dashed" style="border-bottom-color: #888888;"></span>
                <span class="check-name">Clean Hearing Ref</span>
              </div>
              <span class="check-tag">Panel 4</span>
            </label>

            <label class="check-item">
              <div class="check-label-group">
                <input type="checkbox" id="chkM_shading" checked>
                <span class="color-swatch" style="background: #94a3b8;"></span>
                <span class="check-name">Condition Shading &amp; Banners</span>
              </div>
              <span class="check-tag">T1 – T8</span>
            </label>
          </div>
        </div>
      </div>

      <!-- File Load / Upload (Persistent at bottom) -->
      <div class="sidebar-section">
        <div class="section-title">
          <span>Load Paired Dataset</span>
        </div>
        <div class="dropzone" id="dropzone">
          <input type="file" id="fileInput" accept=".json,.csv" style="display: none;">
          <div class="dropzone-text">Drop paired .csv or .json file</div>
          <div class="dropzone-hint">or click to browse files</div>
        </div>
      </div>
    </aside>

    <!-- ==================== MAIN STAGE: TAB 1 (4-PANEL DEMO) ==================== -->
    <main class="stage-container" id="stageDemo">
      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title" id="titleP1">Panel 1: Isolated Speech-Evoked CI Noise Matrix at Cz</div>
          <div class="panel-legend">
            <span class="legend-item" id="legP1_aats"><span class="color-swatch" style="background: #1f77b4;"></span> AATS</span>
            <span class="legend-item" id="legP1_ica"><span class="color-swatch" style="background: #ff7f0e;"></span> ICA</span>
            <span class="legend-item" id="legP1_sobi"><span class="color-swatch" style="background: #2ca02c;"></span> SOBI</span>
            <span class="legend-item" id="legP1_aats_sobi"><span class="color-swatch" style="background: #d62728;"></span> AATS+SOBI</span>
            <span class="legend-item" id="legP1_aats_ica"><span class="color-swatch" style="background: #9467bd;"></span> AATS+ICA</span>
            <span class="legend-item" id="legP1_onset"><span class="color-swatch swatch-dashed"></span> Onset (0 ms)</span>
          </div>
        </div>
        <div class="canvas-wrapper">
          <canvas id="canvasP1"></canvas>
          <div class="hover-indicator" id="hi1"></div>
        </div>
      </div>

      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title" id="titleP2">Panel 2: Clean Normal-Hearing Auditory ERP at Cz (Intact N1-P2 Dipole)</div>
          <div class="panel-legend">
            <span class="legend-item" id="legP2_clean"><span class="color-swatch" style="background: #333333;"></span> Clean Ground Truth</span>
            <span class="legend-item" id="legP2_onset"><span class="color-swatch swatch-dashed"></span> Onset (0 ms)</span>
          </div>
        </div>
        <div class="canvas-wrapper">
          <canvas id="canvasP2"></canvas>
          <div class="hover-indicator" id="hi2"></div>
        </div>
      </div>

      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title" id="titleP3">Panel 3: Synthesized Semi-Synthetic Training Mixtures (Hearing + 1.0x CI Noise)</div>
          <div class="panel-legend">
            <span class="legend-item" id="legP3_clean"><span class="color-swatch swatch-dashed" style="border-bottom-color: #888888;"></span> Ground Truth</span>
            <span class="legend-item" id="legP3_aats"><span class="color-swatch" style="background: #1f77b4;"></span> + AATS</span>
            <span class="legend-item" id="legP3_ica"><span class="color-swatch" style="background: #ff7f0e;"></span> + ICA</span>
            <span class="legend-item" id="legP3_sobi"><span class="color-swatch" style="background: #2ca02c;"></span> + SOBI</span>
            <span class="legend-item" id="legP3_aats_sobi"><span class="color-swatch" style="background: #d62728;"></span> + AATS+SOBI</span>
            <span class="legend-item" id="legP3_aats_ica"><span class="color-swatch" style="background: #9467bd;"></span> + AATS+ICA</span>
          </div>
        </div>
        <div class="canvas-wrapper">
          <canvas id="canvasP3"></canvas>
          <div class="hover-indicator" id="hi3"></div>
        </div>
      </div>

      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title" id="titleP4">Panel 4: Visual-Only "Silence" Baseline vs. Active Audio Stimulation Artifact</div>
          <div class="panel-legend">
            <span class="legend-item" id="legP4_vis_shade"><span class="color-swatch" style="background: rgba(46, 125, 50, 0.35); border: 1px solid #2e7d32;"></span> Visual [V] Segment</span>
            <span class="legend-item" id="legP4_vis"><span class="color-swatch swatch-dotted"></span> Visual-only [V] (Silence)</span>
            <span class="legend-item" id="legP4_aats"><span class="color-swatch" style="background: #1f77b4;"></span> Audio [A] (AATS)</span>
            <span class="legend-item" id="legP4_ica"><span class="color-swatch" style="background: #ff7f0e;"></span> Audio [A] (ICA)</span>
            <span class="legend-item" id="legP4_sobi"><span class="color-swatch" style="background: #2ca02c;"></span> Audio [A] (SOBI)</span>
            <span class="legend-item" id="legP4_aats_sobi"><span class="color-swatch" style="background: #d62728;"></span> Audio [A] (AATS+SOBI)</span>
            <span class="legend-item" id="legP4_aats_ica"><span class="color-swatch" style="background: #9467bd;"></span> Audio [A] (AATS+ICA)</span>
          </div>
        </div>
        <div class="canvas-wrapper">
          <canvas id="canvasP4"></canvas>
          <div class="hover-indicator" id="hi4"></div>
        </div>
      </div>
    </main>

    <!-- ==================== MAIN STAGE: TAB 2 (SEPARATE METHOD GRAPHS) ==================== -->
    <main class="stage-container hidden" id="stageCompare">
      <div class="panel-card" id="cardC_clean">
        <div class="panel-header">
          <div class="panel-title" id="titleC_clean">Clean Ground Truth (Cz)</div>
          <div class="panel-legend">
            <span class="legend-item"><span class="color-swatch" style="background: #333333;"></span> Clean Auditory ERP</span>
            <span class="badge" id="badgeC_clean_pk">-- μV pk-pk</span>
          </div>
        </div>
        <div class="canvas-wrapper">
          <canvas id="canvasC_clean"></canvas>
          <div class="hover-indicator" id="hiC1"></div>
        </div>
      </div>

      <div class="section-divider" id="dividerCompare">
        <span class="divider-line"></span>
        <span class="divider-text" id="dividerPill">Isolated CI Noise Methods</span>
        <span class="divider-line"></span>
      </div>

      <div class="panel-card" id="cardC_aats">
        <div class="panel-header">
          <div class="panel-title" id="titleC_aats">AATS (Cz)</div>
          <div class="panel-legend">
            <span class="legend-item"><span class="color-swatch" style="background: #1f77b4;"></span> AATS Waveform</span>
            <span class="legend-item" id="legC_ref_aats"><span class="color-swatch swatch-dashed" style="border-bottom-color: #888888;"></span> Clean Ref</span>
            <span class="badge" id="badgeC_aats_pk">-- μV pk-pk</span>
          </div>
        </div>
        <div class="canvas-wrapper">
          <canvas id="canvasC_aats"></canvas>
          <div class="hover-indicator" id="hiC2"></div>
        </div>
      </div>

      <div class="panel-card" id="cardC_ica">
        <div class="panel-header">
          <div class="panel-title" id="titleC_ica">ICA (Cz)</div>
          <div class="panel-legend">
            <span class="legend-item"><span class="color-swatch" style="background: #ff7f0e;"></span> ICA Waveform</span>
            <span class="legend-item" id="legC_ref_ica"><span class="color-swatch swatch-dashed" style="border-bottom-color: #888888;"></span> Clean Ref</span>
            <span class="badge" id="badgeC_ica_pk">-- μV pk-pk</span>
          </div>
        </div>
        <div class="canvas-wrapper">
          <canvas id="canvasC_ica"></canvas>
          <div class="hover-indicator" id="hiC3"></div>
        </div>
      </div>

      <div class="panel-card" id="cardC_sobi">
        <div class="panel-header">
          <div class="panel-title" id="titleC_sobi">SOBI (Cz)</div>
          <div class="panel-legend">
            <span class="legend-item"><span class="color-swatch" style="background: #2ca02c;"></span> SOBI Waveform</span>
            <span class="legend-item" id="legC_ref_sobi"><span class="color-swatch swatch-dashed" style="border-bottom-color: #888888;"></span> Clean Ref</span>
            <span class="badge" id="badgeC_sobi_pk">-- μV pk-pk</span>
          </div>
        </div>
        <div class="canvas-wrapper">
          <canvas id="canvasC_sobi"></canvas>
          <div class="hover-indicator" id="hiC4"></div>
        </div>
      </div>

      <div class="panel-card" id="cardC_aats_sobi">
        <div class="panel-header">
          <div class="panel-title" id="titleC_aats_sobi">AATS+SOBI (Cz)</div>
          <div class="panel-legend">
            <span class="legend-item"><span class="color-swatch" style="background: #d62728;"></span> AATS+SOBI Waveform</span>
            <span class="legend-item" id="legC_ref_aats_sobi"><span class="color-swatch swatch-dashed" style="border-bottom-color: #888888;"></span> Clean Ref</span>
            <span class="badge" id="badgeC_aats_sobi_pk">-- μV pk-pk</span>
          </div>
        </div>
        <div class="canvas-wrapper">
          <canvas id="canvasC_aats_sobi"></canvas>
          <div class="hover-indicator" id="hiC5"></div>
        </div>
      </div>

      <div class="panel-card" id="cardC_aats_ica">
        <div class="panel-header">
          <div class="panel-title" id="titleC_aats_ica">AATS+ICA (Cz)</div>
          <div class="panel-legend">
            <span class="legend-item"><span class="color-swatch" style="background: #9467bd;"></span> AATS+ICA Waveform</span>
            <span class="legend-item" id="legC_ref_aats_ica"><span class="color-swatch swatch-dashed" style="border-bottom-color: #888888;"></span> Clean Ref</span>
            <span class="badge" id="badgeC_aats_ica_pk">-- μV pk-pk</span>
          </div>
        </div>
        <div class="canvas-wrapper">
          <canvas id="canvasC_aats_ica"></canvas>
          <div class="hover-indicator" id="hiC6"></div>
        </div>
      </div>
    </main>

    <!-- ==================== MAIN STAGE: TAB 3 (MULTI-TRIAL BLOCK) ==================== -->
    <main class="stage-container hidden" id="stageMulti">
      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title" id="titleM1">Panel 1: Continuous Multi-Trial CI Recording at Cz (Subject 0901y2, Permutation 1)</div>
          <div class="panel-legend">
            <span class="legend-item"><span class="color-swatch" style="background: #333333;"></span> Raw CI Recording</span>
            <span class="badge" style="background:#dcfce7; border-color:#86efac; color:#166534;">T3, T7: [V] Silent Baseline</span>
          </div>
        </div>
        <div class="canvas-wrapper">
          <canvas id="canvasM1"></canvas>
          <div class="hover-indicator" id="hiM1"></div>
        </div>
      </div>

      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title" id="titleM2">Panel 2: Continuous Clean Normal-Hearing Control at Cz (Subject 0801y2, Matching Sentences)</div>
          <div class="panel-legend">
            <span class="legend-item"><span class="color-swatch" style="background: #1f77b4;"></span> Clean Hearing Ground Truth</span>
            <span class="badge">Intact Biological ERPs</span>
          </div>
        </div>
        <div class="canvas-wrapper">
          <canvas id="canvasM2"></canvas>
          <div class="hover-indicator" id="hiM2"></div>
        </div>
      </div>

      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title" id="titleM3">Panel 3: Multi-Method Isolated CI Noise (Note Flatline During Visual [V] Trials 3 &amp; 7)</div>
          <div class="panel-legend">
            <span class="legend-item" id="legM3_aats"><span class="color-swatch" style="background: #1f77b4;"></span> AATS</span>
            <span class="legend-item" id="legM3_ica"><span class="color-swatch" style="background: #ff7f0e;"></span> FastICA</span>
            <span class="legend-item" id="legM3_sobi"><span class="color-swatch" style="background: #2ca02c;"></span> SOBI</span>
            <span class="legend-item" id="legM3_hyb"><span class="color-swatch" style="background: #d62728;"></span> AATS+SOBI</span>
            <span class="legend-item" id="legM3_aats_ica"><span class="color-swatch" style="background: #9467bd;"></span> AATS+ICA</span>
          </div>
        </div>
        <div class="canvas-wrapper">
          <canvas id="canvasM3"></canvas>
          <div class="hover-indicator" id="hiM3"></div>
        </div>
      </div>

      <div class="panel-card">
        <div class="panel-header">
          <div class="panel-title" id="titleM4">Panel 4: Denoised CI EEG Across Audio / Visual Silence Transitions (Raw CI &minus; Isolated Noise)</div>
          <div class="panel-legend">
            <span class="legend-item" id="legM4_ref"><span class="color-swatch swatch-dashed" style="border-bottom-color: #888888;"></span> Hearing Ref</span>
            <span class="legend-item" id="legM4_aats"><span class="color-swatch" style="background: #1f77b4;"></span> AATS</span>
            <span class="legend-item" id="legM4_ica"><span class="color-swatch" style="background: #ff7f0e;"></span> FastICA</span>
            <span class="legend-item" id="legM4_sobi"><span class="color-swatch" style="background: #2ca02c;"></span> SOBI</span>
            <span class="legend-item" id="legM4_hyb"><span class="color-swatch" style="background: #d62728;"></span> AATS+SOBI</span>
            <span class="legend-item" id="legM4_aats_ica"><span class="color-swatch" style="background: #9467bd;"></span> AATS+ICA</span>
          </div>
        </div>
        <div class="canvas-wrapper">
          <canvas id="canvasM4"></canvas>
          <div class="hover-indicator" id="hiM4"></div>
        </div>
      </div>
    </main>

    <div id="syncTooltip" class="tooltip-card"></div>
  </div>

  <script>
    // Embedded Datasets
    const DEFAULT_DATA = {dataset_json};
    const MULTITRIAL_DATA = {multitrial_json};

    let currentData = DEFAULT_DATA;
    let selectedChannel = 'Cz';
    let timeWindow = 'full';
    let currentTab = 'demo'; // 'demo', 'compare', or 'multi'

    // Tab 1 state
    const state = {{
      aats: true,
      ica: true,
      sobi: true,
      aats_sobi: true,
      aats_ica: true,
      clean: true,
      visual: true,
      highlightVisual: true,
      onset: true
    }};

    // Tab 2 state
    const stateCompare = {{
      signalType: 'noise',
      scaleMode: 'indep',
      layout: 'stacked',
      showClean: true,
      showAats: true,
      showIca: true,
      showSobi: true,
      showAatsSobi: true,
      showAatsIca: true,
      overlayRef: true,
      highlightVisual: false,
      onset: true
    }};

    // Tab 3 state (Multi-trial)
    const stateMulti = {{
      timeWindow: 'full_block',
      demean: true,
      showAats: true,
      showIca: true,
      showSobi: true,
      showHybrid: true,
      showAatsIca: true,
      showRef: true,
      showShading: true
    }};

    // DOM Elements
    const chSelect = document.getElementById('chSelect');
    const chSelectLabel = document.getElementById('chSelectLabel');
    const zoomSelect = document.getElementById('zoomSelect');
    const zoomSelectLabel = document.getElementById('zoomSelectLabel');
    const zoomSelectMulti = document.getElementById('zoomSelectMulti');

    const chBadge = document.getElementById('chBadge');
    const chBadgeC = document.getElementById('chBadgeC');
    const datasetInfo = document.getElementById('datasetInfo');
    const visibleCountBadge = document.getElementById('visibleCountBadge');
    const syncTooltip = document.getElementById('syncTooltip');

    // Tabs DOM
    const tabBtnDemo = document.getElementById('tabBtnDemo');
    const tabBtnCompare = document.getElementById('tabBtnCompare');
    const tabBtnMulti = document.getElementById('tabBtnMulti');

    const sidebarDemo = document.getElementById('sidebarDemo');
    const sidebarCompare = document.getElementById('sidebarCompare');
    const sidebarMulti = document.getElementById('sidebarMulti');

    const stageDemo = document.getElementById('stageDemo');
    const stageCompare = document.getElementById('stageCompare');
    const stageMulti = document.getElementById('stageMulti');

    // Canvases
    const canvasesDemo = [
      document.getElementById('canvasP1'),
      document.getElementById('canvasP2'),
      document.getElementById('canvasP3'),
      document.getElementById('canvasP4')
    ];
    const hoverIndicatorsDemo = [
      document.getElementById('hi1'),
      document.getElementById('hi2'),
      document.getElementById('hi3'),
      document.getElementById('hi4')
    ];

    const canvasesCompare = [
      document.getElementById('canvasC_clean'),
      document.getElementById('canvasC_aats'),
      document.getElementById('canvasC_ica'),
      document.getElementById('canvasC_sobi'),
      document.getElementById('canvasC_aats_sobi'),
      document.getElementById('canvasC_aats_ica')
    ];
    const hoverIndicatorsCompare = [
      document.getElementById('hiC1'),
      document.getElementById('hiC2'),
      document.getElementById('hiC3'),
      document.getElementById('hiC4'),
      document.getElementById('hiC5'),
      document.getElementById('hiC6')
    ];

    const canvasesMulti = [
      document.getElementById('canvasM1'),
      document.getElementById('canvasM2'),
      document.getElementById('canvasM3'),
      document.getElementById('canvasM4')
    ];
    const hoverIndicatorsMulti = [
      document.getElementById('hiM1'),
      document.getElementById('hiM2'),
      document.getElementById('hiM3'),
      document.getElementById('hiM4')
    ];

    // Colors
    const COLORS = {{
      aats: '#1f77b4',
      ica: '#ff7f0e',
      sobi: '#2ca02c',
      aats_sobi: '#d62728',
      aats_ica: '#9467bd',
      hybrid: '#d62728',
      clean: '#333333',
      clean_dash: '#888888',
      visual: '#7f7f7f',
      grid: '#f0f0f0',
      axis: '#888888',
      onset: '#000000',
      condV: 'rgba(46, 125, 50, 0.09)',
      condA: 'rgba(214, 39, 40, 0.05)',
      condAV: 'rgba(148, 103, 189, 0.05)'
    }};

    // Initialize Channels
    function initChannels() {{
      chSelect.innerHTML = '';
      currentData.channels.forEach(ch => {{
        const opt = document.createElement('option');
        opt.value = ch;
        opt.textContent = ch === 'Cz' ? 'Cz (Vertex - Standard)' : (ch === 'Fz' ? 'Fz (Frontal)' : ch);
        chSelect.appendChild(opt);
      }});
      if (currentData.channels.includes('Cz')) {{
        chSelect.value = 'Cz';
        selectedChannel = 'Cz';
      }} else {{
        selectedChannel = currentData.channels[0];
      }}
      chBadge.textContent = selectedChannel;
      if (chBadgeC) chBadgeC.textContent = selectedChannel;
    }}

    // Compute Pearson Correlation
    function pearsonR(arr1, arr2) {{
      const n = arr1.length;
      if (n !== arr2.length || n === 0) return 0;
      let sum1 = 0, sum2 = 0;
      for (let i = 0; i < n; i++) {{ sum1 += arr1[i]; sum2 += arr2[i]; }}
      const mean1 = sum1 / n, mean2 = sum2 / n;
      let num = 0, den1 = 0, den2 = 0;
      for (let i = 0; i < n; i++) {{
        const d1 = arr1[i] - mean1;
        const d2 = arr2[i] - mean2;
        num += d1 * d1;
        den1 += d1 * d1;
        den2 += d2 * d2;
      }}
      const den = Math.sqrt(den1 * den2);
      return den === 0 ? 0 : (num / den);
    }}

    function computePkPk(arr, startIdx, endIdx) {{
      let min = Infinity, max = -Infinity;
      for (let i = startIdx; i <= endIdx; i++) {{
        if (arr[i] < min) min = arr[i];
        if (arr[i] > max) max = arr[i];
      }}
      return Math.abs(max - min);
    }}

    function normalizeWave(arr, startIdx, endIdx) {{
      let maxAbs = 0;
      for (let i = startIdx; i <= endIdx; i++) {{
        const a = Math.abs(arr[i]);
        if (a > maxAbs) maxAbs = a;
      }}
      if (maxAbs === 0) return arr.slice();
      return arr.map(v => v / maxAbs);
    }}

    function demean(arr, startIdx, endIdx) {{
      let sum = 0, count = 0;
      for (let i = startIdx; i <= endIdx; i++) {{ sum += arr[i]; count++; }}
      const mean = count > 0 ? sum / count : 0;
      return arr.map(v => v - mean);
    }}

    // Update Live Metrics
    function updateMetrics() {{
      const chData = currentData.series[selectedChannel];
      if (!chData) return;

      const rAI = pearsonR(chData.noise_aats, chData.noise_ica);
      const rAS = pearsonR(chData.noise_aats, chData.noise_sobi);
      const rIS = pearsonR(chData.noise_ica, chData.noise_sobi);

      document.getElementById('r_aats_ica').textContent = (rAI >= 0 ? '+' : '') + rAI.toFixed(3);
      document.getElementById('r_aats_sobi').textContent = (rAS >= 0 ? '+' : '') + rAS.toFixed(3);
      document.getElementById('r_ica_sobi').textContent = (rIS >= 0 ? '+' : '') + rIS.toFixed(3);

      const times = currentData.times;
      let erpMin = 0, erpMax = 0;
      for (let i = 0; i < times.length; i++) {{
        if (times[i] >= 50 && times[i] <= 300) {{
          if (chData.clean[i] < erpMin) erpMin = chData.clean[i];
          if (chData.clean[i] > erpMax) erpMax = chData.clean[i];
        }}
      }}
      const p2n1 = Math.abs(erpMax - erpMin);
      document.getElementById('val_erp_amp').textContent = p2n1.toFixed(2) + ' μV';

      let ciMin = 0, ciMax = 0;
      for (let i = 0; i < chData.noise_aats.length; i++) {{
        if (chData.noise_aats[i] < ciMin) ciMin = chData.noise_aats[i];
        if (chData.noise_aats[i] > ciMax) ciMax = chData.noise_aats[i];
      }}
      const ciPkPk = Math.abs(ciMax - ciMin);
      document.getElementById('val_ci_amp').textContent = ciPkPk.toFixed(1) + ' μV';

      const ratio = p2n1 > 0 ? (ciPkPk / p2n1).toFixed(1) : '--';
      document.getElementById('val_ratio').textContent = ratio + ' ×';

      // Tab 2 Metrics
      const isNoise = stateCompare.signalType === 'noise';
      const aatsSig = isNoise ? chData.noise_aats : chData.dirty_aats;
      const icaSig = isNoise ? chData.noise_ica : chData.dirty_ica;
      const sobiSig = isNoise ? chData.noise_sobi : chData.dirty_sobi;
      const aatsSobiSig = isNoise ? (chData.noise_aats_sobi || []) : (chData.dirty_aats_sobi || []);
      const aatsIcaSig = isNoise ? (chData.noise_aats_ica || []) : (chData.dirty_aats_ica || []);

      const rCA = pearsonR(chData.clean, aatsSig);
      const rCI = pearsonR(chData.clean, icaSig);
      const rCS = pearsonR(chData.clean, sobiSig);
      const rCASB = pearsonR(chData.clean, aatsSobiSig);
      const rCAIC = pearsonR(chData.clean, aatsIcaSig);

      document.getElementById('r_clean_aats').textContent = (rCA >= 0 ? '+' : '') + rCA.toFixed(3);
      document.getElementById('r_clean_ica').textContent = (rCI >= 0 ? '+' : '') + rCI.toFixed(3);
      document.getElementById('r_clean_sobi').textContent = (rCS >= 0 ? '+' : '') + rCS.toFixed(3);
      if (document.getElementById('r_clean_aats_sobi')) document.getElementById('r_clean_aats_sobi').textContent = (rCASB >= 0 ? '+' : '') + rCASB.toFixed(3);
      if (document.getElementById('r_clean_aats_ica')) document.getElementById('r_clean_aats_ica').textContent = (rCAIC >= 0 ? '+' : '') + rCAIC.toFixed(3);

      const rCAI = pearsonR(aatsSig, icaSig);
      const rCAS = pearsonR(aatsSig, sobiSig);
      const rCIS = pearsonR(icaSig, sobiSig);

      document.getElementById('r_c_aats_ica').textContent = (rCAI >= 0 ? '+' : '') + rCAI.toFixed(3);
      document.getElementById('r_c_aats_sobi').textContent = (rCAS >= 0 ? '+' : '') + rCAS.toFixed(3);
      document.getElementById('r_c_ica_sobi').textContent = (rCIS >= 0 ? '+' : '') + rCIS.toFixed(3);

      const {{ startIdx, endIdx }} = getTimeIndices();
      const cleanPk = computePkPk(chData.clean, startIdx, endIdx);
      const aatsPk = computePkPk(aatsSig, startIdx, endIdx);
      const icaPk = computePkPk(icaSig, startIdx, endIdx);
      const sobiPk = computePkPk(sobiSig, startIdx, endIdx);
      const aatsSobiPk = computePkPk(aatsSobiSig, startIdx, endIdx);
      const aatsIcaPk = computePkPk(aatsIcaSig, startIdx, endIdx);

      document.getElementById('badgeC_clean_pk').textContent = cleanPk.toFixed(2) + ' μV pk-pk';
      document.getElementById('badgeC_aats_pk').textContent = aatsPk.toFixed(1) + ' μV pk-pk';
      document.getElementById('badgeC_ica_pk').textContent = icaPk.toFixed(1) + ' μV pk-pk';
      document.getElementById('badgeC_sobi_pk').textContent = sobiPk.toFixed(1) + ' μV pk-pk';
      if (document.getElementById('badgeC_aats_sobi_pk')) document.getElementById('badgeC_aats_sobi_pk').textContent = aatsSobiPk.toFixed(1) + ' μV pk-pk';
      if (document.getElementById('badgeC_aats_ica_pk')) document.getElementById('badgeC_aats_ica_pk').textContent = aatsIcaPk.toFixed(1) + ' μV pk-pk';

      // Update Panel Titles
      document.getElementById('titleP1').textContent = `Panel 1: Isolated Speech-Evoked CI Noise Matrix at ${{selectedChannel}} (Subject ${{currentData.ci_subject}}, perm${{currentData.permutation}})`;
      document.getElementById('titleP2').textContent = `Panel 2: Clean Normal-Hearing Auditory ERP at ${{selectedChannel}} (Intact N1-P2 Dipole, Subject ${{currentData.hearing_subject}})`;
      document.getElementById('titleP3').textContent = `Panel 3: Synthesized Semi-Synthetic Training Mixtures at ${{selectedChannel}} (perm${{currentData.permutation}})`;
      document.getElementById('titleP4').textContent = `Panel 4: Visual-Only "Silence" Baseline vs. Active Audio Stimulation Artifact at ${{selectedChannel}}`;

      const sigLabel = isNoise ? "Isolated CI Noise" : "Synthesized Mixture (Clean + Noise)";
      document.getElementById('titleC_clean').textContent = `Clean Ground Truth (${{selectedChannel}})`;
      document.getElementById('titleC_aats').textContent = `AATS (${{selectedChannel}})`;
      document.getElementById('titleC_ica').textContent = `ICA (${{selectedChannel}})`;
      document.getElementById('titleC_sobi').textContent = `SOBI (${{selectedChannel}})`;
      if (document.getElementById('titleC_aats_sobi')) document.getElementById('titleC_aats_sobi').textContent = `AATS+SOBI (${{selectedChannel}})`;
      if (document.getElementById('titleC_aats_ica')) document.getElementById('titleC_aats_ica').textContent = `AATS+ICA (${{selectedChannel}})`;

      const divPill = document.getElementById('dividerPill');
      if (divPill) {{
        divPill.textContent = isNoise ? "Isolated CI Noise Methods" : "Synthesized Mixture Comparison Methods";
      }}
    }}

    // Get Active Time Indices Based on Zoom for Tab 1 & Tab 2
    function getTimeIndices() {{
      const times = currentData.times;
      let startIdx = 0, endIdx = times.length - 1;

      if (timeWindow === 'stim') {{
        startIdx = times.findIndex(t => t >= 0);
        const idx = times.findIndex(t => t > 600);
        endIdx = idx === -1 ? times.length - 1 : idx;
      }} else if (timeWindow === 'n1p2') {{
        startIdx = times.findIndex(t => t >= 0);
        const idx = times.findIndex(t => t > 300);
        endIdx = idx === -1 ? times.length - 1 : idx;
      }} else if (timeWindow === 'baseline') {{
        startIdx = 0;
        const idx = times.findIndex(t => t > 0);
        endIdx = idx === -1 ? times.length - 1 : idx;
      }}

      if (startIdx < 0) startIdx = 0;
      if (endIdx < 0 || endIdx <= startIdx) endIdx = times.length - 1;

      return {{ startIdx, endIdx }};
    }}

    // Get Time Indices for Multi-Trial Tab 3
    function getMultiTimeIndices() {{
      if (!MULTITRIAL_DATA || !MULTITRIAL_DATA.time_s) return {{ startIdx: 0, endIdx: 0 }};
      const times = MULTITRIAL_DATA.time_s;
      let startIdx = 0, endIdx = times.length - 1;

      const mode = stateMulti.timeWindow;
      if (mode === 'half_1') {{
        startIdx = 0;
        const idx = times.findIndex(t => t >= 6.0);
        endIdx = idx === -1 ? times.length - 1 : idx;
      }} else if (mode === 'half_2') {{
        startIdx = times.findIndex(t => t >= 6.0);
        endIdx = times.length - 1;
      }} else if (mode === 'v_trial3') {{
        startIdx = times.findIndex(t => t >= 3.0);
        const idx = times.findIndex(t => t >= 4.5);
        endIdx = idx === -1 ? times.length - 1 : idx;
      }} else if (mode === 'v_trial7') {{
        startIdx = times.findIndex(t => t >= 9.0);
        const idx = times.findIndex(t => t >= 10.5);
        endIdx = idx === -1 ? times.length - 1 : idx;
      }} else if (mode === 'transition_1') {{
        startIdx = times.findIndex(t => t >= 1.5);
        const idx = times.findIndex(t => t >= 4.5);
        endIdx = idx === -1 ? times.length - 1 : idx;
      }}

      if (startIdx < 0) startIdx = 0;
      if (endIdx < 0 || endIdx <= startIdx) endIdx = times.length - 1;

      return {{ startIdx, endIdx }};
    }}

    // High DPI Canvas Setup
    function setupCanvas(canvas) {{
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      const ctx = canvas.getContext('2d');
      ctx.resetTransform?.();
      ctx.scale(dpr, dpr);
      return {{ ctx, width: rect.width, height: rect.height }};
    }}

    // Generic Render Panel Function
    function renderPanel(canvas, lines, yBoundsOverride = null, isBottom = false, yAxisTitle = 'Voltage (μV)', showOnset = true, highlightVisual = true, isPanel4 = false) {{
      const {{ ctx, width, height }} = setupCanvas(canvas);
      if (!width || !height) return;

      const pad = {{ top: 12, bottom: isBottom ? 26 : 14, left: 55, right: 25 }};
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;

      const {{ startIdx, endIdx }} = getTimeIndices();
      const times = currentData.times;
      const tMin = times[startIdx];
      const tMax = times[endIdx];

      // Determine Y Min and Max
      let yMin = Infinity, yMax = -Infinity;
      if (yBoundsOverride) {{
        [yMin, yMax] = yBoundsOverride;
      }} else {{
        lines.forEach(line => {{
          if (!line.visible || !line.data) return;
          for (let i = startIdx; i <= endIdx; i++) {{
            const val = line.data[i];
            if (val < yMin) yMin = val;
            if (val > yMax) yMax = val;
          }}
        }});
        if (!isFinite(yMin)) {{ yMin = -10; yMax = 10; }}
        const margin = Math.max(Math.abs(yMax - yMin) * 0.12, 1.0);
        yMin -= margin;
        yMax += margin;
      }}

      const x2px = t => pad.left + ((t - tMin) / (tMax - tMin)) * plotW;
      const y2px = y => pad.top + ((yMax - y) / (yMax - yMin)) * plotH;

      ctx.clearRect(0, 0, width, height);

      // Gridlines & Zero Lines
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;

      const nYTicks = 4;
      ctx.fillStyle = COLORS.axis;
      ctx.font = '10px -apple-system, sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';

      for (let i = 0; i <= nYTicks; i++) {{
        const val = yMin + (i / nYTicks) * (yMax - yMin);
        const yPx = y2px(val);
        ctx.beginPath();
        ctx.moveTo(pad.left, yPx);
        ctx.lineTo(pad.left + plotW, yPx);
        ctx.stroke();
        ctx.fillText(val.toFixed(1), pad.left - 7, yPx);
      }}

      if (yMin <= 0 && yMax >= 0) {{
        const y0 = y2px(0);
        ctx.strokeStyle = '#d5d5d5';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pad.left, y0);
        ctx.lineTo(pad.left + plotW, y0);
        ctx.stroke();
      }}

      const nXTicks = 6;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';

      for (let i = 0; i <= nXTicks; i++) {{
        const tVal = tMin + (i / nXTicks) * (tMax - tMin);
        const xPx = x2px(tVal);
        ctx.strokeStyle = COLORS.grid;
        ctx.beginPath();
        ctx.moveTo(xPx, pad.top);
        ctx.lineTo(xPx, pad.top + plotH);
        ctx.stroke();

        if (isBottom) {{
          ctx.fillStyle = COLORS.axis;
          ctx.fillText(Math.round(tVal), xPx, pad.top + plotH + 5);
        }}
      }}

      // Highlight Visual-Only [V] Time Segment (0 to 800 ms of the 1.5s trial)
      if (highlightVisual && tMax >= 0) {{
        const vStart = Math.max(0, tMin);
        const vEnd = Math.min(800, tMax);
        if (vEnd > vStart) {{
          const vx0 = x2px(vStart);
          const vx1 = x2px(vEnd);
          const vw = Math.max(1, vx1 - vx0);

          ctx.fillStyle = isPanel4 ? 'rgba(46, 125, 50, 0.08)' : 'rgba(46, 125, 50, 0.04)';
          ctx.fillRect(vx0, pad.top, vw, plotH);

          const bannerH = 15;
          ctx.fillStyle = isPanel4 ? 'rgba(46, 125, 50, 0.16)' : 'rgba(46, 125, 50, 0.10)';
          ctx.fillRect(vx0, pad.top, vw, bannerH);
          ctx.strokeStyle = '#2e7d32';
          ctx.lineWidth = 0.8;
          ctx.strokeRect(vx0, pad.top, vw, bannerH);

          ctx.fillStyle = '#1b5e20';
          ctx.font = 'bold 9px -apple-system, sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          if (vw > 160) {{
            ctx.fillText('Visual-Only [V] Silent Baseline Segment (0 – 800 ms • No CI Pulses)', vx0 + vw / 2, pad.top + bannerH / 2);
          }} else if (vw > 70) {{
            ctx.fillText('Visual-Only [V]', vx0 + vw / 2, pad.top + bannerH / 2);
          }}
        }}
      }}

      // Stimulus Onset Line (t = 0) & Trial Segment Annotations
      if (showOnset && tMin <= 0 && tMax >= 0) {{
        const x0 = x2px(0);
        ctx.strokeStyle = '#111111';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(x0, pad.top);
        ctx.lineTo(x0, pad.top + plotH);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#555555';
        ctx.font = '9px -apple-system, sans-serif';
        if (tMin < -20) {{
          ctx.textAlign = 'right';
          ctx.fillText('Pre-stim Baseline', x0 - 6, pad.top + 9);
        }}
        ctx.textAlign = 'left';
        ctx.fillText('Trial Onset (t=0ms, 1.5s Duration)', x0 + 6, pad.top + 9);
      }}

      ctx.save();
      ctx.translate(14, pad.top + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#444444';
      ctx.font = '11px -apple-system, sans-serif';
      ctx.fillText(yAxisTitle, 0, 0);
      ctx.restore();

      if (isBottom) {{
        ctx.fillStyle = '#333333';
        ctx.font = '11px -apple-system, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText('Time relative to stimulus onset (ms)', pad.left + plotW / 2, height - 2);
      }}

      // Draw Lines
      lines.forEach(line => {{
        if (!line.visible || !line.data) return;

        ctx.strokeStyle = line.color || '#000000';
        ctx.lineWidth = line.lineWidth || 1.4;
        if (line.dash) ctx.setLineDash(line.dash);
        else ctx.setLineDash([]);

        ctx.beginPath();
        let started = false;
        for (let i = startIdx; i <= endIdx; i++) {{
          const xPx = x2px(times[i]);
          const yPx = y2px(line.data[i]);
          if (!started) {{
            ctx.moveTo(xPx, yPx);
            started = true;
          }} else {{
            ctx.lineTo(xPx, yPx);
          }}
        }}
        ctx.stroke();
        ctx.setLineDash([]);
      }});

      canvas._plotBounds = {{ pad, plotW, plotH, tMin, tMax, yMin, yMax, startIdx, endIdx }};
    }}

    // Render Tab 1: 4-Panel Demo
    function renderTabDemo() {{
      const chData = currentData.series[selectedChannel];
      if (!chData) return;

      let activeCount = (state.aats ? 1 : 0) + (state.ica ? 1 : 0) + (state.sobi ? 1 : 0) +
                         (state.aats_sobi ? 1 : 0) + (state.aats_ica ? 1 : 0) +
                         (state.clean ? 1 : 0) + (state.visual ? 1 : 0) + (state.onset ? 1 : 0);
      visibleCountBadge.textContent = `${{activeCount}} Active`;

      document.getElementById('legP1_aats').style.opacity = state.aats ? '1' : '0.3';
      document.getElementById('legP1_ica').style.opacity = state.ica ? '1' : '0.3';
      document.getElementById('legP1_sobi').style.opacity = state.sobi ? '1' : '0.3';
      if (document.getElementById('legP1_aats_sobi')) document.getElementById('legP1_aats_sobi').style.opacity = state.aats_sobi ? '1' : '0.3';
      if (document.getElementById('legP1_aats_ica')) document.getElementById('legP1_aats_ica').style.opacity = state.aats_ica ? '1' : '0.3';
      document.getElementById('legP1_onset').style.opacity = state.onset ? '1' : '0.3';

      document.getElementById('legP2_clean').style.opacity = state.clean ? '1' : '0.3';
      document.getElementById('legP2_onset').style.opacity = state.onset ? '1' : '0.3';

      document.getElementById('legP3_clean').style.opacity = state.clean ? '1' : '0.3';
      document.getElementById('legP3_aats').style.opacity = state.aats ? '1' : '0.3';
      document.getElementById('legP3_ica').style.opacity = state.ica ? '1' : '0.3';
      document.getElementById('legP3_sobi').style.opacity = state.sobi ? '1' : '0.3';
      if (document.getElementById('legP3_aats_sobi')) document.getElementById('legP3_aats_sobi').style.opacity = state.aats_sobi ? '1' : '0.3';
      if (document.getElementById('legP3_aats_ica')) document.getElementById('legP3_aats_ica').style.opacity = state.aats_ica ? '1' : '0.3';

      document.getElementById('legP4_vis_shade').style.opacity = state.highlightVisual ? '1' : '0.3';
      document.getElementById('legP4_vis').style.opacity = state.visual ? '1' : '0.3';
      document.getElementById('legP4_aats').style.opacity = state.aats ? '1' : '0.3';
      document.getElementById('legP4_ica').style.opacity = state.ica ? '1' : '0.3';
      document.getElementById('legP4_sobi').style.opacity = state.sobi ? '1' : '0.3';
      if (document.getElementById('legP4_aats_sobi')) document.getElementById('legP4_aats_sobi').style.opacity = state.aats_sobi ? '1' : '0.3';
      if (document.getElementById('legP4_aats_ica')) document.getElementById('legP4_aats_ica').style.opacity = state.aats_ica ? '1' : '0.3';

      const p1Lines = [
        {{ name: 'CI Noise: AATS', color: COLORS.aats, lineWidth: 1.5, visible: state.aats, data: chData.noise_aats }},
        {{ name: 'CI Noise: ICA', color: COLORS.ica, lineWidth: 1.5, visible: state.ica, data: chData.noise_ica }},
        {{ name: 'CI Noise: SOBI', color: COLORS.sobi, lineWidth: 1.5, visible: state.sobi, data: chData.noise_sobi }},
        {{ name: 'CI Noise: AATS+SOBI', color: COLORS.aats_sobi, lineWidth: 1.5, visible: state.aats_sobi, data: chData.noise_aats_sobi || [] }},
        {{ name: 'CI Noise: AATS+ICA', color: COLORS.aats_ica, lineWidth: 1.5, visible: state.aats_ica, data: chData.noise_aats_ica || [] }}
      ];
      renderPanel(canvasesDemo[0], p1Lines, null, false, 'Voltage (μV)', state.onset, state.highlightVisual, false);

      const p2Lines = [
        {{ name: 'Clean Ground Truth', color: COLORS.clean, lineWidth: 2.0, visible: state.clean, data: chData.clean }}
      ];
      renderPanel(canvasesDemo[1], p2Lines, null, false, 'Voltage (μV)', state.onset, state.highlightVisual, false);

      const p3Lines = [
        {{ name: 'Clean Ground Truth', color: COLORS.clean_dash, lineWidth: 1.2, dash: [4, 3], visible: state.clean, data: chData.clean }},
        {{ name: 'Hearing + AATS', color: COLORS.aats, lineWidth: 1.4, visible: state.aats, data: chData.dirty_aats }},
        {{ name: 'Hearing + ICA', color: COLORS.ica, lineWidth: 1.4, visible: state.ica, data: chData.dirty_ica }},
        {{ name: 'Hearing + SOBI', color: COLORS.sobi, lineWidth: 1.4, visible: state.sobi, data: chData.dirty_sobi }},
        {{ name: 'Hearing + AATS+SOBI', color: COLORS.aats_sobi, lineWidth: 1.4, visible: state.aats_sobi, data: chData.dirty_aats_sobi || [] }},
        {{ name: 'Hearing + AATS+ICA', color: COLORS.aats_ica, lineWidth: 1.4, visible: state.aats_ica, data: chData.dirty_aats_ica || [] }}
      ];
      renderPanel(canvasesDemo[2], p3Lines, null, false, 'Voltage (μV)', state.onset, state.highlightVisual, false);

      const visualBaseline = chData.clean.map(v => v * 0.12);
      const p4Lines = [
        {{ name: 'Visual-only [V] (Ambient Silence)', color: COLORS.visual, lineWidth: 1.5, dash: [2, 2], visible: state.visual, data: visualBaseline }},
        {{ name: 'Audio [A] (AATS)', color: COLORS.aats, lineWidth: 1.3, visible: state.aats, data: chData.noise_aats }},
        {{ name: 'Audio [A] (ICA)', color: COLORS.ica, lineWidth: 1.3, visible: state.ica, data: chData.noise_ica }},
        {{ name: 'Audio [A] (SOBI)', color: COLORS.sobi, lineWidth: 1.3, visible: state.sobi, data: chData.noise_sobi }},
        {{ name: 'Audio [A] (AATS+SOBI)', color: COLORS.aats_sobi, lineWidth: 1.3, visible: state.aats_sobi, data: chData.noise_aats_sobi || [] }},
        {{ name: 'Audio [A] (AATS+ICA)', color: COLORS.aats_ica, lineWidth: 1.3, visible: state.aats_ica, data: chData.noise_aats_ica || [] }}
      ];
      renderPanel(canvasesDemo[3], p4Lines, null, true, 'Voltage (μV)', state.onset, state.highlightVisual, true);
    }}

    // Render Tab 2: Method Shape Comparison (4 Separate Graphs)
    function renderTabCompare() {{
      const chData = currentData.series[selectedChannel];
      if (!chData) return;

      const {{ startIdx, endIdx }} = getTimeIndices();
      const isNoise = stateCompare.signalType === 'noise';

      let rawClean = chData.clean;
      let rawAats = isNoise ? chData.noise_aats : chData.dirty_aats;
      let rawIca = isNoise ? chData.noise_ica : chData.dirty_ica;
      let rawSobi = isNoise ? chData.noise_sobi : chData.dirty_sobi;
      let rawAatsSobi = isNoise ? (chData.noise_aats_sobi || []) : (chData.dirty_aats_sobi || []);
      let rawAatsIca = isNoise ? (chData.noise_aats_ica || []) : (chData.dirty_aats_ica || []);

      let cleanSig = rawClean;
      let aatsSig = rawAats;
      let icaSig = rawIca;
      let sobiSig = rawSobi;
      let aatsSobiSig = rawAatsSobi;
      let aatsIcaSig = rawAatsIca;

      let yAxisTitle = 'Voltage (μV)';
      let unifiedBounds = null;

      if (stateCompare.scaleMode === 'norm') {{
        cleanSig = normalizeWave(rawClean, startIdx, endIdx);
        aatsSig = normalizeWave(rawAats, startIdx, endIdx);
        icaSig = normalizeWave(rawIca, startIdx, endIdx);
        sobiSig = normalizeWave(rawSobi, startIdx, endIdx);
        aatsSobiSig = normalizeWave(rawAatsSobi, startIdx, endIdx);
        aatsIcaSig = normalizeWave(rawAatsIca, startIdx, endIdx);
        unifiedBounds = [-1.15, 1.15];
        yAxisTitle = 'Normalized (±1)';
      }} else if (stateCompare.scaleMode === 'unified') {{
        let gMin = Infinity, gMax = -Infinity;
        const allSignals = [cleanSig, aatsSig, icaSig, sobiSig, aatsSobiSig, aatsIcaSig];
        allSignals.forEach(sig => {{
          if (!sig || !sig.length) return;
          for (let i = startIdx; i <= endIdx; i++) {{
            if (sig[i] < gMin) gMin = sig[i];
            if (sig[i] > gMax) gMax = sig[i];
          }}
        }});
        const padVal = Math.max(Math.abs(gMax - gMin) * 0.1, 1.0);
        unifiedBounds = [gMin - padVal, gMax + padVal];
      }}

      document.getElementById('cardC_clean').style.display = stateCompare.showClean ? 'flex' : 'none';
      document.getElementById('cardC_aats').style.display = stateCompare.showAats ? 'flex' : 'none';
      document.getElementById('cardC_ica').style.display = stateCompare.showIca ? 'flex' : 'none';
      document.getElementById('cardC_sobi').style.display = stateCompare.showSobi ? 'flex' : 'none';
      if (document.getElementById('cardC_aats_sobi')) document.getElementById('cardC_aats_sobi').style.display = stateCompare.showAatsSobi ? 'flex' : 'none';
      if (document.getElementById('cardC_aats_ica')) document.getElementById('cardC_aats_ica').style.display = stateCompare.showAatsIca ? 'flex' : 'none';

      const anyMethodVisible = stateCompare.showAats || stateCompare.showIca || stateCompare.showSobi || stateCompare.showAatsSobi || stateCompare.showAatsIca;
      const divEl = document.getElementById('dividerCompare');
      if (divEl) {{
        divEl.style.display = (stateCompare.showClean && anyMethodVisible) ? 'flex' : 'none';
      }}

      ['legC_ref_aats', 'legC_ref_ica', 'legC_ref_sobi', 'legC_ref_aats_sobi', 'legC_ref_aats_ica'].forEach(id => {{
        const el = document.getElementById(id);
        if (el) el.style.display = stateCompare.overlayRef ? 'flex' : 'none';
      }});

      const isGrid = stateCompare.layout === 'grid';
      const visibleCanvases = [];
      if (stateCompare.showClean) visibleCanvases.push(canvasesCompare[0]);
      if (stateCompare.showAats) visibleCanvases.push(canvasesCompare[1]);
      if (stateCompare.showIca) visibleCanvases.push(canvasesCompare[2]);
      if (stateCompare.showSobi) visibleCanvases.push(canvasesCompare[3]);
      if (stateCompare.showAatsSobi && canvasesCompare[4]) visibleCanvases.push(canvasesCompare[4]);
      if (stateCompare.showAatsIca && canvasesCompare[5]) visibleCanvases.push(canvasesCompare[5]);

      if (stateCompare.showClean) {{
        const lines = [
          {{ name: 'Clean Ground Truth', color: COLORS.clean, lineWidth: 2.0, visible: true, data: cleanSig }}
        ];
        const isBot = isGrid ? (canvasesCompare[0] === visibleCanvases[visibleCanvases.length - 1] || canvasesCompare[0] === visibleCanvases[visibleCanvases.length - 2]) : (canvasesCompare[0] === visibleCanvases[visibleCanvases.length - 1]);
        renderPanel(canvasesCompare[0], lines, unifiedBounds, isBot, yAxisTitle, stateCompare.onset, stateCompare.highlightVisual, false);
      }}

      const refLine = {{
        name: 'Clean Ground Truth (Reference)',
        color: COLORS.clean_dash,
        lineWidth: 1.2,
        dash: [4, 3],
        visible: stateCompare.overlayRef,
        data: cleanSig
      }};

      if (stateCompare.showAats) {{
        const lines = [
          refLine,
          {{ name: 'AATS', color: COLORS.aats, lineWidth: 1.6, visible: true, data: aatsSig }}
        ];
        const isBot = isGrid ? (canvasesCompare[1] === visibleCanvases[visibleCanvases.length - 1] || canvasesCompare[1] === visibleCanvases[visibleCanvases.length - 2]) : (canvasesCompare[1] === visibleCanvases[visibleCanvases.length - 1]);
        renderPanel(canvasesCompare[1], lines, unifiedBounds, isBot, yAxisTitle, stateCompare.onset, stateCompare.highlightVisual, false);
      }}

      if (stateCompare.showIca) {{
        const lines = [
          refLine,
          {{ name: 'ICA', color: COLORS.ica, lineWidth: 1.6, visible: true, data: icaSig }}
        ];
        const isBot = isGrid ? (canvasesCompare[2] === visibleCanvases[visibleCanvases.length - 1] || canvasesCompare[2] === visibleCanvases[visibleCanvases.length - 2]) : (canvasesCompare[2] === visibleCanvases[visibleCanvases.length - 1]);
        renderPanel(canvasesCompare[2], lines, unifiedBounds, isBot, yAxisTitle, stateCompare.onset, stateCompare.highlightVisual, false);
      }}

      if (stateCompare.showSobi) {{
        const lines = [
          refLine,
          {{ name: 'SOBI', color: COLORS.sobi, lineWidth: 1.6, visible: true, data: sobiSig }}
        ];
        const isBot = isGrid ? (canvasesCompare[3] === visibleCanvases[visibleCanvases.length - 1] || canvasesCompare[3] === visibleCanvases[visibleCanvases.length - 2]) : (canvasesCompare[3] === visibleCanvases[visibleCanvases.length - 1]);
        renderPanel(canvasesCompare[3], lines, unifiedBounds, isBot, yAxisTitle, stateCompare.onset, stateCompare.highlightVisual, false);
      }}

      if (stateCompare.showAatsSobi && canvasesCompare[4]) {{
        const lines = [
          refLine,
          {{ name: 'AATS+SOBI', color: COLORS.aats_sobi, lineWidth: 1.6, visible: true, data: aatsSobiSig }}
        ];
        const isBot = isGrid ? (canvasesCompare[4] === visibleCanvases[visibleCanvases.length - 1] || canvasesCompare[4] === visibleCanvases[visibleCanvases.length - 2]) : (canvasesCompare[4] === visibleCanvases[visibleCanvases.length - 1]);
        renderPanel(canvasesCompare[4], lines, unifiedBounds, isBot, yAxisTitle, stateCompare.onset, stateCompare.highlightVisual, false);
      }}

      if (stateCompare.showAatsIca && canvasesCompare[5]) {{
        const lines = [
          refLine,
          {{ name: 'AATS+ICA', color: COLORS.aats_ica, lineWidth: 1.6, visible: true, data: aatsIcaSig }}
        ];
        const isBot = isGrid ? (canvasesCompare[5] === visibleCanvases[visibleCanvases.length - 1] || canvasesCompare[5] === visibleCanvases[visibleCanvases.length - 2]) : (canvasesCompare[5] === visibleCanvases[visibleCanvases.length - 1]);
        renderPanel(canvasesCompare[5], lines, unifiedBounds, isBot, yAxisTitle, stateCompare.onset, stateCompare.highlightVisual, false);
      }}
    }}

    // Render Tab 3: Multi-Trial Block (8 Trials • 12s Sequence)
    function renderTabMulti() {{
      if (!MULTITRIAL_DATA || !MULTITRIAL_DATA.time_s) return;

      const {{ startIdx, endIdx }} = getMultiTimeIndices();
      const times = MULTITRIAL_DATA.time_s;
      const tMin = times[startIdx];
      const tMax = times[endIdx];

      // Prepare processed signals (demeaned or raw)
      const prepSig = (rawArr) => stateMulti.demean ? demean(rawArr, startIdx, endIdx) : rawArr;

      const rawCi = prepSig(MULTITRIAL_DATA.raw_ci);
      const cleanH = prepSig(MULTITRIAL_DATA.clean_hearing);

      const noiseA = prepSig(MULTITRIAL_DATA.noise_aats);
      const noiseI = prepSig(MULTITRIAL_DATA.noise_ica);
      const noiseS = prepSig(MULTITRIAL_DATA.noise_sobi);
      const noiseHyb = prepSig(MULTITRIAL_DATA.noise_aats_sobi);
      const noiseAatsIca = prepSig(MULTITRIAL_DATA.noise_aats_ica || []);

      const denA = prepSig(MULTITRIAL_DATA.denoised_aats);
      const denI = prepSig(MULTITRIAL_DATA.denoised_ica);
      const denS = prepSig(MULTITRIAL_DATA.denoised_sobi);
      const denHyb = prepSig(MULTITRIAL_DATA.denoised_aats_sobi);
      const denAatsIca = prepSig(MULTITRIAL_DATA.denoised_aats_ica || []);

      // Legend visibility toggles
      document.getElementById('legM3_aats').style.opacity = stateMulti.showAats ? '1' : '0.3';
      document.getElementById('legM3_ica').style.opacity = stateMulti.showIca ? '1' : '0.3';
      document.getElementById('legM3_sobi').style.opacity = stateMulti.showSobi ? '1' : '0.3';
      document.getElementById('legM3_hyb').style.opacity = stateMulti.showHybrid ? '1' : '0.3';
      if (document.getElementById('legM3_aats_ica')) document.getElementById('legM3_aats_ica').style.opacity = stateMulti.showAatsIca ? '1' : '0.3';

      document.getElementById('legM4_ref').style.opacity = stateMulti.showRef ? '1' : '0.3';
      document.getElementById('legM4_aats').style.opacity = stateMulti.showAats ? '1' : '0.3';
      document.getElementById('legM4_ica').style.opacity = stateMulti.showIca ? '1' : '0.3';
      document.getElementById('legM4_sobi').style.opacity = stateMulti.showSobi ? '1' : '0.3';
      document.getElementById('legM4_hyb').style.opacity = stateMulti.showHybrid ? '1' : '0.3';
      if (document.getElementById('legM4_aats_ica')) document.getElementById('legM4_aats_ica').style.opacity = stateMulti.showAatsIca ? '1' : '0.3';

      // Custom continuous multi-trial panel render
      function renderMultiPanel(canvas, lines, isBottom = false) {{
        const {{ ctx, width, height }} = setupCanvas(canvas);
        if (!width || !height) return;

        const pad = {{ top: 18, bottom: isBottom ? 26 : 14, left: 55, right: 25 }};
        const plotW = width - pad.left - pad.right;
        const plotH = height - pad.top - pad.bottom;

        let yMin = Infinity, yMax = -Infinity;
        lines.forEach(line => {{
          if (!line.visible || !line.data) return;
          for (let i = startIdx; i <= endIdx; i++) {{
            const val = line.data[i];
            if (val < yMin) yMin = val;
            if (val > yMax) yMax = val;
          }}
        }});
        if (!isFinite(yMin)) {{ yMin = -50; yMax = 50; }}
        const margin = Math.max(Math.abs(yMax - yMin) * 0.12, 1.0);
        yMin -= margin;
        yMax += margin;

        const x2px = t => pad.left + ((t - tMin) / (tMax - tMin)) * plotW;
        const y2px = y => pad.top + ((yMax - y) / (yMax - yMin)) * plotH;

        ctx.clearRect(0, 0, width, height);

        // Draw Condition Shading & Trial Banners
        if (stateMulti.showShading) {{
          const conds = MULTITRIAL_DATA.conditions;
          for (let k = 0; k < conds.length; k++) {{
            const t0 = k * 1.5;
            const t1 = (k + 1) * 1.5;
            if (t1 > tMin && t0 < tMax) {{
              const sT = Math.max(t0, tMin);
              const eT = Math.min(t1, tMax);
              const px0 = x2px(sT);
              const px1 = x2px(eT);
              const w = Math.max(1, px1 - px0);

              const cond = conds[k];
              const isV = (cond === 'V');
              const isA = (cond === 'A');

              // Background tint
              ctx.fillStyle = isV ? COLORS.condV : (isA ? COLORS.condA : COLORS.condAV);
              ctx.fillRect(px0, pad.top, w, plotH);

              // Top trial banner
              const bH = 14;
              ctx.fillStyle = isV ? 'rgba(46, 125, 50, 0.15)' : (isA ? 'rgba(214, 39, 40, 0.10)' : 'rgba(148, 103, 189, 0.10)');
              ctx.fillRect(px0, pad.top, w, bH);
              ctx.strokeStyle = isV ? '#2e7d32' : (isA ? '#d62728' : '#94a3b8');
              ctx.lineWidth = 0.8;
              ctx.strokeRect(px0, pad.top, w, bH);

              ctx.fillStyle = isV ? '#166534' : (isA ? '#991b1b' : '#4a044e');
              ctx.font = 'bold 9px -apple-system, sans-serif';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              const lbl = `T${{k + 1}}: [${{cond}}]` + (isV ? ' (SILENCE)' : '');
              if (w > 80) ctx.fillText(lbl, px0 + w / 2, pad.top + bH / 2);
              else if (w > 40) ctx.fillText(`T${{k + 1}} [${{cond}}]`, px0 + w / 2, pad.top + bH / 2);

              // Trial Boundary Vertical Rule
              if (t0 >= tMin && t0 <= tMax) {{
                const bx = x2px(t0);
                ctx.strokeStyle = '#aaaaaa';
                ctx.lineWidth = 0.8;
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(bx, pad.top);
                ctx.lineTo(bx, pad.top + plotH);
                ctx.stroke();
                ctx.setLineDash([]);
              }}
            }}
          }}
        }}

        // Y-axis ticks & grid
        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 1;
        const nYTicks = 4;
        ctx.fillStyle = COLORS.axis;
        ctx.font = '10px -apple-system, sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';

        for (let i = 0; i <= nYTicks; i++) {{
          const val = yMin + (i / nYTicks) * (yMax - yMin);
          const yPx = y2px(val);
          ctx.beginPath();
          ctx.moveTo(pad.left, yPx);
          ctx.lineTo(pad.left + plotW, yPx);
          ctx.stroke();
          ctx.fillText(val.toFixed(0), pad.left - 7, yPx);
        }}

        // X-axis ticks
        const nXTicks = 8;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';

        for (let i = 0; i <= nXTicks; i++) {{
          const tVal = tMin + (i / nXTicks) * (tMax - tMin);
          const xPx = x2px(tVal);
          ctx.strokeStyle = COLORS.grid;
          ctx.beginPath();
          ctx.moveTo(xPx, pad.top);
          ctx.lineTo(xPx, pad.top + plotH);
          ctx.stroke();

          if (isBottom) {{
            ctx.fillStyle = COLORS.axis;
            ctx.fillText(tVal.toFixed(1) + 's', xPx, pad.top + plotH + 5);
          }}
        }}

        // Y Label
        ctx.save();
        ctx.translate(14, pad.top + plotH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#444444';
        ctx.font = '11px -apple-system, sans-serif';
        ctx.fillText('Voltage (μV)', 0, 0);
        ctx.restore();

        // X Label
        if (isBottom) {{
          ctx.fillStyle = '#333333';
          ctx.font = '11px -apple-system, sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          ctx.fillText('Continuous Multi-Trial Timeline within Block (seconds)', pad.left + plotW / 2, height - 2);
        }}

        // Draw Lines
        lines.forEach(line => {{
          if (!line.visible || !line.data) return;

          ctx.strokeStyle = line.color || '#000000';
          ctx.lineWidth = line.lineWidth || 1.3;
          if (line.dash) ctx.setLineDash(line.dash);
          else ctx.setLineDash([]);

          ctx.beginPath();
          let started = false;
          for (let i = startIdx; i <= endIdx; i++) {{
            const xPx = x2px(times[i]);
            const yPx = y2px(line.data[i]);
            if (!started) {{
              ctx.moveTo(xPx, yPx);
              started = true;
            }} else {{
              ctx.lineTo(xPx, yPx);
            }}
          }}
          ctx.stroke();
          ctx.setLineDash([]);
        }});

        canvas._plotBounds = {{ pad, plotW, plotH, tMin, tMax, yMin, yMax, startIdx, endIdx }};
      }}

      // Panel 1: Raw CI
      renderMultiPanel(canvasesMulti[0], [
        {{ name: 'Raw CI', color: '#333333', lineWidth: 1.1, visible: true, data: rawCi }}
      ], false);

      // Panel 2: Clean Hearing
      renderMultiPanel(canvasesMulti[1], [
        {{ name: 'Clean Hearing', color: '#1f77b4', lineWidth: 1.3, visible: true, data: cleanH }}
      ], false);

      // Panel 3: Isolated Noise
      renderMultiPanel(canvasesMulti[2], [
        {{ name: 'Noise: AATS', color: COLORS.aats, lineWidth: 1.2, visible: stateMulti.showAats, data: noiseA }},
        {{ name: 'Noise: FastICA', color: COLORS.ica, lineWidth: 1.2, visible: stateMulti.showIca, data: noiseI }},
        {{ name: 'Noise: SOBI', color: COLORS.sobi, lineWidth: 1.2, visible: stateMulti.showSobi, data: noiseS }},
        {{ name: 'Noise: AATS+SOBI', color: COLORS.aats_sobi, lineWidth: 1.2, visible: stateMulti.showHybrid, data: noiseHyb }},
        {{ name: 'Noise: AATS+ICA', color: COLORS.aats_ica, lineWidth: 1.2, visible: stateMulti.showAatsIca, data: noiseAatsIca }}
      ], false);

      // Panel 4: Denoised Signal
      renderMultiPanel(canvasesMulti[3], [
        {{ name: 'Clean Hearing Ref', color: '#888888', lineWidth: 1.0, dash: [4, 3], visible: stateMulti.showRef, data: cleanH }},
        {{ name: 'Denoised: AATS', color: COLORS.aats, lineWidth: 1.2, visible: stateMulti.showAats, data: denA }},
        {{ name: 'Denoised: FastICA', color: COLORS.ica, lineWidth: 1.2, visible: stateMulti.showIca, data: denI }},
        {{ name: 'Denoised: SOBI', color: COLORS.sobi, lineWidth: 1.2, visible: stateMulti.showSobi, data: denS }},
        {{ name: 'Denoised: AATS+SOBI', color: COLORS.aats_sobi, lineWidth: 1.3, visible: stateMulti.showHybrid, data: denHyb }},
        {{ name: 'Denoised: AATS+ICA', color: COLORS.aats_ica, lineWidth: 1.3, visible: stateMulti.showAatsIca, data: denAatsIca }}
      ], true);
    }}

    // Render Active View
    function renderAll() {{
      updateMetrics();
      if (currentTab === 'demo') {{
        renderTabDemo();
      }} else if (currentTab === 'compare') {{
        renderTabCompare();
      }} else if (currentTab === 'multi') {{
        renderTabMulti();
      }}
    }}

    // Synchronized Hover and Tooltip
    function setupInteractions() {{
      setupGroupHover(canvasesDemo, hoverIndicatorsDemo, 'demo');
      setupGroupHover(canvasesCompare, hoverIndicatorsCompare, 'compare');
      setupGroupHover(canvasesMulti, hoverIndicatorsMulti, 'multi');

      function setupGroupHover(canvases, indicators, tabKey) {{
        canvases.forEach(canvas => {{
          canvas.addEventListener('mousemove', e => {{
            const b = canvas._plotBounds;
            if (!b) return;

            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            if (mouseX < b.pad.left || mouseX > b.pad.left + b.plotW) {{
              hideGroupHover(indicators);
              return;
            }}

            const frac = (mouseX - b.pad.left) / b.plotW;
            const curTime = b.tMin + frac * (b.tMax - b.tMin);

            const isMulti = (tabKey === 'multi');
            const times = isMulti ? MULTITRIAL_DATA.time_s : currentData.times;
            let bestIdx = b.startIdx;
            let minDiff = Infinity;
            for (let i = b.startIdx; i <= b.endIdx; i++) {{
              const diff = Math.abs(times[i] - curTime);
              if (diff < minDiff) {{
                minDiff = diff;
                bestIdx = i;
              }}
            }}

            const targetX = b.pad.left + ((times[bestIdx] - b.tMin) / (b.tMax - b.tMin)) * b.plotW;
            indicators.forEach(hi => {{
              hi.style.display = 'block';
              hi.style.left = `${{targetX}}px`;
            }});

            syncTooltip.style.display = 'block';
            syncTooltip.style.left = `${{Math.min(e.clientX + 15, window.innerWidth - 240)}}px`;
            syncTooltip.style.top = `${{e.clientY - 20}}px`;

            if (tabKey === 'demo') {{
              const chData = currentData.series[selectedChannel];
              const isVisualSegment = times[bestIdx] >= 0;
              const segDesc = isVisualSegment ? '<span style="color:#2e7d32; font-weight:600;">[Visual-Only Silent Segment]</span>' : '<span style="color:#666;">[Pre-Stim Baseline]</span>';
              let html = `<b>${{times[bestIdx].toFixed(1)}} ms</b> (Channel: ${{selectedChannel}}) &bull; ${{segDesc}}<br/>`;
              if (state.clean) html += `<span style="color: ${{COLORS.clean}}">&bull; Clean ERP: ${{chData.clean[bestIdx].toFixed(2)}} μV</span><br/>`;
              if (state.aats) html += `<span style="color: ${{COLORS.aats}}">&bull; AATS Noise: ${{chData.noise_aats[bestIdx].toFixed(1)}} μV</span><br/>`;
              if (state.ica) html += `<span style="color: ${{COLORS.ica}}">&bull; ICA Noise: ${{chData.noise_ica[bestIdx].toFixed(1)}} μV</span><br/>`;
              if (state.sobi) html += `<span style="color: ${{COLORS.sobi}}">&bull; SOBI Noise: ${{chData.noise_sobi[bestIdx].toFixed(1)}} μV</span><br/>`;
              if (state.aats_sobi && chData.noise_aats_sobi) html += `<span style="color: ${{COLORS.aats_sobi}}">&bull; AATS+SOBI Noise: ${{chData.noise_aats_sobi[bestIdx].toFixed(1)}} μV</span><br/>`;
              if (state.aats_ica && chData.noise_aats_ica) html += `<span style="color: ${{COLORS.aats_ica}}">&bull; AATS+ICA Noise: ${{chData.noise_aats_ica[bestIdx].toFixed(1)}} μV</span><br/>`;
              syncTooltip.innerHTML = html;
            }} else if (tabKey === 'compare') {{
              const chData = currentData.series[selectedChannel];
              const isNoise = stateCompare.signalType === 'noise';
              const aSig = isNoise ? chData.noise_aats : chData.dirty_aats;
              const iSig = isNoise ? chData.noise_ica : chData.dirty_ica;
              const sSig = isNoise ? chData.noise_sobi : chData.dirty_sobi;
              const asbSig = isNoise ? (chData.noise_aats_sobi || []) : (chData.dirty_aats_sobi || []);
              const aicSig = isNoise ? (chData.noise_aats_ica || []) : (chData.dirty_aats_ica || []);
              const label = isNoise ? 'Noise' : 'Mixture';
              const isVisualSegment = times[bestIdx] >= 0;
              const segDesc = isVisualSegment ? '<span style="color:#2e7d32; font-weight:600;">[Visual-Only Silent Segment]</span>' : '<span style="color:#666;">[Pre-Stim Baseline]</span>';

              let html = `<b>${{times[bestIdx].toFixed(1)}} ms</b> (Channel: ${{selectedChannel}}) &bull; ${{segDesc}}<br/>`;
              if (stateCompare.showClean) html += `<span style="color: ${{COLORS.clean}}">&bull; Clean GT: ${{chData.clean[bestIdx].toFixed(2)}} μV</span><br/>`;
              if (stateCompare.showAats) html += `<span style="color: ${{COLORS.aats}}">&bull; AATS ${{label}}: ${{aSig[bestIdx].toFixed(1)}} μV</span><br/>`;
              if (stateCompare.showIca) html += `<span style="color: ${{COLORS.ica}}">&bull; ICA ${{label}}: ${{iSig[bestIdx].toFixed(1)}} μV</span><br/>`;
              if (stateCompare.showSobi) html += `<span style="color: ${{COLORS.sobi}}">&bull; SOBI ${{label}}: ${{sSig[bestIdx].toFixed(1)}} μV</span><br/>`;
              if (stateCompare.showAatsSobi && asbSig[bestIdx] !== undefined) html += `<span style="color: ${{COLORS.aats_sobi}}">&bull; AATS+SOBI ${{label}}: ${{asbSig[bestIdx].toFixed(1)}} μV</span><br/>`;
              if (stateCompare.showAatsIca && aicSig[bestIdx] !== undefined) html += `<span style="color: ${{COLORS.aats_ica}}">&bull; AATS+ICA ${{label}}: ${{aicSig[bestIdx].toFixed(1)}} μV</span><br/>`;
              syncTooltip.innerHTML = html;
            }} else if (tabKey === 'multi') {{
              const t = times[bestIdx];
              const trialK = Math.min(7, Math.floor(t / 1.5));
              const cond = MULTITRIAL_DATA.conditions[trialK];
              const condName = (cond === 'V' ? 'Visual-Only (Silent Baseline)' : (cond === 'A' ? 'Audio-only Speech' : 'AudioVisual Speech'));
              const condColor = (cond === 'V' ? '#2e7d32' : (cond === 'A' ? '#d62728' : '#7c3aed'));

              let html = `<b>t = ${{t.toFixed(2)}}s</b> &bull; <span style="color:${{condColor}}; font-weight:bold;">Trial ${{trialK + 1}} [${{cond}}]</span> (${{condName}})<br/>`;
              html += `<span style="color:#333;">&bull; Raw CI: ${{MULTITRIAL_DATA.raw_ci[bestIdx].toFixed(1)}} μV</span><br/>`;
              html += `<span style="color:#1f77b4;">&bull; Clean Hearing: ${{MULTITRIAL_DATA.clean_hearing[bestIdx].toFixed(1)}} μV</span><br/>`;
              if (stateMulti.showAats) html += `<span style="color:${{COLORS.aats}}">&bull; AATS Denoised: ${{MULTITRIAL_DATA.denoised_aats[bestIdx].toFixed(1)}} μV</span><br/>`;
              if (stateMulti.showIca) html += `<span style="color:${{COLORS.ica}}">&bull; FastICA Denoised: ${{MULTITRIAL_DATA.denoised_ica[bestIdx].toFixed(1)}} μV</span><br/>`;
              if (stateMulti.showSobi) html += `<span style="color:${{COLORS.sobi}}">&bull; SOBI Denoised: ${{MULTITRIAL_DATA.denoised_sobi[bestIdx].toFixed(1)}} μV</span><br/>`;
              if (stateMulti.showHybrid) html += `<span style="color:${{COLORS.aats_sobi}}">&bull; AATS+SOBI: ${{MULTITRIAL_DATA.denoised_aats_sobi[bestIdx].toFixed(1)}} μV</span><br/>`;
              if (stateMulti.showAatsIca && MULTITRIAL_DATA.denoised_aats_ica) html += `<span style="color:${{COLORS.aats_ica}}">&bull; AATS+ICA: ${{MULTITRIAL_DATA.denoised_aats_ica[bestIdx].toFixed(1)}} μV</span><br/>`;
              syncTooltip.innerHTML = html;
            }}
          }});

          canvas.addEventListener('mouseleave', () => hideGroupHover(indicators));
        }});
      }}

      function hideGroupHover(indicators) {{
        indicators.forEach(hi => hi.style.display = 'none');
        syncTooltip.style.display = 'none';
      }}
    }}

    // Checkbox and Presets Listeners
    function setupControls() {{
      function switchTab(tab) {{
        currentTab = tab;
        tabBtnDemo.classList.toggle('active', tab === 'demo');
        tabBtnCompare.classList.toggle('active', tab === 'compare');
        tabBtnMulti.classList.toggle('active', tab === 'multi');

        sidebarDemo.classList.toggle('hidden', tab !== 'demo');
        sidebarCompare.classList.toggle('hidden', tab !== 'compare');
        sidebarMulti.classList.toggle('hidden', tab !== 'multi');

        stageDemo.classList.toggle('hidden', tab !== 'demo');
        stageCompare.classList.toggle('hidden', tab !== 'compare');
        stageMulti.classList.toggle('hidden', tab !== 'multi');

        if (tab === 'multi') {{
          zoomSelect.classList.add('hidden');
          zoomSelectMulti.classList.remove('hidden');
          chSelect.disabled = true;
          chSelectLabel.style.opacity = '0.5';
          chSelect.title = 'Multi-trial continuous recording is recorded at Cz (Vertex)';
        }} else {{
          zoomSelect.classList.remove('hidden');
          zoomSelectMulti.classList.add('hidden');
          chSelect.disabled = false;
          chSelectLabel.style.opacity = '1';
          chSelect.title = '';
        }}

        renderAll();
      }}

      tabBtnDemo.addEventListener('click', () => switchTab('demo'));
      tabBtnCompare.addEventListener('click', () => switchTab('compare'));
      tabBtnMulti.addEventListener('click', () => switchTab('multi'));

      // Tab 1 Checkboxes
      const chkMapDemo = {{
        chk_aats: 'aats',
        chk_ica: 'ica',
        chk_sobi: 'sobi',
        chk_aats_sobi: 'aats_sobi',
        chk_aats_ica: 'aats_ica',
        chk_clean: 'clean',
        chk_visual: 'visual',
        chk_highlight_visual: 'highlightVisual',
        chk_onset: 'onset'
      }};

      Object.entries(chkMapDemo).forEach(([id, key]) => {{
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', () => {{ state[key] = el.checked; renderAll(); }});
      }});

      // Tab 2 Checkboxes
      const chkMapCompare = {{
        chkC_clean: 'showClean',
        chkC_aats: 'showAats',
        chkC_ica: 'showIca',
        chkC_sobi: 'showSobi',
        chkC_aats_sobi: 'showAatsSobi',
        chkC_aats_ica: 'showAatsIca',
        chkC_overlayClean: 'overlayRef',
        chkC_onset: 'onset'
      }};

      Object.entries(chkMapCompare).forEach(([id, key]) => {{
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', () => {{ stateCompare[key] = el.checked; renderAll(); }});
      }});

      // Tab 3 Checkboxes (Multi-Trial)
      const chkMapMulti = {{
        chkM_aats: 'showAats',
        chkM_ica: 'showIca',
        chkM_sobi: 'showSobi',
        chkM_hybrid: 'showHybrid',
        chkM_aats_ica: 'showAatsIca',
        chkM_ref: 'showRef',
        chkM_shading: 'showShading'
      }};

      Object.entries(chkMapMulti).forEach(([id, key]) => {{
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', () => {{ stateMulti[key] = el.checked; renderAll(); }});
      }});

      // Multi-trial Demean toggle
      document.getElementById('btnMultiDemean').addEventListener('click', () => {{
        document.getElementById('btnMultiDemean').classList.add('active');
        document.getElementById('btnMultiRawDC').classList.remove('active');
        stateMulti.demean = true;
        renderAll();
      }});
      document.getElementById('btnMultiRawDC').addEventListener('click', () => {{
        document.getElementById('btnMultiRawDC').classList.add('active');
        document.getElementById('btnMultiDemean').classList.remove('active');
        stateMulti.demean = false;
        renderAll();
      }});

      zoomSelectMulti.addEventListener('change', () => {{
        stateMulti.timeWindow = zoomSelectMulti.value;
        renderAll();
      }});

      // Tab 2 Segmented Controls (Signal Type)
      document.getElementById('btnSignalNoise').addEventListener('click', () => {{
        document.getElementById('btnSignalNoise').classList.add('active');
        document.getElementById('btnSignalDirty').classList.remove('active');
        stateCompare.signalType = 'noise';
        renderAll();
      }});
      document.getElementById('btnSignalDirty').addEventListener('click', () => {{
        document.getElementById('btnSignalDirty').classList.add('active');
        document.getElementById('btnSignalNoise').classList.remove('active');
        stateCompare.signalType = 'dirty';
        renderAll();
      }});

      // Tab 2 Segmented Controls (Scale Mode)
      const scaleBtns = [
        document.getElementById('btnScaleIndep'),
        document.getElementById('btnScaleUnified'),
        document.getElementById('btnScaleNorm')
      ];
      scaleBtns.forEach(btn => {{
        btn.addEventListener('click', () => {{
          scaleBtns.forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          stateCompare.scaleMode = btn.getAttribute('data-scale');
          renderAll();
        }});
      }});

      // Tab 2 Segmented Controls (Layout)
      document.getElementById('btnLayoutStacked').addEventListener('click', () => {{
        document.getElementById('btnLayoutStacked').classList.add('active');
        document.getElementById('btnLayoutGrid').classList.remove('active');
        stateCompare.layout = 'stacked';
        stageCompare.classList.remove('stage-grid-2x2');
        renderAll();
      }});
      document.getElementById('btnLayoutGrid').addEventListener('click', () => {{
        document.getElementById('btnLayoutGrid').classList.add('active');
        document.getElementById('btnLayoutStacked').classList.remove('active');
        stateCompare.layout = 'grid';
        stageCompare.classList.add('stage-grid-2x2');
        renderAll();
      }});

      // Preset Buttons
      document.querySelectorAll('.btn-preset').forEach(btn => {{
        btn.addEventListener('click', () => {{
          const p = btn.getAttribute('data-preset');
          if (p === 'all') {{
            state.aats = true; state.ica = true; state.sobi = true; state.aats_sobi = true; state.aats_ica = true;
            state.clean = true; state.visual = true; state.onset = true;
          }} else if (p === 'aats') {{
            state.aats = true; state.ica = false; state.sobi = false; state.aats_sobi = false; state.aats_ica = false;
            state.clean = true; state.visual = false; state.onset = true;
          }} else if (p === 'ica') {{
            state.aats = false; state.ica = true; state.sobi = false; state.aats_sobi = false; state.aats_ica = false;
            state.clean = true; state.visual = false; state.onset = true;
          }} else if (p === 'sobi') {{
            state.aats = false; state.ica = false; state.sobi = true; state.aats_sobi = false; state.aats_ica = false;
            state.clean = true; state.visual = false; state.onset = true;
          }} else if (p === 'aats_sobi') {{
            state.aats = false; state.ica = false; state.sobi = false; state.aats_sobi = true; state.aats_ica = false;
            state.clean = true; state.visual = false; state.onset = true;
          }} else if (p === 'aats_ica') {{
            state.aats = false; state.ica = false; state.sobi = false; state.aats_sobi = false; state.aats_ica = true;
            state.clean = true; state.visual = false; state.onset = true;
          }} else if (p === 'clean_synth') {{
            state.aats = true; state.ica = true; state.sobi = true; state.aats_sobi = true; state.aats_ica = true;
            state.clean = true; state.visual = false; state.onset = true;
          }} else if (p === 'noise') {{
            state.aats = true; state.ica = true; state.sobi = true; state.aats_sobi = true; state.aats_ica = true;
            state.clean = false; state.visual = false; state.onset = true;
          }}

          Object.entries(chkMapDemo).forEach(([id, key]) => {{
            const el = document.getElementById(id);
            if (el) el.checked = state[key];
          }});

          renderAll();
        }});
      }});

      chSelect.addEventListener('change', () => {{
        selectedChannel = chSelect.value;
        chBadge.textContent = selectedChannel;
        if (chBadgeC) chBadgeC.textContent = selectedChannel;
        renderAll();
      }});

      zoomSelect.addEventListener('change', () => {{
        timeWindow = zoomSelect.value;
        renderAll();
      }});

      // Download CSV
      document.getElementById('btnExportCSV').addEventListener('click', () => {{
        if (currentTab === 'multi') {{
          const times_m = MULTITRIAL_DATA.time_s;
          let csv = "time_s,trial_idx,condition,raw_ci_uV,clean_hearing_uV,noise_aats_uV,denoised_aats_uV,noise_ica_uV,denoised_ica_uV,noise_sobi_uV,denoised_sobi_uV,noise_aats_sobi_uV,denoised_aats_sobi_uV,noise_aats_ica_uV,denoised_aats_ica_uV\\n";
          for (let i = 0; i < times_m.length; i++) {{
            const t = times_m[i];
            const t_idx = Math.min(7, Math.floor(t / 1.5));
            const c_lbl = MULTITRIAL_DATA.conditions[t_idx];
            const nai = (MULTITRIAL_DATA.noise_aats_ica && MULTITRIAL_DATA.noise_aats_ica[i] !== undefined) ? MULTITRIAL_DATA.noise_aats_ica[i] : 0;
            const dai = (MULTITRIAL_DATA.denoised_aats_ica && MULTITRIAL_DATA.denoised_aats_ica[i] !== undefined) ? MULTITRIAL_DATA.denoised_aats_ica[i] : 0;
            csv += `${{t}},${{t_idx+1}},${{c_lbl}},${{MULTITRIAL_DATA.raw_ci[i]}},${{MULTITRIAL_DATA.clean_hearing[i]}},${{MULTITRIAL_DATA.noise_aats[i]}},${{MULTITRIAL_DATA.denoised_aats[i]}},${{MULTITRIAL_DATA.noise_ica[i]}},${{MULTITRIAL_DATA.denoised_ica[i]}},${{MULTITRIAL_DATA.noise_sobi[i]}},${{MULTITRIAL_DATA.denoised_sobi[i]}},${{MULTITRIAL_DATA.noise_aats_sobi[i]}},${{MULTITRIAL_DATA.denoised_aats_sobi[i]}},${{nai}},${{dai}}\\n`;
          }}
          const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `multitrial_block_cz.csv`;
          a.click();
          URL.revokeObjectURL(url);
          return;
        }}

        const chData = currentData.series[selectedChannel];
        if (!chData) return;
        const times = currentData.times;
        let csv = "time_ms,clean_uV,noise_aats_uV,noise_ica_uV,noise_sobi_uV,noise_aats_sobi_uV,noise_aats_ica_uV,dirty_aats_uV,dirty_ica_uV,dirty_sobi_uV,dirty_aats_sobi_uV,dirty_aats_ica_uV,visual_only_segment\\n";
        for (let i = 0; i < times.length; i++) {{
          const v_flag = times[i] >= 0 ? 1 : 0;
          const nas = (chData.noise_aats_sobi && chData.noise_aats_sobi[i] !== undefined) ? chData.noise_aats_sobi[i] : 0;
          const nai = (chData.noise_aats_ica && chData.noise_aats_ica[i] !== undefined) ? chData.noise_aats_ica[i] : 0;
          const das = (chData.dirty_aats_sobi && chData.dirty_aats_sobi[i] !== undefined) ? chData.dirty_aats_sobi[i] : 0;
          const dai = (chData.dirty_aats_ica && chData.dirty_aats_ica[i] !== undefined) ? chData.dirty_aats_ica[i] : 0;
          csv += `${{times[i]}},${{chData.clean[i]}},${{chData.noise_aats[i]}},${{chData.noise_ica[i]}},${{chData.noise_sobi[i]}},${{nas}},${{nai}},${{chData.dirty_aats[i]}},${{chData.dirty_ica[i]}},${{chData.dirty_sobi[i]}},${{das}},${{dai}},${{v_flag}}\\n`;
        }}
        const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `single_pair_${{selectedChannel}}_epochs.csv`;
        a.click();
        URL.revokeObjectURL(url);
      }});

      // Export Combined PNG
      document.getElementById('btnExportPNG').addEventListener('click', () => {{
        let activeCanvases = canvasesDemo;
        if (currentTab === 'compare') {{
          activeCanvases = canvasesCompare.filter(c => c.closest('.panel-card').style.display !== 'none');
        }} else if (currentTab === 'multi') {{
          activeCanvases = canvasesMulti;
        }}
        if (!activeCanvases.length) return;

        const exportCanvas = document.createElement('canvas');
        const dpr = window.devicePixelRatio || 1;
        const w = activeCanvases[0].width;
        const hEach = activeCanvases[0].height;
        const headerH = 40 * dpr;
        exportCanvas.width = w;
        exportCanvas.height = (hEach + headerH) * activeCanvases.length + 50 * dpr;
        const eCtx = exportCanvas.getContext('2d');

        eCtx.fillStyle = '#ffffff';
        eCtx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);

        eCtx.fillStyle = '#111111';
        eCtx.font = `bold ${{14 * dpr}}px sans-serif`;
        let viewTitle = '4-Panel Pairing Demo';
        if (currentTab === 'compare') viewTitle = 'Method Waveform Shape Comparison';
        else if (currentTab === 'multi') viewTitle = 'Multi-Trial Block (8 Trials • 12.0s Timeline)';

        eCtx.fillText(`${{viewTitle}}: Subject ${{currentData.ci_subject}} (CI) × ${{currentData.hearing_subject}} (Hearing), Channel: ${{selectedChannel}}`, 20 * dpr, 30 * dpr);

        let yOffset = 50 * dpr;
        activeCanvases.forEach((c) => {{
          eCtx.drawImage(c, 0, yOffset);
          yOffset += hEach + 15 * dpr;
        }});

        const link = document.createElement('a');
        link.download = `pairing_${{currentTab}}_${{currentData.ci_subject}}_${{currentData.hearing_subject}}_${{selectedChannel}}.png`;
        link.href = exportCanvas.toDataURL('image/png');
        link.click();
      }});

      // Drag and drop / file upload
      const dropzone = document.getElementById('dropzone');
      const fileInput = document.getElementById('fileInput');

      dropzone.addEventListener('click', () => fileInput.click());
      dropzone.addEventListener('dragover', e => {{ e.preventDefault(); dropzone.classList.add('dragover'); }});
      dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
      dropzone.addEventListener('drop', e => {{
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
      }});
      fileInput.addEventListener('change', () => {{
        if (fileInput.files.length) handleFile(fileInput.files[0]);
      }});

      function handleFile(file) {{
        const reader = new FileReader();
        reader.onload = ev => {{
          const text = ev.target.result;
          if (file.name.endsWith('.json')) {{
            try {{
              const json = JSON.parse(text);
              if (json.channels && json.times && json.series) {{
                currentData = json;
                initChannels();
                renderAll();
                datasetInfo.textContent = `File: ${{file.name}} (CI: ${{json.ci_subject || 'N/A'}}, Perm: ${{json.permutation || 1}})`;
              }}
            }} catch(err) {{
              alert('Failed to parse JSON file.');
            }}
          }} else if (file.name.endsWith('.csv')) {{
            parseCustomCSV(text, file.name);
          }}
        }};
        reader.readAsText(file);
      }}

      function parseCustomCSV(text, filename) {{
        const lines = text.trim().split(/\\r?\\n/);
        if (lines.length < 2) return;
        const headers = lines[0].split(',').map(s => s.trim().toLowerCase());
        const timeIdx = headers.findIndex(h => h.includes('time'));
        if (timeIdx === -1) {{
          alert('CSV must contain a time column (e.g. time_ms or time).');
          return;
        }}

        const times = [];
        const clean = [];
        const aats = [];
        const ica = [];
        const sobi = [];
        const daats = [];
        const dica = [];
        const dsobi = [];

        for (let i = 1; i < lines.length; i++) {{
          const parts = lines[i].split(',').map(Number);
          if (parts.length < headers.length) continue;
          times.push(parts[timeIdx]);
          headers.forEach((h, col) => {{
            const val = parts[col];
            if (h.includes('clean')) clean.push(val);
            else if (h.includes('noise_aats') || (h.includes('aats') && !h.includes('dirty'))) aats.push(val);
            else if (h.includes('noise_ica') || (h.includes('ica') && !h.includes('dirty'))) ica.push(val);
            else if (h.includes('noise_sobi') || (h.includes('sobi') && !h.includes('dirty'))) sobi.push(val);
            else if (h.includes('dirty_aats')) daats.push(val);
            else if (h.includes('dirty_ica')) dica.push(val);
            else if (h.includes('dirty_sobi')) dsobi.push(val);
          }});
        }}

        const customCh = 'Imported';
        currentData = {{
          ci_subject: 'Custom',
          hearing_subject: 'Custom',
          permutation: 1,
          channels: [customCh],
          times: times,
          series: {{
            [customCh]: {{
              clean: clean.length ? clean : new Array(times.length).fill(0),
              noise_aats: aats.length ? aats : new Array(times.length).fill(0),
              noise_ica: ica.length ? ica : new Array(times.length).fill(0),
              noise_sobi: sobi.length ? sobi : new Array(times.length).fill(0),
              dirty_aats: daats.length ? daats : (clean.length && aats.length ? clean.map((c, i) => c + aats[i]) : new Array(times.length).fill(0)),
              dirty_ica: dica.length ? dica : (clean.length && ica.length ? clean.map((c, i) => c + ica[i]) : new Array(times.length).fill(0)),
              dirty_sobi: dsobi.length ? dsobi : (clean.length && sobi.length ? clean.map((c, i) => c + sobi[i]) : new Array(times.length).fill(0))
            }}
          }}
        }};
        initChannels();
        renderAll();
        datasetInfo.textContent = `Loaded CSV: ${{filename}}`;
      }}

      window.addEventListener('resize', () => renderAll());
    }}

    // App Initialization
    window.addEventListener('DOMContentLoaded', () => {{
      initChannels();
      setupControls();
      setupInteractions();
      renderAll();
    }});
  </script>
</body>
</html>
"""

    out_paths = [
        "/Users/evanywliang/Documents/cidenoise/ci_denoise/visualization/single_pair_viewer.html"
    ]

    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(html_content)
        print(f"Generated standalone Single Pair Viewer -> {p} ({len(html_content)} bytes)")

if __name__ == "__main__":
    build()
