"""
reconstruct.py - export a zarr recording to EEGLAB .set format.

Loads one recording from a zarr (either eeg_raw.zarr or eeg_clean.zarr),
builds an MNE Raw object with channel positions + annotations + every
metadata field we persisted, and writes a SINGLE-FILE .set (data inline)
plus a sidecar <name>_meta.json that carries the metadata that EEGLAB's
native schema cannot hold verbatim (participant ID, per-channel norm
stats, ICA scores, notch list, provenance).

NOTE: an earlier version split the output into .set (metadata) + .fdt
(binary data) via a scipy loadmat/savemat round-trip. That corrupted
`EEG.event` and `EEG.chanlocs` into cell arrays (because scipy can't
re-emit `mat_struct` Python objects as MATLAB struct arrays), causing
`eeg_checkset` to fail in EEGLAB with "Invalid input argument of type
'cell'". We now ship the eeglabio output verbatim (struct arrays
preserved) and never touch the .set after eeglabio writes it.

Usage:
  python reconstruct.py --zarr eeg_raw.zarr     --name noise0_perm1 --stream data           --out /tmp/noise0_perm1_raw.set
  python reconstruct.py --zarr eeg_clean.zarr   --name noise0_perm1 --stream broadband_data --out /tmp/noise0_perm1_clean_bb.set
  python reconstruct.py --zarr eeg_clean.zarr   --name noise0_perm1 --stream erp_data       --out /tmp/noise0_perm1_clean_erp.set

  # Round-trip test (writes a .set, reads it back, compares):
  python reconstruct.py --zarr eeg_raw.zarr --name noise0_perm1 --stream data --out /tmp/rt.set --roundtrip

  # Cropped export for size-limited testing:
  python reconstruct.py --zarr eeg_n2n.zarr --name noise0_perm1 --t-end 150 --out /tmp/sample.set --roundtrip

The .set is directly openable in EEGLAB with `pop_loadset`. The sidecar
JSON can be loaded in Python with `json.load(open(meta_json_path))` to
recover participant ID, norm stats, pipeline provenance, etc.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import zarr
import mne


# All the zarr attribute keys that we want to carry through, per domain.
# Keys not present in a given recording are skipped silently. Everything
# here gets serialised into the sidecar JSON; a compact version also
# goes into raw.info['description'] so it survives in the .set file's
# 'comments' field.
METADATA_KEYS = [
    # shared
    'participant_id', 'perm', 'source_file',
    'interpolated_channels', 'complete',
    'annotations_description',
    # raw-zarr only
    'filter_lfreq', 'filter_hfreq',
    # clean-zarr only
    'broadband_sfreq', 'erp_sfreq',
    'notch_freqs', 'ica_excluded', 'ica_scores',
    'pipeline_complete', 'ica_error',
    # classical_baseline ICA-specific (eeg_ica.zarr / eeg_n2n.zarr)
    'ica_method', 'ica_n_components', 'ica_decim',
    'ica_max_iter', 'ica_random_state', 'n_ica_excluded',
    'preprocessing_notebook',
    # N2N-specific (eeg_n2n.zarr)
    'n2n_n_events_processed', 'n2n_inference_taper', 'n2n_version',
]

# Per-channel arrays we want in the sidecar JSON (small).
EXTRA_ARRAY_KEYS = [
    'bad_channel_mask',
    'ica_hf_ratio', 'ica_spatial_kurt', 'ica_exclude',
]
# bad_segment_mask is handled separately as RLE -- it's per-sample bool over
# the whole recording (12.5M entries) so we don't want it as a raw list in JSON.

# Zarrs store data already in V — `mne.io.Raw.get_data()` returns SI units
# (V) and that's what `prepare_data.py` writes. `mne.export.export_raw(fmt='eeglab')`
# converts V → µV on the way out, and `read_raw_eeglab` converts µV → V on the
# way in, so passing V through unchanged round-trips exactly (verified
# 2026-05-13 by writing then reading a real .set with no scaling: max abs and
# every sample matched to float64 precision).


def _auto_pick_stream(group):
    """If the user did not specify --stream, pick the first one present."""
    for key in ('data', 'broadband_data', 'erp_data'):
        if key in group:
            return key
    raise ValueError(f'no known stream array in group {group.path}; '
                     f'have {list(group.array_keys())}')


def _sfreq_for_stream(group, stream):
    if stream == 'data':
        return float(group.attrs['sfreq'])
    if stream == 'broadband_data':
        return float(group.attrs['broadband_sfreq'])
    if stream == 'erp_data':
        return float(group.attrs['erp_sfreq'])
    raise ValueError(f'unknown stream {stream}')


def _gather_metadata(group):
    """Collect everything useful from a zarr group into a JSON-safe dict."""
    meta = {}
    for k in METADATA_KEYS:
        if k in group.attrs:
            v = group.attrs[k]
            # annotations_description is a JSON string; expand it
            if k == 'annotations_description' and isinstance(v, str):
                try:
                    v = json.loads(v)
                except Exception:
                    pass
            # zarr attrs come back as python types already; just make sure
            # numpy arrays are lists for JSON
            if isinstance(v, np.ndarray):
                v = v.tolist()
            meta[k] = v

    # Per-channel norm stats (arrays, not attrs)
    if 'norm_mean' in group:
        meta['norm_mean'] = np.array(group['norm_mean']).reshape(-1).tolist()
    if 'norm_std' in group:
        meta['norm_std']  = np.array(group['norm_std']).reshape(-1).tolist()
    # Extra small arrays (bad-channel mask, ICA diagnostics)
    for k in EXTRA_ARRAY_KEYS:
        if k in group:
            arr = np.array(group[k])
            if arr.dtype.kind == 'b':
                meta[k] = arr.astype(int).tolist()
            else:
                meta[k] = arr.tolist()
    # bad_segment_mask: per-sample bool → RLE'd into (start, end) sample-index
    # ranges. Much more compact in JSON and easy to read back.
    if 'bad_segment_mask' in group:
        m = np.array(group['bad_segment_mask']).astype(bool)
        if m.any():
            # Find edges: True runs
            d = np.diff(m.astype(np.int8))
            starts = np.where(d == 1)[0] + 1
            ends   = np.where(d == -1)[0] + 1
            if m[0]:  starts = np.concatenate(([0], starts))
            if m[-1]: ends   = np.concatenate((ends, [len(m)]))
            meta['bad_segment_ranges_samples'] = [
                [int(s), int(e)] for s, e in zip(starts, ends)
            ]
        else:
            meta['bad_segment_ranges_samples'] = []
        meta['bad_segment_n_samples_total'] = int(m.size)
        meta['bad_segment_n_samples_bad']   = int(m.sum())
    return meta


def build_raw_from_zarr(zarr_path, domain, name, stream=None,
                        montage_name='standard_1020',
                        t_end_seconds=None):
    """Load a zarr recording into an mne.io.RawArray.

    Args:
        t_end_seconds: if not None, crop the data and annotations to
            [0, t_end_seconds). Annotations whose onset is past the cut
            get dropped; annotations whose end runs past the cut get
            their duration clamped. Useful for producing a small
            ~200 MB EEGLAB test file from a full recording.

    Returns:
        raw  -- mne.io.RawArray with annotations set and montage applied
        meta -- the full sidecar metadata dict for JSON dump
        stream -- which stream was actually loaded
    """
    store = zarr.open(str(zarr_path), mode='r')
    group = store[domain][name]
    if stream is None:
        stream = _auto_pick_stream(group)
    if stream not in group:
        raise ValueError(f'stream {stream!r} not present in {domain}/{name}; '
                         f'available: {list(group.array_keys())}')

    data = np.array(group[stream]).astype(np.float64)
    sfreq = _sfreq_for_stream(group, stream)
    ch_names = list(group.attrs['ch_names'])
    ch_types = list(group.attrs.get('ch_types', ['eeg'] * data.shape[0]))
    assert data.shape[0] == len(ch_names), \
        f'channel count mismatch: data {data.shape[0]} vs ch_names {len(ch_names)}'

    # Optional crop -- kept simple: clip data to the first n_keep samples.
    if t_end_seconds is not None:
        n_keep = int(round(float(t_end_seconds) * float(sfreq)))
        if n_keep <= 0 or n_keep > data.shape[1]:
            raise ValueError(
                f't_end_seconds={t_end_seconds} maps to n_keep={n_keep} '
                f'which is outside [1, {data.shape[1]}]'
            )
        data = data[:, :n_keep]

    # Annotations
    onsets = np.array(group['annotations_onset']) \
        if 'annotations_onset' in group else np.array([])
    durations = np.array(group['annotations_duration']) \
        if 'annotations_duration' in group else np.zeros_like(onsets)
    descs_raw = group.attrs.get('annotations_description', '[]')
    descs = json.loads(descs_raw) if isinstance(descs_raw, str) else list(descs_raw)
    # some recordings have a mismatch between descs and onsets if cropped;
    # truncate to the shorter of the two so MNE doesn't complain
    n = min(len(onsets), len(descs))
    onsets = onsets[:n]; durations = durations[:n]; descs = descs[:n]

    # If the data was cropped, drop annotations past the cut and clamp
    # any annotation whose end extends past the cut.
    if t_end_seconds is not None and n > 0:
        keep = onsets < float(t_end_seconds)
        onsets    = onsets[keep]
        durations = durations[keep]
        descs     = [d for d, k in zip(descs, keep) if k]
        end = onsets + durations
        clamp = end > float(t_end_seconds)
        if clamp.any():
            durations = np.where(clamp, float(t_end_seconds) - onsets, durations)
        n = len(onsets)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq,
                           ch_types=ch_types, verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)

    # Build the annotation set: stimulus events + bad-segment annotations.
    # MNE convention is that any annotation whose description starts with
    # 'BAD_' is treated as a bad segment (skipped by epoching, masked in
    # filtering). EEGLAB picks these up as boundary events when reading the
    # round-tripped .set.
    onset_list  = list(map(float, onsets))
    dur_list    = list(map(float, durations))
    desc_list   = [str(d) for d in descs]

    if 'bad_segment_mask' in group:
        m = np.array(group['bad_segment_mask']).astype(bool)
        # Match the crop applied to `data` above so BAD segments don't
        # reference samples past the end of the (cropped) Raw.
        if t_end_seconds is not None:
            m = m[:data.shape[1]]
        if m.any():
            d = np.diff(m.astype(np.int8))
            starts = np.where(d == 1)[0] + 1
            ends   = np.where(d == -1)[0] + 1
            if m[0]:  starts = np.concatenate(([0], starts))
            if m[-1]: ends   = np.concatenate((ends, [len(m)]))
            for s, e in zip(starts, ends):
                onset_list.append(float(s) / float(sfreq))
                dur_list.append(float(e - s) / float(sfreq))
                desc_list.append('BAD_segment')

    if onset_list:
        ann = mne.Annotations(onset=onset_list, duration=dur_list,
                              description=desc_list)
        raw.set_annotations(ann, verbose=False)

    # Mark bad channels in raw.info so they round-trip into the .set file.
    if 'bad_channel_mask' in group:
        bad_mask = np.array(group['bad_channel_mask']).astype(bool)
        bads = [ch for ch, is_bad in zip(ch_names, bad_mask) if is_bad]
        if bads:
            raw.info['bads'] = list(bads)

    # Standard montage for EEGLAB topomaps. on_missing='warn' because M1/M2
    # aren't in standard_1020 but we'll still have valid positions for the
    # 19 scalp electrodes.
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        raw.set_montage(montage, on_missing='warn', match_case=False,
                        verbose=False)
    except Exception as e:
        print(f'  warning: could not set montage {montage_name}: {e}',
              flush=True)

    # Gather everything else into a sidecar dict
    meta = _gather_metadata(group)
    meta['__reconstruct_info__'] = {
        'zarr_path': str(zarr_path),
        'domain':    domain,
        'name':      name,
        'stream':    stream,
        'sfreq':     float(sfreq),
        'n_channels': int(data.shape[0]),
        'n_samples':  int(data.shape[1]),
        'n_events':   int(n),
    }

    # Pack a compact version into raw.info['description'] — EEGLAB shows
    # this as the dataset's 'comments' field, so the metadata is visible
    # even without the sidecar.
    compact = {k: meta[k] for k in (
        'participant_id', 'perm', 'source_file',
        'broadband_sfreq', 'erp_sfreq',
        'notch_freqs', 'ica_excluded',
        'filter_lfreq', 'filter_hfreq',
        'interpolated_channels',
    ) if k in meta}
    compact['__stream__'] = stream
    try:
        with raw.info._unlock():
            raw.info['description'] = json.dumps(compact, separators=(',', ':'))
    except Exception:
        raw.info['description'] = json.dumps(compact, separators=(',', ':'))

    return raw, meta, stream


def export_to_set(zarr_path, domain, name, out_path, stream=None,
                  write_sidecar=True, t_end_seconds=None):
    """Write a single-file .set (data inline) plus sidecar JSON.

    Returns (set_path, meta_path, meta).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() != '.set':
        out_path = out_path.with_suffix('.set')

    raw, meta, stream = build_raw_from_zarr(zarr_path, domain, name,
                                            stream=stream,
                                            t_end_seconds=t_end_seconds)

    # eeglabio writes a single-file .set with `event` and `chanlocs` as
    # proper MATLAB struct arrays (via numpy.rec.fromarrays + scipy savemat).
    # Do NOT touch the file after this -- a loadmat/savemat round-trip with
    # struct_as_record=False corrupts the struct arrays into cell arrays
    # and breaks pop_loadset in EEGLAB.
    mne.export.export_raw(str(out_path), raw, fmt='eeglab',
                          overwrite=True, verbose=False)

    if write_sidecar:
        meta_path = out_path.with_name(out_path.stem + '_meta.json')
        with open(meta_path, 'w') as fh:
            json.dump(meta, fh, indent=2, default=str)
    else:
        meta_path = None

    return out_path, meta_path, meta


# -------------------------------- round-trip test -----------------------------
def _rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-40))


def roundtrip_check(zarr_path, domain, name, out_path, stream=None,
                    tol_relative_rms=1e-4, t_end_seconds=None):
    """Write zarr -> .set, then verify the .set is EEGLAB-valid.

    This goes deeper than the previous version, which only re-read with
    `mne.io.read_raw_eeglab`. MNE's reader uses scipy with
    `struct_as_record=False`, which silently accepts struct fields that
    EEGLAB itself rejects (e.g. cell arrays where struct arrays are
    expected -- the exact failure mode that broke pop_loadset on the
    professor's machine).

    Two layers of verification:

    1. Structural (scipy.io.loadmat with struct_as_record=True): the .set
       must contain `data` (numeric), `event` (struct array with at least
       type/latency/duration), `chanlocs` (struct array with at least
       labels/X/Y/Z), and matching srate/nbchan/pnts. This catches the
       cell-vs-struct corruption that EEGLAB chokes on.
    2. Numeric (mne.io.read_raw_eeglab): data round-trips within float32
       precision (V on both sides; eeglabio handles V<->µV internally).

    Returns dict {ok: bool, errors: [...], ...}.
    """
    # Zarr stores data already in V; MNE returns V on read. No scaling.
    store = zarr.open(str(zarr_path), mode='r')
    group = store[domain][name]
    if stream is None:
        stream = _auto_pick_stream(group)
    ref_data = np.array(group[stream]).astype(np.float64)
    if t_end_seconds is not None:
        n_keep = int(round(float(t_end_seconds) * _sfreq_for_stream(group, stream)))
        ref_data = ref_data[:, :n_keep]
    ref_sfreq = _sfreq_for_stream(group, stream)
    ref_chn = list(group.attrs['ch_names'])

    # Export
    set_path, meta_path, _ = export_to_set(zarr_path, domain, name,
                                           out_path, stream=stream,
                                           t_end_seconds=t_end_seconds)

    details = {
        'ref_shape':  tuple(ref_data.shape),
        'ref_sfreq':  ref_sfreq,
        'set_path':   str(set_path),
        'meta_path':  str(meta_path),
        'set_bytes':  os.path.getsize(set_path),
    }
    ok = True
    msgs = []

    # ---- Layer 1: structural assertions via scipy ---------------------------
    # struct_as_record=True is critical -- with =False, scipy returns
    # mat_struct Python objects which mask the cell-vs-struct distinction
    # that EEGLAB cares about.
    m = sio.loadmat(str(set_path), squeeze_me=False, struct_as_record=True)

    # Required top-level keys (eeglabio's flat schema)
    required_top = ['data', 'srate', 'nbchan', 'pnts', 'chanlocs', 'event']
    missing = [k for k in required_top if k not in m]
    if missing:
        ok = False
        msgs.append(f'missing top-level keys: {missing}')
        details['ok'] = ok
        details['errors'] = msgs
        return details

    # event: must be a numpy structured array (struct), NOT object dtype (cell)
    ev = m['event']
    details['event_dtype']      = str(ev.dtype)
    details['event_dtype_kind'] = ev.dtype.kind
    details['event_fieldnames'] = list(ev.dtype.names) if ev.dtype.names else None
    details['event_n']          = int(ev.size)
    if ev.dtype.names is None:
        ok = False
        msgs.append(f'EEG.event is not a struct array (dtype={ev.dtype}); '
                    f'pop_loadset will error in fieldnames() at line 887')
    else:
        for f in ('type', 'latency', 'duration'):
            if f not in ev.dtype.names:
                ok = False
                msgs.append(f'EEG.event missing field {f!r}')

    # chanlocs: same requirement
    cl = m['chanlocs']
    details['chanlocs_dtype']      = str(cl.dtype)
    details['chanlocs_dtype_kind'] = cl.dtype.kind
    details['chanlocs_fieldnames'] = list(cl.dtype.names) if cl.dtype.names else None
    details['chanlocs_n']          = int(cl.size)
    if cl.dtype.names is None:
        ok = False
        msgs.append(f'EEG.chanlocs is not a struct array (dtype={cl.dtype})')
    else:
        for f in ('labels', 'X', 'Y', 'Z'):
            if f not in cl.dtype.names:
                ok = False
                msgs.append(f'EEG.chanlocs missing field {f!r}')

    # data: numeric float32 array, NOT a string (we're single-file, not split)
    data_mat = m['data']
    details['data_dtype'] = str(data_mat.dtype)
    details['data_shape'] = tuple(data_mat.shape)
    if data_mat.dtype.kind not in ('f', 'i', 'u'):
        ok = False
        msgs.append(f'EEG.data has non-numeric dtype {data_mat.dtype}; '
                    f'expected float32 inline array for single-file .set')
    if data_mat.shape != ref_data.shape:
        ok = False
        msgs.append(f'EEG.data shape {data_mat.shape} != ref {ref_data.shape}')

    # Scalars: srate, nbchan, pnts
    srate_v = float(np.asarray(m['srate']).squeeze())
    nbchan_v = float(np.asarray(m['nbchan']).squeeze())
    pnts_v   = float(np.asarray(m['pnts']).squeeze())
    details['srate']  = srate_v
    details['nbchan'] = nbchan_v
    details['pnts']   = pnts_v
    if abs(srate_v - ref_sfreq) > 1e-6:
        ok = False; msgs.append(f'srate mismatch: {srate_v} vs {ref_sfreq}')
    if int(nbchan_v) != ref_data.shape[0]:
        ok = False; msgs.append(f'nbchan {int(nbchan_v)} vs ref {ref_data.shape[0]}')
    if int(pnts_v) != ref_data.shape[1]:
        ok = False; msgs.append(f'pnts {int(pnts_v)} vs ref {ref_data.shape[1]}')

    # Numeric round-trip directly from the MAT file. eeglabio multiplies by
    # 1e6 (V -> µV) before saving, so we undo that here to compare in V.
    mat_data_v = np.asarray(data_mat, dtype=np.float64) * 1e-6
    if mat_data_v.shape == ref_data.shape:
        diff = ref_data - mat_data_v
        rel = _rms(diff) / max(_rms(ref_data), 1e-30)
        details['data_rms_ref']    = _rms(ref_data)
        details['mat_rel_rms']     = rel
        details['mat_max_abs_diff'] = float(np.max(np.abs(diff)))
        if rel > tol_relative_rms:
            ok = False
            msgs.append(f'MAT data rel-rms {rel:.2e} exceeds tol '
                        f'{tol_relative_rms:.0e}')

    # ---- Layer 2: MNE re-read (sanity) -------------------------------------
    try:
        reread = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
        read_data = np.asarray(reread.get_data())
        details['mne_read_shape']  = tuple(read_data.shape)
        details['mne_read_sfreq']  = float(reread.info['sfreq'])
        details['mne_n_events']    = len(reread.annotations)
        details['mne_channels_ok'] = list(reread.info['ch_names']) == ref_chn
        if read_data.shape == ref_data.shape:
            mne_rel = _rms(ref_data - read_data) / max(_rms(ref_data), 1e-30)
            details['mne_rel_rms'] = mne_rel
            if mne_rel > tol_relative_rms:
                ok = False
                msgs.append(f'MNE re-read rel-rms {mne_rel:.2e} exceeds '
                            f'tol {tol_relative_rms:.0e}')
    except Exception as e:
        ok = False
        msgs.append(f'MNE read_raw_eeglab failed: {e}')

    # Sidecar JSON readable?
    try:
        side = json.load(open(meta_path))
        details['sidecar_keys'] = sorted(side.keys())
    except Exception as e:
        ok = False; msgs.append(f'sidecar unreadable: {e}')

    details['ok']     = ok
    details['errors'] = msgs
    return details


# ---------------------------------- CLI ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr',   required=True,
                    help='path to eeg_raw.zarr or eeg_clean.zarr')
    ap.add_argument('--domain', default='ci', choices=('ci', 'hearing'))
    ap.add_argument('--name',   required=True,
                    help='recording name, e.g. noise0_perm1')
    ap.add_argument('--stream', default=None,
                    help='data / broadband_data / erp_data; auto if omitted')
    ap.add_argument('--out',    required=True,
                    help='output .set path (the .fdt is written alongside; '
                         'a <name>_meta.json sidecar is written too)')
    ap.add_argument('--roundtrip', action='store_true',
                    help='after export, read it back and compare against the source')
    ap.add_argument('--t-end', type=float, default=None,
                    help='if set, crop output to [0, t_end) seconds before export '
                         '(used to produce a small test file)')
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass

    if args.roundtrip:
        details = roundtrip_check(args.zarr, args.domain, args.name,
                                  args.out, stream=args.stream,
                                  t_end_seconds=args.t_end)
        print('=== round-trip check ===')
        for k, v in details.items():
            print(f'  {k}: {v}')
        sys.exit(0 if details['ok'] else 1)
    else:
        set_path, meta_path, meta = export_to_set(
            args.zarr, args.domain, args.name, args.out, stream=args.stream,
            t_end_seconds=args.t_end,
        )
        print(f'wrote {set_path}')
        print(f'wrote {meta_path}')
        print(f'metadata keys preserved: {sorted(meta.keys())}')


if __name__ == '__main__':
    main()
