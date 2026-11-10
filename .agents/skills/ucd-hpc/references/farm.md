# Farm

CA&ES condo cluster; PIs from any college can buy in. Docs: <https://docs.hpc.ucdavis.edu/farm/>.
Support: `farm-hpc@ucdavis.edu`.

- Login `ssh USER@farm.hpc.ucdavis.edu`, **SSH key only**. OnDemand:
  <https://ondemand.farm.hpc.ucdavis.edu>. `cluster_name.conf` → `farm`.
- Accounts via HiPPO <https://hippo.ucdavis.edu/Farm>. Free tier = sponsor `CA&ES free tier`
  (`publicgrp`), which grants **`low` only** — `publicgrp` + `high` is `Invalid account or
  account/partition combination`. Hardware and storage bought through HiPPO on a five-year life.
- Login prints the Slurm resources table and, when relevant, an **expired storage** warning: data
  on hardware past five years is unsupported and not backed up.

## Partitions (verified 2026-09; `DefaultTime` unset — always pass `--time`)

| Partition | Max time | Default mem/CPU | Preempt | Nodes / purpose |
|---|---|---|---|---|
| `low` | 7 d | 2000 M | `REQUEUE` | ~98 nodes: all parallel + bigmem + every GPU node when idle; free for everyone |
| `high` | 150 d | 2000 M | none | ~66 parallel nodes (64 CPU/256 GB class, newer 128-CPU); purchased |
| `bml` / `bmh` | 7 d / 150 d | 4096 M | `REQUEUE` / none | 22 bigmem nodes (up to 128 CPU / 2 TB); scavenger / purchased |
| `bgpu` | 150 d | 3072 M | none | 2 nodes: `a5500:4`, `v100:1` |
| `gpuh` / `gpum` | 7 d | 15360 M | none | 2 `titan` nodes; high / medium priority |
| `gpu-a100-h` | 34 d | 7168 M | none | 3 nodes (`a100` ×4 and ×8), `MaxNodes=1` |
| `gpu-6000_ada-h` / `gpu-h100-h` | 30 d | 12000 M / 4000 M | none | 2 nodes `6000_ada:4` / 1 node `h100:4` |

Naming: `-h` = owner priority, `-m`/`gpum` = medium, `l` = scavenger.

## GPUs

`--gpus=TYPE:N` types, all also in `low` when idle: `a100`, `h100`, `6000_ada`, `a5500`, `v100`,
`titan` (`sinfo -p low -h -o "%G|%D"`). The free tier's only GPU route is `--partition=low`
(preemptible); owners use the `*-h`, `bgpu`, `gpuh` partitions. Node features: `cpu`, `bm`, `gpu`,
`gpu:a100,nvlink:4`, `gpu:h100,nvlink:4`, `gpu:6000ada`, `gpu:a5500,bgpu`, `gpu:v100,bgpu`, `gpu:titan`.

## Storage

Home 20 GB, **no backup**. Group shares `/group/<pi>grp` (NFS; `ls /group` shows only mounted
shares — `cd` into yours by name) or, for newer purchases, `/quobyte/<pi>grp` (one-writer rule,
`qinfo quota`). Several `/group` shares run near 100% — `df -h` before large writes. Per-job `/tmp`
and `/scratch` (~1.8 TB local, `$TMPDIR=/tmp`) are deleted at job end. **Nothing on Farm is backed
up**; users arrange their own copies. Globus: `UC Davis Farm home`; PI shares on request, writable at
`/group/<pi>grp/globus-write/<login>/`.

## Software

Same CVMFS module tree and central conda as Hive (`module avail`, `module load conda`, `conda/*`,
`R/*`, matlab). Loaded at login: `slurm`, `openmpi/5.0.5`. Apptainer; OnDemand apps.
