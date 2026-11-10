# Hive

Centrally managed campus cluster, open to all UC Davis staff, faculty, and graduate students
(HPC2's hardware merged in July 2026). Docs: <https://docs.hpc.ucdavis.edu/hive/>.

- Login `ssh USER@hive.hpc.ucdavis.edu` (→ `login1`/`login2`); SSH key **or** campus passphrase.
  OnDemand: <https://ondemand.hive.hpc.ucdavis.edu>. `cluster_name.conf` → `hive`.
- Accounts via HiPPO <https://hippo.ucdavis.edu/Hive>; free tier = sponsor
  `HPC@UCD Sponsored Public Access (publicgrp)`. PI groups buy CPUs/GPUs/TB
  (<https://hpc.ucdavis.edu/rates#hive>). Every login prints the Slurm resources table.

## Partitions (verified 2026-09; `DefaultTime` unset everywhere — always pass `--time`)

| Partition | Max time | Default mem/CPU | Preempt | Contents |
|---|---|---|---|---|
| `high` | 30 d | 4000 M | none | ~150 CPU nodes + 12 general A6000/Blackwell GPUs; purchased shares and the free tier |
| `low` | 7 d | 4000 M | `REQUEUE`, 130 s grace | **all** nodes and every GPU (86) when idle; free for everyone |
| `gpu-a100` | 30 d | 8000 M | none | 9 nodes, 44 A100 80 GB; owning groups |
| `gpu-a6000` / `gpu-6000-blackwell` | 30 d | 16000 M | none | 1 node × 4 A6000 / 4 nodes × 14 RTX PRO 6000 Blackwell |
| `gpu-a100-40gb`, `gpu-5000-ada`, `gpu-l40s` | 30 d | 2000 M | none | 1 node each |
| `gpu-h100` | 30 d | | none | one group's H100 nodes (also via `low`) |
| `hpccf`, `burnin` | | | | staff; ignore |

## Free tier (`publicgrp`)

| Partition | Per job | Pool shared by all free-tier users |
|---|---|---|
| `high` | 8 CPUs, 128 GB, 1 GPU, 30 days | 128 CPUs, 2000 GB, 5 GPUs |
| `low` | no per-job caps, 7 days, preemptible | everything idle |

Over a per-job cap fails at submit (`QOSMaxCpuPerJobLimit`, `QOSMaxMemoryPerJob`,
`QOSMaxGRESPerJob`, `Requested time limit is invalid`); a full pool pends as `QOSGrp*`. Needing more
than 8 CPUs / 128 GB means `low` (and preemption) or joining a PI group. The free-tier GPU in `high`
may be an A6000 **or** an RTX PRO 6000 Blackwell, which needs CUDA ≥ 12.8 builds.

## GPUs and CPUs

`--gpus=TYPE:1` types (recheck with `sinfo -p low -h -o "%G|%D"`): `a6000` (`high`, `low`,
`gpu-a6000`), `6000_blackwell` (`high`, `low`, `gpu-6000-blackwell`), `a100` (`low`, `gpu-a100`;
some nodes are typed `nvidia_a100-sxm4-80gb` / `nvidia_a100_80gb_pcie` and match only `--gpus=1`
plus the partition), `l40s`, `5000_ada`. `--gpus=1` takes any idle GPU. Verify with
`sbatch --test-only`. CPU generations as features: `zen`, `zen2`, `zen3`, `zen4`, `zen5`,
`icelake` (`--constraint='(zen3|zen4)'`); `cluster-context.sh` prints the node shapes.

## Hive-specific rules

- Group storage is all Quobyte (`/quobyte/<pi>grp`): unique `--output`/`--error` per writer
  (`%j`, `%A_%a`, `%N`) is mandatory.
- **Never `--exclusive`**: it does not schedule and shows a bogus `QOSGrpCpuLimit`.
- MPI: `--ntasks=N`, `--nodes=1-4`, `--constraint='(zen2|zen3|zen4)'`,
  `--distribution=block,pack`, `--switches=1@1-00`; MPI-IO in `/nfs/hive/scratch-mpi-io/<jobid>/`
  after joining `mpi-io-grp`.
- `high` normally starts within minutes; otherwise the group pool is full (`QOSGrp*`), the request
  needs a large contiguous chunk, or a maintenance reservation is pending.

## Storage

Home 20 GB, backed up nightly (restore: `/quobyte/backups/bin/restore.sh`). `/quobyte/<pi>grp`
purchased capacity, backups only for `BACKED-UP/` if bought, quota via `qinfo quota`. Per-job
`/tmp` and `/scratch` (~1.6 TB local, `$TMPDIR=/tmp`, wiped at exit). `/nfs/hive/scratch` shared
network scratch (persists, not purged, not backed up, clean up yourself). `/nfs/peloton/waltz`
legacy Peloton storage; LSSC0 data was migrated into `/quobyte/<pi>grp`.

## Software

Full CVMFS tree (`module avail`: core, lang, general; `conda/*`, `R/*`; matlab, fsl, phenix).
Loaded at login: `slurm`, `openmpi/5.0.5`. `module load dev` (pre-release) and `module load zen2`
(AMD-optimized) add trees. Apptainer; OnDemand JupyterLab/RStudio/VS Code/desktop.
