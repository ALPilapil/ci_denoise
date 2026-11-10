# Franklin

College of Biological Sciences cluster (CNS, MMG, MCB, cryo-EM facility, approved collaborators):
genomics, cryo-EM, computational neuroscience. Docs: <https://docs.hpc.ucdavis.edu/franklin/>.

- Login `ssh USER@franklin.hpc.ucdavis.edu`, **SSH key only**; `-Y` for the Relion GUI. OnDemand:
  <https://ondemand.franklin.hpc.ucdavis.edu>. `cluster_name.conf` → `franklin`.
- Accounts via HiPPO <https://hippo.ucdavis.edu/Franklin>. **No free tier**: every user is
  sponsored by a PI or department; `low` access arrives through department QOS such as
  `mcbdept-low-qos`. Login prints the Slurm resources table.

## Partitions (verified 2026-09; `DefaultTime` unset — always pass `--time`)

| Partition | Max time | Default mem/CPU | Preempt | Nodes |
|---|---|---|---|---|
| `low` (**the default partition**) | 14 d | 2000 M | `REQUEUE`, 60 s grace | all 16 nodes (7 CPU + 9 GPU) when idle |
| `high` | 60 d | 2000 M | none | 7 CPU nodes, 256 logical CPUs / 1 TB each |
| `<group>-gpu` (`jawdatgrp-gpu`, `jalettsgrp-gpu`, `mmgdept-gpu`, `mcbdept-gpu`, `cashjngrp-gpu`, `mmaldogrp-gpu`, `cnsdept-gpu`, `ajfishergrp-gpu`) | unlimited | node default | none | the owning group's GPU node, 7–8 GPUs |

Omitting `--partition` lands on preemptible `low`; say so.

## GPUs

Untyped (`gpu:8`), so `--gpus=N` only; pick the node class by CPU feature:

| Constraint | Nodes | CPUs / RAM | GPUs |
|---|---|---|---|
| `--constraint=amd` | `gpu-7-42/50/54/58`, `gpu-9-58/66` | 256 / 1 TB | 8× RTX A4000 / A5000 / 6000 Ada (varies by owner) |
| `--constraint=intel` | `gpu-7-62`, `gpu-9-18`, `gpu-9-26` | 40 / 384 GB | 7–8× RTX 2080 Ti |

Group members use their `<group>-gpu` partition (no time limit); others reach idle GPUs via `low`.
Use `+amd` module variants on AMD nodes and `+intel` on Intel nodes.

## Storage

Home 20 GB, no backup. Lab storage `/group/<pi>grp` (autofs — appears once accessed by name;
`id -Gn` shows the group), purchased per lab, **no backups**. Per-job `$TMPDIR=/tmp` (~0.9 TB
local); **no `/scratch`**. `/share/databases`: `alphafold` (`$ALPHAFOLD_DB_ROOT`), `blast` (nr,
refseq_protein, swissprot, pdb, 16S/18S/28S, ...; weekly), `relion`, `cryolo`, `deepemhancer`,
`medic`. Cryo-EM facility Windows machines can mount lab storage via Samba on request. Globus:
`UC Davis Franklin home`; PI shares on request at `/group/<pi>grp/globus-write/<login>/`.

## Software

Environment Modules 5.5 (docs say lmod; same commands). Trees: `/share/apps/22.04/modulefiles/spack/core`
(gcc 7.5/11.4/13.2, aocc/4.1.0, intel-oneapi-compilers/2023.2.1, cuda 8.0/11.2/11.7,
openmpi/4.1.5{,+amd,+intel}, slurm), `.../spack/software` (`+amd`/`+intel` variants),
`/share/apps/franklin/modulefiles` (matlab, sbgrid, scipion), plus CVMFS `sw/modulefiles` and
`sw/conda/modulefiles`. Loaded at login:
`slurm`, `openmpi/default`, `ucx`.

- **Relion**: `relion/{cpu,gpu}/4.0.1+amd`, `relion/gpu/4.0.1+intel`, 3.1.3 and 5.0-beta variants;
  `module load relion/gpu` for the default. The GUI (`ssh -Y`) submits Slurm jobs itself; switch
  versions inside a project only with `relion-helper` (`module load conda; conda activate relion-helper`).
- **AlphaFold**: `module load alphafold/2.3.2`, then `alphafold-wrapped --output_dir=... --fasta_paths=...
  --max_template_date=... --use_gpu_relax=true` (fills the database paths). GPU node required.
- **Conda**: `module load conda` is the same central CVMFS miniforge as Farm and Hive (same
  `.condarc`, `conda/*` modules, rules in `software.md`). Cryo-EM envs have no modules; activate
  them by name (`conda env list`). `envs_dirs` searches CVMFS first, so `cryodrgn`,
  `deepemhancer-0.16`, `medic-1.0`, `pyem`, `relion-5.0`, `relion-helper` resolve to CVMFS
  copies; `cryolo-1.9.3-cuda-11`, `gpu-isac-2.3.4`, `sphire-2.91`, `spisonet`, `warp`, `scipion3`,
  `alphafold-2.3.2` exist only in the old `/share/apps/conda/environments`. Modules: `conda/topaz`,
  `motioncor2`, `ctffind/4.1.14+amd|+intel`, `gctf`, `scipion/3.0.12`, `sbgrid`.
