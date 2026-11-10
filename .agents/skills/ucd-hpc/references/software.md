# Software on HPC@UCD

| Mechanism | Farm & Hive | Franklin |
|---|---|---|
| Spack modules | `/cvmfs/hpc.ucdavis.edu/sw/spack/modulefiles/main/.../{core,lang,general}` (identical on both) | `/share/apps/22.04/modulefiles/spack/{core,software}` |
| Central conda + envs | `/cvmfs/hpc.ucdavis.edu/sw/conda/modulefiles` (`conda`, `conda/NAME/VERSION`, `R/VERSION`) | same CVMFS tree; older cryo-EM envs in `/share/apps/conda/environments` (activate by name) |
| Hand-installed | `/cvmfs/hpc.ucdavis.edu/sw/modulefiles` (matlab, fsl, phenix) | same tree mounted |
| Containers / web | `module load apptainer`; Open OnDemand (JupyterLab, RStudio, VS Code, desktop) | same |

Everything is Environment Modules 5.x (Franklin's docs say lmod; the commands are the same).
Release notes: `/cvmfs/hpc.ucdavis.edu/sw/RELEASES.md`. Look before asserting:

```bash
module avail NAME | module search NAME | module -t avail conda/ | module whatis NAME | module show NAME/VERSION
```

## Module naming

- `name/version`, with `(default)` marking what a bare `module load name` gives.
- Compiler suffix on CVMFS: `hdf5/1.14.5%oneapi@2025.0.0` is the Intel build; unsuffixed is GCC 13.2.
  Do not mix compilers in one job. MPI variants: `+openmpi-5.0.5` vs `+mpich-4.3.2`.
- Franklin arch variants: `+amd` (default, AMD nodes), `+intel` (the RTX 2080 Ti nodes only),
  unsuffixed = generic. The wrong one fails with `Illegal instruction`.
- Nested names need at least `name/variant`: `module load relion/gpu`, `conda/pytorch/2.9.1`.
- Extra CVMFS trees: `module load dev` (pre-release, may break), `module load zen2` (AMD-optimized
  builds; pair with `--constraint=zen2`). Every module sets `$<NAME>_ROOT`.

## Modules in job scripts

`module` is a shell function from `/etc/profile.d/modules.sh`; scripts submitted from zsh, cron,
OnDemand, or a scrubbed environment do not have it. Start scripts with
`source /etc/profile.d/modules.sh` (or `#!/bin/bash -l`), then `module load`. Two traps:
`modules.sh` reads `$MANPATH` unguarded, so source it **before** `set -u` or wrap it in
`set +u ... set -u`; and on Farm/Hive it purges then loads `slurm` and `openmpi`, so a later
`module purge` removes `srun` — `module load slurm` again if the script needs it.

## Conda and Python

- `module load conda` (miniforge: conda 25.1, `mamba` 2.0.5; does the shell hook for you). It is
  the same CVMFS install on all three clusters. Never `conda init` or install Miniconda in
  `$HOME`: it clashes with the central install and breaks OnDemand
  (`ERROR: CONDA_EXE is currently defined`). Migration = delete the `# >>> conda initialize >>>`
  block from the shell rc and log in again.
- The conda module conflicts with `gcc`, `oneapi`, `aocc`, `nvhpc`: a later `module load gcc`
  silently unloads conda, so an env's compiler has to come from the env itself.
- Environments and caches go on group storage, not the 20 GB home:

  ```bash
  module load conda
  conda config --add envs_dirs /quobyte/PIGRP/$USER/envs     # /group/PIGRP/... on Farm/Franklin
  conda config --add pkgs_dirs /quobyte/PIGRP/$USER/conda-pkgs
  mamba create -n myenv python=3.12 scanpy jupyterlab ipykernel
  conda clean --all                                           # reclaim the old home cache
  ```

- Default packages: the site `.condarc` sets `create_default_packages: gcc=13`, so
  `conda create` adds a GCC 13 toolchain (~75 MB) to every env, ready for pip/R source builds.
  `conda create --no-default-packages` skips it. `mamba create` (what the docs recommend)
  ignores the setting and rejects that flag; add `gcc=13` to the spec when an env will compile.
- Solves of large environments exceed login-node limits: do them in
  `srun -A ACC -p PART -t 1:00:00 -c 2 --mem=16G --pty bash -l`.
- Central envs: `module load conda/pytorch/2.9.1` = `module load conda` + `conda activate` of that
  env. **Verify what you got** (`python -c "import torch; print(torch.__version__)"`); a
  modulefile's `_conda_envname` (shown by `module show`) can point at a different version than its
  name. `envs_dirs` already lists `/cvmfs/hpc.ucdavis.edu/sw/conda/environments/` (then, on
  Franklin, `/share/apps/conda/environments/`), so `conda activate NAME` works for every central
  env, with or without a module; `conda env list` shows them.
- **Never `pip install` into a central env.** It is read-only CVMFS, so pip silently falls back to
  `~/.local/lib/pythonX.Y/site-packages`. That fills home, shadows packages in *every* env of that
  Python version, and causes import errors later. To add packages to a central env, layer a venv:

  ```bash
  module load conda/pytorch/2.9.1
  python -m venv --system-site-packages /quobyte/PIGRP/$USER/venvs/torch-extra   # /group/... on Farm/Franklin
  source /quobyte/PIGRP/$USER/venvs/torch-extra/bin/activate && pip install PKG
  ```

  In jobs, load the same module, then `source .../bin/activate`. The venv breaks if that
  central env changes, so recreate it then. Otherwise build your own env. `pip install` is fine
  inside an env you created; avoid `pip install --user` (same `~/.local` problem).
  `export PYTHONNOUSERSITE=1` hides an existing `~/.local` that is shadowing packages.
- In job scripts: `source /etc/profile.d/modules.sh; module load conda; conda activate ENV`.
  `module load python/3.11.9` + `venv` is the conda-free alternative.
- `mamba` prints harmless `Cache file ... was modified by another program` warnings.

## R, Jupyter, Apptainer

- `module load R/4.4.2` (or `R/4.3.3`). RStudio Server runs via OnDemand; user libraries default to
  `~/R/...` — set `R_LIBS_USER` in `~/.Renviron` to group storage if home fills.
- OnDemand JupyterLab takes a conda env name that must appear in `conda env list` (registering
  `envs_dirs` makes a prefix env show by name); a kernel registered with
  `python -m ipykernel install --user --name ENV` also appears in the picker. Form fields map to
  `sbatch` flags and the same QOS caps apply.
- Apptainer: `module load apptainer`; `export APPTAINER_CACHEDIR=/quobyte/PIGRP/$USER/apptainer-cache`
  (keeps `~/.apptainer` small); `apptainer build img.sif docker://IMAGE` once, in a job;
  `apptainer exec --nv img.sif CMD` for GPUs; `APPTAINER_BIND=/quobyte/PIGRP,...` for extra paths.
  No Docker daemon; any OCI image works this way.

## Compilers, MPI, CUDA

Loaded at login on Farm/Hive: `slurm`, `openmpi/5.0.5`. Defaults: `gcc/13.2.0`, `cuda/13.3.0`,
`oneapi/2025.0.0`, `nvhpc/24.9`, `aocc/5.0.0`; also `mpich/4.3.2`, MKL, FFTW, HDF5, NetCDF, Boost.
Franklin: `gcc/13.2.0`, `cuda/11.7.1`, `openmpi/4.1.5{,+amd,+intel}`. CUDA modules are for
building; GPU nodes carry the driver. `-march=native` builds belong on a node of the target type.

## Requesting software

Read <https://hpc.ucdavis.edu/software-installation-policy> and use the linked form; packages
already in Spack or conda-forge/bioconda are approved fastest. Meanwhile: conda on group storage,
or Apptainer.
