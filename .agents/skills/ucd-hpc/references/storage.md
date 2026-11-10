# Storage and data movement on HPC@UCD

| Space | Hive | Farm | Franklin | Notes |
|---|---|---|---|---|
| Home | `/home/$USER`, 20 GB, **backed up nightly** | `/home/$USER`, 20 GB, no backup | `/home/$USER`, 20 GB, no backup | `df -h ~`. Dotfiles and small scripts only |
| Group / PI share | `/quobyte/<pi>grp` (Quobyte) | `/group/<pi>grp` (NFS) and/or `/quobyte/<pi>grp` (newer) | `/group/<pi>grp` (ZFS/NFS, autofs) | Data, envs, containers, results. Named after the PI's login |
| Per-job local scratch | `/tmp` and `/scratch` (~1.6 TB), `$TMPDIR=/tmp` | `/tmp` and `/scratch` (~1.8 TB), `$TMPDIR=/tmp` | `/tmp` (~0.9 TB), no `/scratch` | Private, fastest, **deleted at job end** — copy out before exit |
| Network scratch | `/nfs/hive/scratch` (22 TB, shared, not purged, not backed up) | — | — | Intermediates shared between jobs; clean up yourself |
| Special | `/nfs/hive/scratch-mpi-io` (needs `mpi-io-grp`), `/nfs/peloton/waltz` (legacy, read) | `/share/apps` | `/share/databases` (alphafold, blast, relion, ...) | |

`/group` mounts on demand: `ls /group` may not list a share until you `cd` into it. Your group is
the `...grp` entry in `id -Gn`.

## Walking these file systems costs everyone

Every directory that `find`, `du`, `ls -R`, or `rsync --dry-run` enters is a round trip to a
metadata server shared by the whole cluster; conda environments and single-cell datasets are tens
of thousands of small files each. Bound walks on the login node (`-maxdepth 2`), run deeper ones
in a job, ask cheaper questions first (`df -h`, `qinfo quota PATH`, `ls` with a glob,
`sacct --format=WorkDir`), and when a full inventory is really needed produce it once in a job,
write the listing to a file, and answer every question from the file.

## Quotas

`df -h ~` (home), `qinfo quota /quobyte/PIGRP` (may say "No effective quotas"),
`du -sh ~/.conda ~/.cache ~/.apptainer ~/.local 2>/dev/null | sort -h`. `Disk quota exceeded` in
home is almost always conda/pip/Apptainer caches or environments; relocate them (`software.md`),
`conda clean --all`, delete the old copies. Extra home space is not sold.

## Quobyte: one writer per file

On Hive (all `/quobyte`) and Farm `/quobyte`, several nodes appending to one file cause lock
contention that hangs `slurmd`, drops the node from the cluster, and gets the account locked.
`--output`/`--error` carry `%j` (`%A_%a` for arrays, plus `%N` when multi-node); array tasks and
MPI ranks write distinct files (`out-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}`,
`$(hostname)-${SLURM_JOB_ID}`); anything that must share a file uses locking, or writes to
`$TMPDIR` and merges afterwards. Crontabs are per login node, so the same cron entry on `login1`
and `login2` is two writers too.

## Permissions

Shares are group-owned; `chmod g+s DIR` makes new files inherit the group. Lab mates read with
`chmod -R g+rX DIR`, write with `g+rwX`; per-user access with `setfacl -m u:LOGIN:rx DIR`. Not `777`.

## Backups

Hive only. Home nightly (7 daily / 5 weekly / 3 monthly). Group shares only inside
`/quobyte/PIGRP/BACKED-UP/` and only if the PI bought backup space; symlinks are stored as links.
Restore: `cd ~` (or the share), `/quobyte/backups/bin/restore.sh`, copy from the mounted
`snapshots/` in a second shell, Ctrl-C to unmount. **Farm and Franklin have no backups**; Farm
warns at login about storage past its five-year life. Backups are on campus, not off-site.

## Data transfer

Hosts `farm|franklin|hive.hpc.ucdavis.edu`; Farm/Franklin are key-only, Hive also takes the campus
passphrase. Transfers run on login nodes — expected use, but not dozens of parallel streams.

- **rsync** resumes and preserves metadata; trailing slashes matter (`src/` = contents):
  `rsync -a --one-file-system --info=progress2 ./data/ USER@hive.hpc.ucdavis.edu:/quobyte/PIGRP/USER/data/`.
  `--delete` permanently removes extras on the destination: only for an explicit mirror.
  Cluster to cluster: run rsync on one login node with the other as remote.
- **scp** for a few files. **Globus** (free tier): `UC Davis <Cluster> home` collections exist;
  PI shares are exported on request and writable only under `/globus-write/<login>/`.
- **Box**: `module load rclone`, configure a Box remote (OAuth via an OnDemand desktop browser;
  steps at <https://docs.hpc.ucdavis.edu/data-transfer/>), then `rclone copy|sync`.
- **OnDemand Files** for files under a few hundred MB. **Public datasets**: `aria2c`, `wget`,
  `awscli`, `sratoolkit`, `aspera-cli` modules, from inside a job.

## Using scratch

```bash
cd "$TMPDIR"; cp /quobyte/PIGRP/USER/input.bam .
tool --threads "$SLURM_CPUS_PER_TASK" input.bam > output.bam
cp output.bam /quobyte/PIGRP/USER/results/output-${SLURM_JOB_ID}.bam   # before the job exits
```

Copy-out is part of the script (ideally under `trap ... EXIT`). Many-small-file workloads run far
faster from local scratch than from a network share.
