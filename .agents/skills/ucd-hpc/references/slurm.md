# Slurm on HPC@UCD clusters

Shared behavior for Farm, Franklin, and Hive (Slurm 26.x, `select/cons_tres`, cgroup enforcement,
backfill, partition-priority preemption). Partition tables are in the cluster files; monitoring
and failure diagnosis are in `debugging.md`.

## Where the commands live

All three clusters: `/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/bin/`
(`sinfo`, `squeue`, `sbatch`, `srun`, `salloc`, `sacct`, `sacctmgr`, `scontrol`, `scancel`,
`sstat`, `sshare`, `sprio`, `sreport`). Interactive logins have them on `PATH` via the `slurm`
module. A non-interactive shell (`ssh host cmd`, cron, a job step after `module purge`) may not:

```bash
source /etc/profile.d/modules.sh   # loads the slurm module and defines `module`
export PATH=/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/bin:$PATH
```

Never `find` for them; `command -v sinfo` answers instantly and the path above is fixed. (Franklin
also carries a stale Slurm 23.02 tree under `/share/apps`; ignore it.)

## Accounts, associations, QOS

- Access is an **association**: user + account + partition + QOS. Accounts are PI groups
  (`<pi-login>grp`), departments, or `publicgrp` (free tier).
- `sacctmgr show assoc user=$USER format=account%20,partition%20,qos%40`, or
  `/opt/hpccf/bin/slurm-show-resources.py` for the same as a table with the group pool and per-job
  caps (also printed at login). The account marked `⁺` is the default; any other needs `--account`.
- Limits are on the QOS, not the partition: `sacctmgr show qos NAME format=name,maxtres,grptres,maxtrespu,maxwall -P`.
  `MaxTRES` per job, `GrpTRES` the account's whole pool, `MaxTRESPU` per user. `DenyOnLimit`
  means a request over a per-job cap is rejected at submit time.
- `high`-type partitions (`high`, `bmh`, `*-h`, `<group>-gpu`) are owned resources with no
  preemption; `low`/`bml` run on idle owned nodes with `PreemptMode=REQUEUE` and a short grace
  (Hive 130 s, Franklin 60 s, Farm 0). A preempted job restarts from the top of its script.

## Requesting resources

**CPUs.** `--cpus-per-task/-c N` keeps N CPUs on one node (threads/OpenMP); `--ntasks/-n N` makes N
tasks that may span nodes (MPI, many processes; `srun` in the script launches one per task);
`--nodes/-N` sets a minimum or range (`1-4`). SMT is on: odd requests round up, and `-c 1` alone
yields 2 tasks unless `-n 1` is also given. Read the grant with `$SLURM_CPUS_PER_TASK`,
`$SLURM_NTASKS`, `$SLURM_JOB_NUM_NODES`.

**Memory.** `--mem=SIZE` per node, `--mem-per-cpu=SIZE`, or `--mem-per-gpu=SIZE` (mutually
exclusive). Units `K|M|G|T`; a bare number is MB. No request means `DefMemPerCPU × CPUs` for that
partition. Exceeding it kills the step (`oom-kill event(s)`); size from `MaxRSS` in `sacct`.

**GPUs.** `--gpus=N` or `--gpus=TYPE:N` (`--gres=gpu:TYPE:N` is equivalent); also
`--gpus-per-task`, `--cpus-per-gpu`, `--mem-per-gpu`. Types: `sinfo -p PART -h -o "%G|%D"`. Hive
types include `a100`, `a6000`, `6000_blackwell`, `l40s`, `5000_ada`; Farm `a100`, `h100`,
`6000_ada`, `a5500`, `v100`, `titan`; Franklin is untyped (`gpu:8`), select by
`--constraint=amd|intel`. `$CUDA_VISIBLE_DEVICES` is set in the job. Request CPUs and RAM too.

**Time.** `--time=D-HH:MM:SS` (`90` = 90 min, `2:00:00`, `1-00`). Over `MaxTime` fails with
`Requested time limit is invalid`. A limit reaching into a maintenance reservation pends as
`ReqNodeNotAvail, Reserved for maintenance`; shorten it.

**Constraints.** `--constraint='(zen3|zen4)'` (Hive CPU generations), `--constraint=amd` (Franklin).
`sinfo -N -h -o "%N %f" | sort -u` lists features. Every constraint lengthens the wait.

## Job scripts

```bash
#!/bin/bash
#SBATCH --job-name=NAME
#SBATCH --account=ACCOUNT
#SBATCH --partition=PARTITION
#SBATCH --time=D-HH:MM:SS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=N
#SBATCH --mem=SIZE
#SBATCH --output=NAME-%j.out
#SBATCH --error=NAME-%j.err
# optional: --gpus=1  --array=1-100%10  --dependency=afterok:ID  --constraint=...  --nodes=1-4
#           --mail-type=END,FAIL --mail-user=you@ucdavis.edu

source /etc/profile.d/modules.sh
module load PKG/VERSION
cd "$SLURM_SUBMIT_DIR"
```

- `#SBATCH` lines precede the first command; command-line flags override them.
- Filename patterns: `%j` job id, `%x` name, `%u` user, `%N` first node, `%A`/`%a` array job/task.
  Default output is `slurm-%j.out` in the submit directory.
- Variables: `SLURM_JOB_ID`, `SLURM_ARRAY_TASK_ID`, `SLURM_CPUS_PER_TASK`, `SLURM_NTASKS`,
  `SLURM_MEM_PER_NODE` (MB), `SLURM_SUBMIT_DIR`, `SLURM_JOB_NODELIST`, `TMPDIR` (per-job scratch).
- `sbatch --test-only script.sh` asks the scheduler to accept or reject without submitting. Its
  success line's "to start at" time is **not a wait estimate** (see `debugging.md`).

**Arrays.** `--array=1-300%20` (300 tasks, 20 at a time; `%20` shows as `JobArrayTaskLimit`, which
is fine), `--output=NAME-%A_%a.out`, and map the index to an input:
`SAMPLE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" samples.txt)`. Resource flags are per task; every task
needs its own output and result files. `scancel ID_5`, `scancel ID_[10-20]`, `scancel ID`.

**Dependencies.** `--dependency=afterok:ID` / `afterany` / `afternotok` / `singleton`;
`ID=$(sbatch --parsable step1.sh)`. A failed parent leaves children `DependencyNeverSatisfied`
forever; cancel them.

**MPI.** One task per rank (`--ntasks=128`, not `--cpus-per-task=128`); hybrid is
`--ntasks=16 --cpus-per-task=8` with `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`. Launch with `srun`
(PMIx) or `mpirun` from the MPI module used to build (`openmpi/5.0.5` default on Farm/Hive). Hive:
`--constraint='(zen2|zen3|zen4)'`, `--nodes=1-4`, `--distribution=block,pack`,
`--switches=1@1-00`, and `%N` in output names.

## Running work and waiting for it

```bash
# one command, output to the terminal, blocks until done — the right tool for anything you were
# tempted to run on the login node
srun -A ACC -p PART -t 15 -c 2 --mem=4G find /quobyte/PIGRP -maxdepth 6 -name '*.bam'

# a script, logs on disk, blocks until the job leaves the queue; exit status is the job's
sbatch --wait -A ACC -p PART -t 2:00:00 --mem=8G job.sh

# or submit, then poll (never busy-loop, never mirror the work onto the login node "meanwhile")
JOB=$(sbatch --parsable job.sh); squeue -j "$JOB" -h -o "%T %R"; sacct -j "$JOB" -X --format=State,Elapsed,MaxRSS

# interactive shell on a compute node (run inside tmux); add --gpus=1 for a GPU
srun -A ACC -p PART -t 2:00:00 -c 4 --mem=16G --pty bash -l

# shell inside an already running job (top, nvidia-smi, ls $TMPDIR); direct ssh is not allowed
srun --jobid=ID --overlap --pty bash -l
```

`salloc` with the same flags holds an allocation and returns a login-node shell from which `srun CMD`
runs steps. JupyterLab, RStudio, VS Code, and a desktop are Slurm jobs launched from Open OnDemand
(`https://ondemand.<cluster>.hpc.ucdavis.edu`) with the same account/partition/limit choices.
