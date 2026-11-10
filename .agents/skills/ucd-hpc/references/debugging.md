# Debugging jobs on HPC@UCD

Start with `scripts/job-postmortem.sh JOBID`; this page explains what it prints and covers what it
does not.

## Triage

```bash
squeue --me                                                    # queued or running? reason if pending
squeue --me -o "%.30i %.12P %.20j %.3t %.12M %.12l %.5C %.10m %R" # + time used/limit, CPUs, mem
sacct -j ID -X --format=JobID,State,ExitCode,Elapsed,Timelimit,ReqMem,MaxRSS,NodeList,Reason
sacct -u $USER -S 2026-09-01 -X --format=JobID,JobName%20,Partition,State,ExitCode,Elapsed
scontrol show job ID          # everything, only while Slurm still remembers the job
scancel ID | scancel -u $USER | scontrol hold ID | scontrol release ID | sshare -U
```

`sacct` prints one line per job plus one per step (`.batch`, `.extern`, `.0`); `MaxRSS` and CPU
time are on the step lines. Exit codes are `program:signal`. Log files: `scontrol show job` prints
`StdOut=`/`StdErr=` while it remembers the job; otherwise `slurm-ID.out` (or the `--output` name) in
`sacct -j ID -X -n --format=WorkDir%200`. OnDemand sessions: click the `Session ID`, read `output.log`.

## Submission rejected (`sbatch: error: ...`)

| Message | Meaning | Fix |
|---|---|---|
| `Invalid account or account/partition combination specified` | no default account, or this account has no association with that partition | add `--account=`; pick from `sacctmgr show assoc user=$USER format=account%20,partition%20,qos%40`. Farm `publicgrp` has `low` only |
| `QOSMaxCpuPerJobLimit`, `QOSMaxMemoryPerJob`, `QOSMaxGRESPerJob` | over the per-job cap of this QOS (Hive free tier: 8 CPUs, 128 GB, 1 GPU) | shrink, use another account, or `low` (no per-job caps, preemptible) |
| `Requested time limit is invalid (missing or exceeds some limit)` | `--time` above the partition/QOS max | see the cluster file's `MaxTime` |
| `Requested node configuration is not available` | no single node fits the CPU/mem/GPU combination, or that GPU type is not in the partition | `sinfo -p PART -N -o "%N %c %m %G"`; lower or change partition |
| `Job violates accounting/QOS policy (...)` | generic `DenyOnLimit` refusal | `sbatch --test-only` names the limit; compare `sacctmgr show qos` |
| `Invalid generic resource (gres) specification` | typo in `--gpus=TYPE:N` or type not on this cluster | `sinfo -h -o "%G" \| sort -u` |
| `Invalid partition name specified` | partitions differ per cluster | see the cluster file |
| `Batch script contains DOS line breaks` | CRLF from a Windows editor | `sed -i 's/\r$//' script.sh` |

## Pending reasons (`squeue` REASON column)

| Reason | Meaning | Action |
|---|---|---|
| `Priority` | others ahead in the queue | wait; smaller/shorter requests backfill sooner; `sshare -U` |
| `Resources` | at the front, waiting for hardware | wait or shrink |
| `QOSGrpCpuLimit` / `QOSGrpMemLimit` / `QOSGrpGRES` | the account's pool is full of group members' jobs (Hive free tier: 128 CPUs / 2 TB / 5 GPUs) | wait, coordinate, or `low`. On Hive also caused by `--exclusive`: remove it |
| `QOSMax*PerUser*`, `AssocMax*`, `AssocGrp*` | per-user or association cap | wait or use another account |
| `JobArrayTaskLimit` | the `%N` throttle working | nothing |
| `Dependency` / `DependencyNeverSatisfied` | waiting on a parent / the parent failed | nothing / `scancel` and resubmit after fixing the parent |
| `ReqNodeNotAvail, Reserved for maintenance` | `--time` reaches into the maintenance window | shorten `--time`; `scontrol show reservation` |
| `ReqNodeNotAvail, UnavailableNodes:` | requested/constrained nodes are down | drop constraints; `sinfo -R` |
| `BadConstraints` | `--constraint` matches no node in the partition | `sinfo -N -h -o "%N %f"` |
| `JobHeldUser` / `JobHeldAdmin` / `launch failed requeued held` | held | `scontrol release ID`; report repeats |
| `InvalidAccount` / `InvalidQOS` | association changed | `sacctmgr show assoc user=$USER` |

**Is the queue actually busy?** Not a question `sbatch --test-only` answers: its "to start at" time
is one scheduler pass, identical for 1 or 48 CPUs, and was 1.5 h out on Hive with 1,551 CPUs idle.
Ask directly:

```bash
sinfo -p PART -h -o "%C"                             # allocated/idle/other/total CPUs
squeue -t PD -h -o "%r" | sort | uniq -c | sort -rn  # why pending jobs are pending
```

`Resources`/`Priority` are real contention; `QOSGrp*`, `Dependency`, `JobArrayTaskLimit` are jobs
blocked by their own groups and not standing between you and a node.

## The job ended badly

| State / evidence | Cause | Fix |
|---|---|---|
| `OUT_OF_MEMORY`, exit `0:125`, `oom-kill event(s)` in the log | exceeded the memory request | request ~1.3× `MaxRSS`; `--mem` is per node |
| `TIMEOUT`, `CANCELLED AT ... DUE TO TIME LIMIT` | hit `--time` | raise within `MaxTime`, longer partition, or checkpoint |
| `PREEMPTED` / requeued, `Restarts>0` | `low`/`bml` job displaced by an owner | expected there; use `high` or make it restart-safe |
| `NODE_FAIL` | node died | resubmit; report if it repeats |
| `CANCELLED by 0` / `by <uid>` | staff / that user ran `scancel` | check MOTD or email; `id UID` |
| `FAILED 127:0`, `command not found` | program not on PATH: no `module load`, env not activated, or `module` itself missing | `source /etc/profile.d/modules.sh` + the right `module load`; `module avail NAME` |
| `FAILED 126:0` | `Permission denied` | `chmod +x`; check the shebang |
| `FAILED 1:0`, `2:0` | the program's own error | read stderr; reproduce in `srun --pty` |
| `0:9` / `137` (SIGKILL), `0:11` / `139` (SIGSEGV) | OOM at process level or `scancel`; crash, often an `+amd`/`zen2` build on the wrong CPU | check for oom lines; use the generic build or `--constraint` |
| `COMPLETED` but wrong/missing output | a later step failed silently | `set -euo pipefail` so failures surface in `State` |

Other frequent log lines: `CUDA out of memory` (GPU RAM, not `--mem`), `Disk quota exceeded` (20 GB
home, see `storage.md`), `No space left on device` (per-job `/tmp` or a full share),
`ModuleNotFoundError` (wrong conda env), `Illegal instruction` (built for a newer CPU; `--constraint`).

**Right-sizing**: `sacct -j ID --format=Elapsed,Timelimit,ReqMem,MaxRSS,AllocCPUS,TotalCPU -P`.
Memory efficiency is `MaxRSS/ReqMem`; CPU efficiency is `TotalCPU/(Elapsed×AllocCPUS)`, and a low
value with many CPUs means the program never got `$SLURM_CPUS_PER_TASK`. `seff` is not installed.

## Login node problems

Symptoms of the per-user caps (2 CPUs, 7.5% RAM, 500 MB swap, 512 processes, 16,384 files):
processes vanish silently (OOM), `Too many open files`, `fork: retry: Resource temporarily
unavailable`. `pgrep --count --uid $USER` counts processes. The fix is a job, never shrinking or
staggering the work to squeeze under the caps. Who is capped and how to check:

```bash
cat /etc/security/systemd-user-limits.sh       # PAM applies CPUQuota=200% MemoryMax=7.5% MemorySwapMax=500M TasksMax=512,
                                               # skips UID<1000, and deletes the limits for hpccfgrp members
id -nG | grep -qw hpccfgrp && echo exempt || echo capped
ls /etc/systemd/system.control/user-$(id -u).slice.d/          # the drop-in; absent when exempt
systemctl show "user-$(id -u).slice" -p CPUQuotaPerSecUSec -p MemoryMax -p TasksMax
```

If you are exempt (staff), the caps still bind the person you are helping. `sinfo: command not
found` in a non-interactive shell: `source /etc/profile.d/modules.sh` or prepend
`/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/bin`; never `find` for it.

## Open OnDemand failures (`Session ID` → `output.log`)

- `EnvironmentNameNotFound` → conda env name typo.
- `R version change [...] detected` or `install.packages()` failing → clear `~/.RData`,
  `~/.local/share/rstudio*` (and `~/.cache/rstudio*`, `~/.config/rstudio*`); disable workspace restore.
- `oom_kill events` / kernel died → more memory in the form. `DUE TO TIME LIMIT` → more hours.
- `QOSMaxMemoryPerJob` etc. → the form exceeded the account's per-job cap.
- `ERROR: CONDA_EXE is currently defined` → self-installed conda; remove the `conda initialize`
  block from the shell rc.

## SSH and tickets

Password prompt on Farm/Franklin (key-only) → key not offered: `chmod 600 ~/.ssh/id_rsa`, `-i KEY`,
or the HiPPO key differs; Hive also accepts the campus passphrase. `REMOTE HOST IDENTIFICATION HAS
CHANGED` → compare fingerprints at <https://docs.hpc.ucdavis.edu/general/access/> before
`ssh-keygen -R`. New accounts need PI approval plus ~1 h; HiPPO syncs every 15 min.

Tickets to `hpc-help@ucdavis.edu` (Farm: `farm-hpc@ucdavis.edu`): username, cluster, account,
exact commands and directory, pasted error text (attachments are stripped; give paths), job ID.
