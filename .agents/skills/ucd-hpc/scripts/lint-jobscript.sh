#!/usr/bin/env bash
# ucd-hpc: lint an sbatch script for HPC@UCD-specific mistakes, then validate it with
# `sbatch --test-only` (asks the scheduler, submits nothing).
# Usage: lint-jobscript.sh SCRIPT [--no-test]
# Exit: 0 clean, 1 errors found, 2 usage. A scheduler that does not answer is a warning, not an error.
set -u
F=${1:-}; [ -n "$F" ] || { echo "usage: lint-jobscript.sh SCRIPT [--no-test]"; exit 2; }
NOTEST=0; [ "${2:-}" = "--no-test" ] && NOTEST=1
[ -r "$F" ] || { echo "cannot read $F"; exit 2; }
ERR=0; WARN=0
err()  { echo "ERROR: $*"; ERR=$((ERR+1)); }
warn() { echo "WARN:  $*"; WARN=$((WARN+1)); }
info() { echo "info:  $*"; }

HPCCF_ROOT=${UCD_HPC_HPCCF_ROOT:-/opt/hpccf}   # overridable so the tests do not pick up the real cluster
CLUSTER=""; [ -r "$HPCCF_ROOT/etc/cluster_name.conf" ] && CLUSTER=$(. "$HPCCF_ROOT/etc/cluster_name.conf" 2>/dev/null; echo "${CLUSTER_NAME:-}")
SLURM_BIN=/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/bin   # fixed path on Hive/Farm/Franklin
if ! command -v sbatch >/dev/null 2>&1; then
  if [ -r /etc/profile.d/modules.sh ]; then
    # `set -u` must be off here: modules.sh dereferences unset variables and would abort us.
    set +u
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh >/dev/null 2>&1 || true
    set -u
  fi
  if ! command -v sbatch >/dev/null 2>&1 && [ -d "$SLURM_BIN" ]; then
    PATH="$SLURM_BIN:$PATH"
  fi
fi

# collect directives, each normalized to '--opt=val' / '-o=val' (from '--opt=val', '--opt val', '-o val', '-oval')
# so that has/val match whole option names, never prefixes (--time vs --time-min, --mem vs --mem-bind).
DIR=$(grep -E '^[[:space:]]*#SBATCH' "$F" | sed -E 's/^[[:space:]]*#SBATCH[[:space:]]+//; s/[[:space:]]+#.*$//; s/[[:space:]]+$//' \
  | sed -E 's/^(--[A-Za-z0-9-]+)[[:space:]]*(=|[[:space:]])[[:space:]]*/\1=/; s/^(-[A-Za-z])[[:space:]]*=?[[:space:]]*([^[:space:]=])/\1=\2/')
# has 'OPT|OPT...' -> the option is set;  val 'OPT|OPT...' -> its first value
has() { grep -qE -- "^($1)(=|$)" <<<"$DIR"; }
val() { sed -nE "s/^($1)=([^[:space:]]*).*/\2/p" <<<"$DIR" | head -1; }
gpu_job() { has '--gpus|-G|--gpus-per-[a-z]+' || [[ $(val '--gres') == *gpu* ]]; }
BODY=$(grep -vE '^[[:space:]]*(#|$)' "$F")

# --- structure -------------------------------------------------------------------------
first=$(head -1 "$F")
[[ $first == '#!'* ]] || err "first line must be a shebang such as #!/bin/bash"
grep -q $'\r' "$F" && err "CRLF line endings (Windows). Fix: sed -i 's/\\r\$//' $F"
# directives after the first command are ignored by sbatch
awk 'BEGIN{cmd=0} /^[[:space:]]*#SBATCH/{ if(cmd) {print "late"; exit} ; next} /^[[:space:]]*#!/{next} /^[[:space:]]*(#|$)/{next} {cmd=1}' "$F" | grep -q late \
  && err "#SBATCH lines appear after the first command; sbatch ignores those"

# --- required requests ---------------------------------------------------------------
has '--account|-A' || err "no --account: many users have no default account (Invalid account or account/partition combination). Choose from: sacctmgr show assoc user=\$USER format=account%20,partition%20,qos%40"
has '--partition|-p' || err "no --partition: defaults differ per cluster (Franklin defaults to preemptible low). Set it explicitly"
has '--time|-t' || err "no --time: set a wall-clock limit (D-HH:MM:SS) under the partition maximum"
if ! has '--mem|--mem-per-cpu|--mem-per-gpu'; then
  warn "no memory request: the partition's DefMemPerCPU applies (2 GB Farm/Franklin, 4 GB Hive high/low, more on GPU partitions). Add --mem=SIZE or --mem-per-cpu=SIZE"
fi
m=$(val '--mem|--mem-per-cpu|--mem-per-gpu'); [ -n "$m" ] && [[ $m =~ ^[0-9]+$ ]] && warn "--mem=$m has no unit: Slurm reads megabytes. Use e.g. ${m}G if you meant gigabytes"
t=$(val '--time|-t'); [ -n "$t" ] && [[ $t =~ ^[0-9]+$ ]] && info "--time=$t is read as $t minutes"

ACC=$(val '--account|-A'); PART=$(val '--partition|-p')
if [ "$CLUSTER" = hive ] && [ "$ACC" = publicgrp ] && [ "$PART" = high ]; then
  c=$(val '--cpus-per-task|-c'); n=$(val '--ntasks|-n'); g=$(val '--gpus|-G'); g=${g##*:}
  [ -n "$c" ] && [ "${c:-1}" -gt 8 ] && err "Hive free tier (publicgrp/high) allows at most 8 CPUs per job (QOSMaxCpuPerJobLimit). Use --partition=low (preemptible) or a PI account"
  [ -n "$g" ] && [ "${g:-1}" -gt 1 ] && err "Hive free tier (publicgrp/high) allows 1 GPU per job (QOSMaxGRESPerJob)"
  [ -n "$m" ] && [[ $m =~ ^([0-9]+)G$ ]] && [ "${BASH_REMATCH[1]}" -gt 128 ] && err "Hive free tier (publicgrp/high) allows 128 GB per job (QOSMaxMemoryPerJob)"
fi
if [ "$CLUSTER" = farm ] && [ "$ACC" = publicgrp ] && [ "$PART" != low ] && [ -n "$PART" ]; then
  err "On Farm publicgrp only has access to --partition=low"
fi
[ "$PART" = low ] || [ "$PART" = bml ] && info "$PART is preemptible (PreemptMode=REQUEUE): the job is killed and restarted from scratch when owners need the nodes. Fine for short or restartable work"

# --- output files / Quobyte ------------------------------------------------------------
out=$(val '--output|-o'); errf=$(val '--error|-e')
ARRAY=0; has '--array|-a' && ARRAY=1
check_out() {
  local name=$1 v=$2
  [ -z "$v" ] && return
  if [ $ARRAY = 1 ]; then
    [[ $v == *%A* && $v == *%a* ]] || [[ $v == *%j* ]] || err "$name=$v for an array job must contain %A_%a (or %j) so every task writes its own file (Quobyte lock-up otherwise)"
  else
    [[ $v == *%j* || $v == *%J* || $v == *%A* ]] || err "$name=$v has no %j: every job (and every rerun) would append to the same file; on Quobyte concurrent writers hang the node and get accounts locked"
  fi
  n=$(val '--nodes|-N'); nt=$(val '--ntasks|-n')
  if { [ -n "$n" ] && [ "${n%%-*}" != 1 ]; } || { [ -n "$nt" ] && [ "$nt" -gt 4 ] 2>/dev/null && [ "$n" != 1 ]; }; then
    [[ $v == *%N* ]] || warn "$name=$v: multi-node job on Quobyte should include %N so each node writes its own file"
  fi
}
check_out --output "$out"; check_out --error "$errf"
[ -z "$out" ] && info "no --output: default is slurm-%j.out in the submit directory (fine, %j is included)"
[ $ARRAY = 1 ] && [ -z "$out" ] && info "array with default output → slurm-%A_%a.out (fine)"
if [ $ARRAY = 1 ]; then
  grep -qE 'SLURM_ARRAY_TASK_ID' <<<"$BODY" || warn "--array set but \$SLURM_ARRAY_TASK_ID is never used: every task would do identical work"
  a=$(val '--array|-a'); [[ $a == *%* ]] || info "no %N throttle on --array=$a; consider --array=${a}%20 to be a good neighbor"
fi

# --- cluster-specific flags -------------------------------------------------------------
has '--exclusive' && { [ "$CLUSTER" = hive ] && err "--exclusive on Hive never schedules (shows a bogus QOSGrpCpuLimit). Remove it" || warn "--exclusive reserves whole nodes; only use if your group owns whole nodes"; }
if [ "$CLUSTER" = franklin ]; then
  g=$(val '--gpus|-G'); [[ $g == *:* ]] && err "Franklin GPUs are untyped: use --gpus=N and pick nodes with --constraint=amd|intel"
fi
if gpu_job; then
  case "$PART" in high|low|bml|bmh|"") [ "$CLUSTER" = franklin ] && [ "$PART" = high ] && err "Franklin high has no GPU nodes; use low or your <group>-gpu partition";; esac
fi

# --- environment inside the script -----------------------------------------------------------
if grep -qE '(^|[;&|[:space:]])module[[:space:]]+(load|purge|unload|add)' <<<"$BODY"; then
  if ! grep -qE 'source[[:space:]]+/etc/profile\.d/modules\.sh|\.[[:space:]]+/etc/profile\.d/modules\.sh' <<<"$BODY" && [[ $first != *"-l"* ]]; then
    warn "uses 'module' but never sources /etc/profile.d/modules.sh (or uses #!/bin/bash -l). Jobs submitted from zsh/cron/OnDemand fail with 'module: command not found'"
  fi
  if grep -qE 'module[[:space:]]+purge' <<<"$BODY" && grep -qE '(^|[[:space:]])(srun|sacct|scontrol|squeue|sbatch)([[:space:]]|$)' <<<"$BODY" && ! grep -qE 'module[[:space:]]+load[[:space:]]+slurm' <<<"$BODY"; then
    warn "module purge removes the slurm module; add 'module load slurm' before using srun/sacct inside the job"
  fi
fi
grep -qE '(conda|mamba)[[:space:]]+activate' <<<"$BODY" && ! grep -qE 'module[[:space:]]+load[[:space:]]+conda' <<<"$BODY" && warn "conda activate without 'module load conda' first (the central conda is not on PATH by default)"
grep -qE '^[[:space:]]*conda[[:space:]]+init|^[[:space:]]*source[[:space:]]+activate' <<<"$BODY" && warn "'conda init' / 'source activate' are not needed and can break the central conda; use module load conda; conda activate ENV"
grep -qE '(^|[[:space:]])(--threads|-t|-p|-@|--cpus|-j|--workers|-n)[[:space:]=]+[0-9]{2,}' <<<"$BODY" && ! grep -q 'SLURM_CPUS_PER_TASK' <<<"$BODY" && info "thread counts look hard-coded; consider \$SLURM_CPUS_PER_TASK so they track --cpus-per-task"
grep -qE 'apptainer[[:space:]]+(build|pull)' <<<"$BODY" && info "apptainer build/pull inside every job is slow; build the .sif once and reuse it"
grep -qE 'apptainer[[:space:]]+(exec|run|shell)' <<<"$BODY" && gpu_job && ! grep -qE 'apptainer[[:space:]]+(exec|run|shell).*--nv' <<<"$BODY" && warn "GPU job runs apptainer without --nv; the container will not see the GPU"
grep -qE 'cd[[:space:]]+\$?TMPDIR|cd[[:space:]]+/tmp|cd[[:space:]]+/scratch' <<<"$BODY" && ! grep -qE '(cp|rsync|mv)[[:space:]].*(/quobyte|/group|\$HOME|~|/nfs)' <<<"$BODY" && warn "works in per-job scratch but never copies results out; /tmp and /scratch are deleted when the job ends"
grep -qE 'rsync.*--delete' <<<"$BODY" && warn "rsync --delete permanently removes extra files on the destination; double-check paths"
grep -qE '(^|[[:space:]])ssh[[:space:]]+[a-z]+-[0-9]' <<<"$BODY" && warn "ssh to compute nodes is not allowed; use srun --jobid=... --overlap --pty bash"
grep -qE '^#SBATCH.*--mail' "$F" && info "--mail-* is set; mail from compute nodes is best-effort"

# --- scheduler validation ------------------------------------------------------------------
if [ $NOTEST = 0 ]; then
  echo
  if command -v sbatch >/dev/null 2>&1; then
    echo "sbatch --test-only $F:"
    out=$(timeout 30 sbatch --test-only "$F" 2>&1); rc=$?
    # Strip the "to start at ..." clause: it is a scheduler-pass artifact, not a queue-wait
    # estimate (identical for 1 and 48 CPUs, often hours ahead of an idle partition), and
    # readers who mistake it for a wait time talk themselves into using the login node.
    sed -E 's/ to start at [^ ]+ [a-z]? ?using/ accepted, using/' <<<"$out" | sed 's/^/   /'
    if [ $rc -eq 124 ]; then
      WARN=$((WARN+1)); echo "   -> sbatch --test-only timed out after 30 s; the scheduler did not answer, so the request is unvalidated. Retry later"
    elif [ $rc -ne 0 ] || grep -qiE 'error|failure|invalid' <<<"$out"; then
      ERR=$((ERR+1)); echo "   -> the scheduler rejects this request; see references/debugging.md for the message"
    else
      echo "   -> accepted; nothing was submitted."
      echo "      (--test-only reports no usable start time; for queue health use"
      echo "       sinfo -p ${PART:-PARTITION} -h -o '%C'  and  squeue -t PD -h -o '%r' | sort | uniq -c)"
    fi
  else
    echo "sbatch not available here; validate on the cluster with: sbatch --test-only $F"
  fi
fi

echo
echo "lint: $ERR error(s), $WARN warning(s) for $F${CLUSTER:+ on $CLUSTER}"
[ $ERR -eq 0 ]
