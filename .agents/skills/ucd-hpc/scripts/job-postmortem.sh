#!/usr/bin/env bash
# ucd-hpc: explain why a Slurm job is pending, failed, or was killed.
# Read-only. Usage: job-postmortem.sh JOBID [--no-logs]
set -u
J=${1:?usage: job-postmortem.sh JOBID [--no-logs]}
SHOWLOGS=1; [ "${2:-}" = "--no-logs" ] && SHOWLOGS=0
EVID=""
hr() { printf '\n== %s ==\n' "$1"; }
T() { timeout 25 "$@" 2>/dev/null; }
SLURM_BIN=/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/bin   # fixed path on Hive/Farm/Franklin; do not go looking for it
if ! command -v sacct >/dev/null 2>&1; then
  if [ -r /etc/profile.d/modules.sh ]; then
    # `set -u` must be off here: modules.sh dereferences unset variables and would abort us.
    set +u
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh >/dev/null 2>&1 || true
    set -u
  fi
  if ! command -v sacct >/dev/null 2>&1 && [ -d "$SLURM_BIN" ]; then
    PATH="$SLURM_BIN:$PATH"
  fi
fi
command -v sacct >/dev/null 2>&1 || { echo "sacct not found: run this on a cluster login node (Slurm is at $SLURM_BIN)"; exit 2; }

# ---- live controller view (only while the job is still in memory) --------------
# A bare array or het-job id returns several records; the one that matches sacct's main row is picked below.
LIVE_ALL=$(T scontrol show job "$J")
FMT=JobID,JobName%28,State,ExitCode,Reason,Partition,Account,QOS%22,Submit,Start,End,Elapsed,Timelimit,ReqCPUS,AllocCPUS,ReqMem,MaxRSS,TotalCPU,NodeList,WorkDir,StdOut,StdErr,JobIDRaw,User
ACC=$(T sacct -j "$J" -P --format="$FMT")

# main record = the line whose JobID is exactly J (or J_arraytask)
MAIN=$(awk -F'|' -v j="$J" 'NR>1 && ($1==j || $1 ~ "^"j"_[0-9]+$" || $1 ~ "^"j"\\+") {print; exit}' <<<"$ACC")
[ -z "$MAIN" ] && MAIN=$(sed -n 2p <<<"$ACC")
IFS='|' read -r JID JNAME STATE EXIT REASON PART ACCT QOS SUBMIT START END ELAPSED TLIM REQCPUS ALLOCCPUS REQMEM MAXRSS TOTALCPU NODES WORKDIR SOUT SERR RAW SUSER <<<"$MAIN"
STATE0=${STATE%% *}

# live record for JID: array task = ArrayJobId+ArrayTaskId, het component = HetJobId+HetJobOffset, else JobId
want_a="JobId=${JID:-$J}"; want_b=""
case "$JID" in
  *_*) want_a="ArrayJobId=${JID%%_*}"; t=${JID#*_}; t=${t#[}; want_b="ArrayTaskId=${t%]}";;
  *+*) want_a="HetJobId=${JID%%+*}"; want_b="HetJobOffset=${JID#*+}";;
esac
# recs 1 = the scontrol record for JID, empty if none matches (e.g. the task already left the controller; a
#          placeholder's StdOut would then name the wrong log);
# recs 2 = the other records worth showing, one per line: array placeholders (pending ranges), other het components
recs() { awk -v RS= -v a="$want_a" -v b="$want_b" -v mode="$1" '
  { n=split($0,t,/[[:space:]]+/); fa=0; fb=(b==""); oth[NR]=0
    for(i=1;i<=n;i++){ if(t[i]==a)fa=1; if(t[i]==b)fb=1
      if((t[i] ~ /^ArrayTaskId=/ && t[i] !~ /^ArrayTaskId=[0-9]+$/) || t[i] ~ /^HetJobId=/)oth[NR]=1 }
    r[NR]=$0; hit[NR]=(fa&&fb) }
  END{ sel=0; for(i=1;i<=NR;i++) if(hit[i]){sel=i;break}
       if(mode==1){ if(sel) print r[sel] } else for(i=1;i<=NR;i++) if(i!=sel && oth[i]){ gsub(/[[:space:]]+/," ",r[i]); print r[i] } }' <<<"$LIVE_ALL"; }
fields() { grep -oE '(JobState|Reason|Dependency|Priority|Partition|Account|QOS|Restarts|TimeLimit|RunTime|StartTime|SubmitTime|NumNodes|NumCPUs|CPUs/Task|TRES|NodeList|BatchHost|Command|WorkDir|StdOut|StdErr|Features|MinMemoryCPU|MinMemoryNode|ExitCode)=[^ ]*' | paste -sd' ' - | fold -s -w 110; }
LIVE=$(recs 1); LIVE_OTHER=$(recs 2)
if [ -z "$LIVE" ] && [ -z "$LIVE_OTHER" ] && [ -n "$LIVE_ALL" ] && [ -z "$JID" ]; then LIVE=$LIVE_ALL; fi
if [ -n "$LIVE" ] || [ -n "$LIVE_OTHER" ]; then
  hr "scontrol show job ${JID:-$J} (live)"
  if [ -n "$LIVE" ]; then fields <<<"$LIVE"; else echo "(${JID:-$J} itself is no longer in the controller)"; fi
  [ -n "$LIVE_OTHER" ] && while IFS= read -r r; do
    echo "-- also $(grep -oE '(^| )(JobId|ArrayTaskId|HetJobOffset)=[^ ]+' <<<"$r" | sed 's/^ //' | paste -sd' ' -):"
    fields <<<"$r"
  done <<<"$LIVE_OTHER"
fi

# ---- accounting -------------------------------------------------------------------
hr "sacct"
[ -z "$(sed -n 2p <<<"$ACC")" ] && { echo "sacct knows nothing about job $J (wrong id, wrong cluster, or too old)."; exit 1; }
echo "$ACC" | cut -d'|' -f1-8,12,13,16,17,19 | column -t -s'|'

# peak RSS over all steps, in MB
to_mb() { local v=${1%[KMGT]}; case "$1" in *K) awk "BEGIN{printf \"%.0f\", $v/1024}";; *M) printf '%s' "${v%.*}";; *G) awk "BEGIN{printf \"%.0f\", $v*1024}";; *T) awk "BEGIN{printf \"%.0f\", $v*1024*1024}";; '') echo 0;; *) awk "BEGIN{printf \"%.0f\", $1/1024/1024}";; esac; }
PEAK=0
while IFS='|' read -r rss; do [ -n "$rss" ] && { m=$(to_mb "$rss"); [ "${m:-0}" -gt "$PEAK" ] 2>/dev/null && PEAK=$m; }; done < <(awk -F'|' 'NR>1{print $17}' <<<"$ACC")
# requested memory in MB (ReqMem like 32G, 4000M, 2Gc = per cpu, 2Gn = per node)
RM=${REQMEM%[cn]}; REQ_MB=$(to_mb "$RM")
case "$REQMEM" in *c) REQ_MB=$(( REQ_MB * ${ALLOCCPUS:-${REQCPUS:-1}} ));; esac

hr "summary"
echo "job $JID ($JNAME): state=$STATE exit=$EXIT partition=$PART account=$ACCT qos=$QOS"
echo "time: elapsed=$ELAPSED of limit=$TLIM   cpus=$ALLOCCPUS   nodes=$NODES"
if [ "${PEAK:-0}" -gt 0 ] && [ "${REQ_MB:-0}" -gt 0 ]; then
  echo "memory: peak RSS ${PEAK} MB of ${REQ_MB} MB requested ($(( PEAK * 100 / REQ_MB ))%)"
elif [ -n "$REQMEM" ]; then
  echo "memory: requested $REQMEM (no RSS recorded; job may not have run)"
fi
# CPU efficiency: TotalCPU (D-HH:MM:SS or MM:SS.mmm) vs Elapsed*AllocCPUS
to_sec() { local s=${1%.*} d=0; [[ $s == *-* ]] && { d=${s%%-*}; s=${s#*-}; }; IFS=: read -r a b c <<<"$s"; if [ -z "$c" ]; then c=$b; b=$a; a=0; fi; echo $(( 10#${d}*86400 + 10#${a:-0}*3600 + 10#${b:-0}*60 + 10#${c:-0} )); }
ES=$(to_sec "${ELAPSED:-0}"); CS=$(to_sec "${TOTALCPU:-0}")
if [ "$ES" -gt 0 ] && [ "${ALLOCCPUS:-0}" -gt 0 ]; then
  echo "cpu efficiency: $(( CS * 100 / (ES * ALLOCCPUS) ))%  (TotalCPU $TOTALCPU over $ALLOCCPUS CPUs × $ELAPSED)"
fi

# ---- diagnosis ---------------------------------------------------------------------
hr "diagnosis"
case "$STATE0" in
  PENDING)
    R=${REASON:-$(grep -oE 'Reason=[^ ]+' <<<"$LIVE" | cut -d= -f2)}
    echo "Job is still waiting. Reason: ${R:-unknown}"
    case "$R" in
      Priority) echo "Others are ahead in the queue. Smaller/shorter requests backfill sooner. Check standing with: sshare -U";;
      Resources) echo "At the front of the queue; waiting for CPUs/memory/GPUs to free up.";;
      QOSGrp*|AssocGrp*) echo "The account's purchased pool (or the free-tier pool) is fully used by group members' jobs. Wait, coordinate, or use --partition=low (preemptible). On Hive this also appears for --exclusive: remove that flag.";;
      QOSMax*PerUser*|AssocMax*) echo "Per-user cap on this QOS. Wait for your own jobs to finish or use another account.";;
      JobArrayTaskLimit) echo "Array throttle (%N) is working as intended.";;
      Dependency) echo "Waiting on a --dependency job.";;
      DependencyNeverSatisfied) echo "The parent job failed; this will never start. scancel $J and resubmit after fixing the parent.";;
      *ReqNodeNotAvail*|*Reserved*maintenance*) echo "Requested time runs into a maintenance reservation or the nodes are down. Shorten --time or wait. scontrol show reservation";;
      BadConstraints) echo "--constraint matches no node in this partition. sinfo -N -p $PART -o '%N %f'";;
      JobHeld*) echo "Held. scontrol release $J (ask support if admin-held).";;
      *launch*failed*) echo "A node failed to start the job; Slurm requeued and held it. scontrol release $J to retry; report if it repeats.";;
      Invalid*) echo "Account/QOS association changed. sacctmgr show assoc user=\$USER format=account%20,partition%20,qos%40";;
    esac;;
  RUNNING) echo "Still running. Inspect it: srun --jobid=$J --overlap --pty bash -l   (then top -u \$USER, nvidia-smi)";;
  OUT_OF_MEMORY|*OOM*) echo "KILLED FOR MEMORY. Peak RSS was ${PEAK} MB against ${REQ_MB} MB requested (and the true peak may be higher than sampled). Raise --mem / --mem-per-cpu to ~1.3× the real need; for multi-node jobs --mem is per node. If the log shows 'CUDA out of memory' the GPU ran out instead: smaller batch or bigger GPU.";;
  TIMEOUT) echo "HIT THE TIME LIMIT ($TLIM). Raise --time up to the partition max ($(T scontrol show partition "$PART" | grep -oE 'MaxTime=[^ ]+' | cut -d= -f2)), move to a longer partition, or checkpoint/split the work.";;
  PREEMPTED|REQUEUED) echo "PREEMPTED: a higher-priority (owner) job needed the node; this happens on low/bml. The job restarts from the top when requeued. Use your group's high partition or make the job restart-safe.";;
  NODE_FAIL) echo "The node failed. Resubmit (Slurm may already have requeued it). If it repeats, report the node ($NODES) to hpc-help@ucdavis.edu.";;
  CANCELLED*) who=$(grep -oE 'by [0-9]+' <<<"$STATE" | awk '{print $2}'); if [ "$who" = 0 ]; then echo "Cancelled by root/staff (policy, emergency, or maintenance). Check email and MOTD."; elif [ -n "$who" ]; then echo "Cancelled by uid $who ($(getent passwd "$who" | cut -d: -f1))."; else echo "Cancelled (by the user or a script)."; fi;;
  FAILED)
    code=${EXIT%%:*}; sig=${EXIT##*:}
    case "$code:$sig" in
      127:*) echo "EXIT 127 'command not found': the program is not on PATH. Usually a missing 'module load', a conda env not activated, or the 'module' command itself missing (add: source /etc/profile.d/modules.sh). Check: module avail NAME";;
      126:*) echo "EXIT 126 'permission denied': chmod +x the script/binary or check its shebang/interpreter.";;
      0:9|137:*) echo "SIGKILL: killed externally. Very often the cgroup OOM killer (check log for oom-kill) or scancel.";;
      0:11|139:*) echo "SEGFAULT: program crash. On Franklin/Hive also caused by running a +amd/zen2-optimized build on the wrong CPU; use the generic build or add --constraint.";;
      0:15|143:*) echo "SIGTERM: terminated (time limit, preemption, or scancel). See log for 'CANCELLED AT ... DUE TO'.";;
      1:*|2:*) echo "The program returned exit $code (its own error). Read stderr below; reproduce interactively with srun --pty.";;
      *) echo "Exit code $EXIT (program:signal). Read the logs below.";;
    esac;;
  COMPLETED) echo "Exited 0. If results are wrong/missing, add 'set -euo pipefail' so intermediate failures surface, and check the log below.";;
  *) echo "State $STATE. Read the logs below.";;
esac

# ---- logs ----------------------------------------------------------------------------
[ "$SHOWLOGS" = 0 ] && exit 0
hr "log files"
# scontrol has the fully expanded paths while the job is live; afterwards expand sacct's stored pattern.
# Each stream falls back on its own (a job may set only --error). Patterns we cannot expand are dropped.
expand() {
  local p=$1 task=4294967294   # Slurm's %a for a job that is not an array
  [[ $JID == *_* ]] && task=${JID#*_}
  [ -n "$p" ] && [ "${p:0:1}" != / ] && p=$WORKDIR/$p
  p=${p//\%A/"${JID%%_*}"}; p=${p//\%a/"$task"}; p=${p//\%j/"$RAW"}; p=${p//\%x/"$JNAME"}; p=${p//\%u/"$SUSER"}
  [[ $p == *%* ]] && p=; echo "$p"
}
OUT=$(grep -oE 'StdOut=[^ ]+' <<<"$LIVE" | cut -d= -f2-); ERR=$(grep -oE 'StdErr=[^ ]+' <<<"$LIVE" | cut -d= -f2-)
[ -z "$OUT" ] && OUT=$(expand "$SOUT"); [ -z "$ERR" ] && ERR=$(expand "$SERR")
CANDS=()
[ -n "$OUT" ] && CANDS+=("$OUT"); [ -n "$ERR" ] && [ "$ERR" != "$OUT" ] && CANDS+=("$ERR")
if [ ${#CANDS[@]} -eq 0 ] && [ -n "$WORKDIR" ] && [ -d "$WORKDIR" ]; then
  base=${JID%%_*}
  while IFS= read -r f; do CANDS+=("$f"); done < <(ls -td "$WORKDIR"/*"$base"* "$WORKDIR"/slurm-"$base".out 2>/dev/null | awk -v b="$base" '!s[$0]++ { n=split($0,a,"/"); if (a[n] ~ "(^|[^0-9])" b "([^0-9]|$)") print }' | while IFS= read -r f; do [ -f "$f" ] && echo "$f"; done | head -4)
fi
PAT='oom-kill|oom_kill|out of memory|OutOfMemory|CANCELLED AT|DUE TO TIME LIMIT|DUE TO PREEMPTION|command not found|No such file|Permission denied|Segmentation fault|Illegal instruction|Killed|Disk quota exceeded|No space left|ModuleNotFoundError|ImportError|CUDA (error|out of memory)|Traceback|[Ee]rror:|ERROR'
if [ ${#CANDS[@]} -eq 0 ]; then
  echo "No log file found. Default is slurm-JOBID.out in the submit dir (${WORKDIR:-unknown}); check the script's --output/--error."
else
  for f in "${CANDS[@]}"; do
    echo
    [ -r "$f" ] || { echo "-- $f (not readable/found)"; continue; }
    total=$(wc -l <"$f")
    # HPCCF's epilog appends a "Job N summary"/"Job N info" block to stdout; show the program's own output separately
    cut=$(grep -nE '^#+ Job [0-9_]+ (summary|info)' "$f" | head -1 | cut -d: -f1)
    if [ -n "$cut" ]; then
      prog=$(( cut - 1 )); echo "-- $f ($prog lines of program output + Slurm epilog summary)"
      epi=$(sed -n "${cut},\$p" "$f" | grep -E '^(State|ExitCode|Reserved walltime|Used walltime|Used CPU time|Mem reserved|Max Mem used|Nodes|Cores|GPUs) ' | sed 's/^/   epilog: /')
      # sacct often lacks MaxRSS for short jobs; the epilog's "Max Mem used" fills the gap
      em=$(grep -oE '^Max Mem used *: *[0-9.]+[KMGT]' <<<"$(sed -n "${cut},\$p" "$f")" | grep -oE '[0-9.]+[KMGT]$')
      if [ -n "$em" ] && [ "${PEAK:-0}" -eq 0 ]; then PEAK=$(to_mb "$em"); echo "   (peak memory from epilog: $em of ${REQMEM:-?} requested)"; fi
      [ -n "$epi" ] && echo "$epi"
    else
      prog=$total; echo "-- $f ($total lines)"
    fi
    hits=$(head -n "$prog" "$f" | grep -nE -i "$PAT" | tail -12)
    [ -n "$hits" ] && { echo "   notable lines:"; sed 's/^/   /' <<<"$hits"; }
    if [ "$prog" -le 17 ]; then
      echo "   program output:"; head -n "$prog" "$f" | sed 's/^/   | /'
    else
      echo "   program output (first 5 / last 12 lines):"
      head -n 5 "$f" | sed 's/^/   | /'; echo "   | ..."
      head -n "$prog" "$f" | tail -n 12 | sed 's/^/   | /'
    fi
    # feed evidence back into the diagnosis
    grep -qiE 'oom-kill|oom_kill|out of memory' <<<"$hits" && EVID="$EVID oom"
    grep -qi 'command not found' <<<"$hits" && EVID="$EVID cmd404"
    grep -qi 'DUE TO TIME LIMIT' <<<"$hits" && EVID="$EVID timeout"
    grep -qi 'DUE TO PREEMPTION' <<<"$hits" && EVID="$EVID preempt"
    grep -qi 'Disk quota exceeded' <<<"$hits" && EVID="$EVID quota"
    grep -qi 'No space left' <<<"$hits" && EVID="$EVID nospace"
    grep -qi 'CUDA out of memory' <<<"$hits" && EVID="$EVID cudaoom"
    grep -qiE 'ModuleNotFoundError|ImportError' <<<"$hits" && EVID="$EVID pyimport"
    grep -qi 'Illegal instruction' <<<"$hits" && EVID="$EVID illegal"
  done
fi

hr "verdict"
case "$STATE0" in OUT_OF_MEMORY|TIMEOUT|PREEMPTED|NODE_FAIL|CANCELLED*|PENDING|RUNNING) ;; *)
  [[ $EVID == *cmd404* ]] && echo "* Log shows 'command not found': a program was not on PATH. The job kept going (no 'set -e'), so Slurm reports $STATE0. Add 'source /etc/profile.d/modules.sh' and 'module load NAME/VERSION' (check: module avail NAME), and add 'set -euo pipefail' so such failures stop the job."
  [[ $EVID == *oom* ]] && echo "* Log shows an oom-kill: a step exceeded the memory request (${REQ_MB} MB). Raise --mem/--mem-per-cpu; peak sampled RSS was ${PEAK} MB, real peak was higher."
  [[ $EVID == *timeout* ]] && echo "* Log shows 'DUE TO TIME LIMIT'. Raise --time or checkpoint."
  [[ $EVID == *preempt* ]] && echo "* Log shows preemption: the job was displaced from a scavenger partition."
  ;;
esac
[[ $EVID == *quota* ]] && echo "* 'Disk quota exceeded': the 20 GB home is full. Write to group storage (/quobyte or /group) and move conda/pip/apptainer caches there (references/storage.md)."
[[ $EVID == *nospace* ]] && echo "* 'No space left on device': per-job /tmp or the target share is full. Check df -h on the path."
[[ $EVID == *cudaoom* ]] && echo "* 'CUDA out of memory': GPU memory, not --mem. Reduce batch size or use a GPU with more memory (--gpus=TYPE:1)."
[[ $EVID == *pyimport* ]] && echo "* Python import error: wrong or unactivated conda environment (module load conda; conda activate ENV)."
[[ $EVID == *illegal* ]] && echo "* 'Illegal instruction': binary built for a newer CPU; add --constraint (e.g. zen3|zen4) or use a generic build."
[ -z "$EVID" ] && case "$STATE0" in COMPLETED) echo "* No fatal messages found in the logs; the job ran to completion. If results are wrong, the problem is in the program's logic or inputs.";; *) echo "* See the diagnosis above; the logs add nothing further.";; esac
case "$STATE0" in OUT_OF_MEMORY|TIMEOUT|PREEMPTED|NODE_FAIL|CANCELLED*|PENDING|RUNNING) [ -n "$EVID" ] && echo "* Log evidence ($EVID) is consistent with the diagnosis above.";; esac
exit 0
