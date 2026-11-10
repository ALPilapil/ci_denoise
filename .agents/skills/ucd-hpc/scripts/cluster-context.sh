#!/usr/bin/env bash
# ucd-hpc: one-shot, read-only snapshot of the cluster and the current user's access.
# Safe on a login node (only cheap Slurm/df queries). Exits 0 with a notice when off-cluster.
#
# Usage: cluster-context.sh [--brief]
set -u
BRIEF=0; [ "${1:-}" = "--brief" ] && BRIEF=1
T() { timeout 25 "$@" 2>/dev/null; }
hr() { printf '\n== %s ==\n' "$1"; }

# --- which cluster -----------------------------------------------------------
HPCCF_ROOT=${UCD_HPC_HPCCF_ROOT:-/opt/hpccf}   # overridable so the tests do not pick up the real cluster
CLUSTER=""
if [ -r "$HPCCF_ROOT/etc/cluster_name.conf" ]; then
  CLUSTER=$(. "$HPCCF_ROOT/etc/cluster_name.conf" 2>/dev/null; echo "${CLUSTER_NAME:-}")
fi
# Slurm lives at a fixed path on all three clusters; never search for it.
SLURM_BIN=/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/bin
if ! command -v sinfo >/dev/null 2>&1; then
  # non-login shells may lack the slurm module
  if [ -r /etc/profile.d/modules.sh ]; then
    # `set -u` must be off here: modules.sh dereferences unset variables and would abort us.
    set +u
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh >/dev/null 2>&1 || true
    set -u
  fi
  if ! command -v sinfo >/dev/null 2>&1 && [ -d "$SLURM_BIN" ]; then
    PATH="$SLURM_BIN:$PATH"
  fi
fi
[ -z "$CLUSTER" ] && CLUSTER=$(T scontrol show config | awk '/^ClusterName/{print $3}')
if [ -z "$CLUSTER" ] || ! command -v sinfo >/dev/null 2>&1; then
  cat <<EOF
NOT ON A UC DAVIS HPC LOGIN NODE (host: $(hostname -f 2>/dev/null || hostname)).
Live probing is unavailable. Use references/<cluster>.md and ask the user to run, on the cluster:
  sacctmgr show assoc user=\$USER format=account%20,partition%20,qos%40
  /opt/hpccf/bin/slurm-show-resources.py
EOF
  exit 0
fi

hr "cluster"
echo "cluster=$CLUSTER  host=$(hostname -f 2>/dev/null || hostname)  user=$USER  slurm=$(T sinfo --version | awk '{print $2}')  date=$(date -Is)"
echo "reference: references/${CLUSTER}.md"

# --- the user's access --------------------------------------------------------
hr "accounts / partitions / QOS for $USER (⁺ = default account)"
if [ -x "$HPCCF_ROOT/bin/slurm-show-resources.py" ]; then
  T "$HPCCF_ROOT/bin/slurm-show-resources.py" | sed '/^$/d'
else
  T sacctmgr show assoc user="$USER" format=account%20,partition%20,qos%40
fi

hr "QOS limits: MaxTRES = per job, GrpTRES = whole account pool, MaxTRESPU = per user"
QOS=$(T sacctmgr show assoc user="$USER" format=qos -Pn | sort -u | paste -sd, -)
if [ -n "$QOS" ]; then
  T sacctmgr show qos "$QOS" format=name%36,maxtres%40,grptres%40,maxtrespu%30,maxwall -P \
    | awk -F'|' 'NR==1{print; next} {printf "%s|%s|%s|%s|%s\n",$1,($2==""?"-":$2),($3==""?"-":$3),($4==""?"-":$4),($5==""?"-":$5)}' \
    | column -t -s'|'
fi

hr "maintenance reservations (jobs whose --time reaches into one wait as ReqNodeNotAvail)"
R=$(T scontrol show reservation); rc=$?
if [ "$rc" -ne 0 ]; then echo "query failed (exit $rc)"
elif grep -q ReservationName <<<"$R"; then awk '/ReservationName/{print $1, $2, $3}' <<<"$R" | sed 's/ReservationName=//' | head -5
else echo "none"; fi

# --- partitions ----------------------------------------------------------------
hr "partitions: max time, default mem per CPU, preemption, nodes alloc/idle/other/total, GPUs"
echo "node counts are PER PARTITION; most nodes belong to several, so the column does not sum to"
echo "the cluster size (see the distinct-node total further down)."
printf '%-20s %-12s %-10s %-10s %-16s %s\n' PARTITION MAXTIME DEFMEM/CPU PREEMPT NODES_A/I/O/T GRES_TYPES
for p in $(T sinfo -h -o "%R" | sort -u); do
  info=$(T scontrol show partition "$p")
  mt=$(grep -oE 'MaxTime=[^ ]+' <<<"$info" | cut -d= -f2)
  dm=$(grep -oE 'DefMemPerCPU=[^ ]+' <<<"$info" | cut -d= -f2)
  pm=$(grep -oE 'PreemptMode=[^ ]+' <<<"$info" | cut -d= -f2)
  nodes=$(T sinfo -h -s -p "$p" -o "%F")
  gres=$(T sinfo -h -p "$p" -o "%G" | grep -v '(null)' | sed -E 's/gpu:([^:(]+):?[0-9]*(\([^)]*\))?/\1/; s/\(S:[^)]*\)//g' | sort -u | paste -sd, -)
  printf '%-20s %-12s %-10s %-10s %-16s %s\n' "$p" "${mt:-?}" "${dm:+${dm}M}" "${pm:-?}" "${nodes:-?}" "${gres:--}"
done

hr "GPU inventory (sinfo GRES string | distinct nodes) — use --gpus=TYPE:N or --gpus=N"
# Summing %D is correct here ONLY because %P is absent from the format: sinfo then collapses the
# per-partition rows, so each node is counted once even though most GPU nodes sit in both `low`
# and an owner partition. Do not add %P to this format string without switching to the
# node-name-deduplicated form used for node shapes below, or the counts will double.
T sinfo -h -o "%G|%D" | grep -v '(null)' | sort | awk -F'|' '{a[$1]+=$2} END{for(k in a) print k"|"a[k]}' | sort | column -t -s'|'

if [ "$BRIEF" = 0 ]; then
  hr "node shapes (distinct nodes x features | CPUs | RAM MB)"
  # `sinfo -N` emits one row per node PER PARTITION, so a node in both `low` and `high` appears
  # twice. Deduplicate on the node name (%N) before counting, otherwise every shared node is
  # counted once per partition it belongs to.
  shapes=$(T sinfo -h -N -o "%N|%f|%c|%m" | sort -u -t'|' -k1,1 | cut -d'|' -f2- | sort | uniq -c | sort -rn)
  awk '{n=$1; $1=""; sub(/^ /,""); printf "%s x %s\n", n, $0}' <<<"$shapes" | head -15 | column -t -s'|'
  total=$(awk '{s+=$1} END{print s+0}' <<<"$shapes")
  shown=$(head -15 <<<"$shapes" | awk '{s+=$1} END{print s+0}')
  rows=$(wc -l <<<"$shapes"); hidden=$(( rows > 15 ? rows - 15 : 0 ))
  if [ "$hidden" -gt 0 ]; then
    echo "  ($total distinct nodes total; $shown shown above, $((total-shown)) in $hidden further shape(s))"
  else
    echo "  ($total distinct nodes total)"
  fi
fi

# --- the queue ------------------------------------------------------------------
hr "my jobs"
Q=$(T squeue --me -o "%.30i %.16P %.20j %.3t %.11M %.11l %.5C %.9m %R"); rc=$?
if [ "$rc" -ne 0 ]; then echo "query failed (exit $rc)"
elif [ -n "$(sed -n 2p <<<"$Q")" ]; then echo "$Q"; else echo "none queued or running"; fi

# --- storage ----------------------------------------------------------------------
hr "storage"
echo "home (20 GB quota per user; on Farm/Franklin df shows the shared server, so use du -sh ~):"; T df -h "$HOME" | tail -1 | awk '{printf "  %s used of %s (%s) on %s\n",$3,$2,$5,$6}'
for g in $(id -Gn); do
  for base in /quobyte /group; do
    d="$base/$g"
    if [ -d "$d" ]; then
      echo "group share: $d"
      T df -h "$d" | tail -1 | awk '{printf "  %s used of %s (%s)\n",$3,$2,$5}'
      if command -v qinfo >/dev/null 2>&1 && [ "$base" = /quobyte ]; then
        q=$(T qinfo quota "$d" | head -4 | sed 's/^/  /'); [ -n "$q" ] && echo "$q"
      fi
      [ -d "$d/BACKED-UP" ] && echo "  has BACKED-UP/ (only that directory is backed up)"
    fi
  done
done
case "$CLUSTER" in
  hive) [ -d /nfs/hive/scratch ] && echo "network scratch: /nfs/hive/scratch (shared, not purged, not backed up)";;
esac
echo "per-job local scratch: \$TMPDIR=/tmp (private, deleted at job end)"

# --- login-node caps: do they apply to THIS account? --------------------------------
hr "login-node caps (policy applies to everyone regardless; see SKILL.md)"
LIMSH=/etc/security/systemd-user-limits.sh
if id -nG 2>/dev/null | grep -qw hpccfgrp; then
  echo "$USER is in hpccfgrp -> EXEMPT from the cgroup caps (the PAM script removes them)."
  echo "  Do NOT read that as permission: the policy, and staff killing offending processes,"
  echo "  apply anyway, and the users you are helping ARE capped."
else
  echo "$USER is not in hpccfgrp -> the per-user caps apply to this account."
fi
if [ -r "$LIMSH" ]; then
  vals=$(grep -oE '(CPUQuota|MemoryMax|MemorySwapMax|TasksMax)=[^ \\]+' "$LIMSH" | paste -sd' ' -)
  echo "  from $LIMSH: ${vals:-unparsed - read the file}"
else
  echo "  $LIMSH not readable here; documented values are 2 CPUs / 7.5% RAM / 500M swap / 512 procs"
fi
command -v systemctl >/dev/null 2>&1 && \
  echo "  effective now: $(T systemctl show "user-$(id -u).slice" -p CPUQuotaPerSecUSec -p MemoryMax -p TasksMax | paste -sd' ' -)"
echo "  open files: $(awk '/nofile/{print $4; exit}' /etc/security/limits.d/slurm.conf 2>/dev/null || ulimit -n) (from limits.d/slurm.conf; applies to everyone)"

# --- software --------------------------------------------------------------------
hr "software"
if ! type module >/dev/null 2>&1 && [ -r /etc/profile.d/modules.sh ]; then
  set +u   # modules.sh dereferences unset variables
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh >/dev/null 2>&1 || true
  set -u
fi
if type module >/dev/null 2>&1; then
  echo "module trees:"; tr ':' '\n' <<<"${MODULEPATH:-}" | sed 's/^/  /'
  n=$(module -t avail 2>&1 | grep -vc ':$'); echo "modules available: ~$n  (module avail NAME / module search NAME)"
  echo "conda: $(module -t avail conda/base 2>&1 | grep -m1 'conda/base' || echo 'module not found')  (central envs: module -t avail conda/)"
  echo "python/R/cuda modules: $(module -t avail python R cuda 2>&1 | grep -E '^(python|R|cuda)/' | paste -sd' ' -)"
else
  echo "module command unavailable in this shell (source /etc/profile.d/modules.sh)"
fi
command -v apptainer >/dev/null 2>&1 && echo "apptainer: $(apptainer --version 2>/dev/null)"
echo "OnDemand: https://ondemand.${CLUSTER}.hpc.ucdavis.edu"
