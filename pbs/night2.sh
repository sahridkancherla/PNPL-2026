#!/bin/bash
# Night 2 (15 -> 16 Oct): submit one stage's jobs chained by PBS dependencies, so that
# training -> holdout prediction -> robust rebuild runs unattended. From the repo root,
# on a login node, after git pull:
#
#     bash pbs/night2.sh today      today's three CPU builds (pull2 with the local check,
#                                   nb6, pull2nb6), and robust8 behind pull2
#     bash pbs/night2.sh configs    K-GPT2 + K-Qwen, W-fixed, All-0 G2 + K2, F0-fixed s1-2:
#                                   train -> predict -> robust, one chain per family
#     bash pbs/night2.sh ssl        the SSL pretraining -> 3 G2 fine-tunes -> predict -> robust
#     bash pbs/night2.sh tf         the temporal-filter front-end: train -> predict -> robust
#     bash pbs/night2.sh conformer  the Conformer encoder: train -> predict -> robust
#     bash pbs/night2.sh tent       TTA on G2/K2/K18 (tent, evaluate, predict) -> robust
#     bash pbs/night2.sh all        LAST: the "everything that passed" rebuild, waiting on
#                                   every predict and tent job the ledger holds
#
# Every job id goes to logs/night2_jobs.txt ("<stage> <what> <id>"). `qstat -u $USER`
# shows the chains (H = held on a dependency). When a training array fails, PBS deletes
# the chain behind it; the morning's `cli.final_status` and the logs say which. Each
# family's file: submissions/ens_t23_<family>_robust_ts_hubh_lone.csv.
#
# ⚠️ Run each stage once: a second run submits its chain again.

set -euo pipefail

cd "$HOME/PNPL-2026"
STAGE="${1:?stage: today | configs | ssl | tf | conformer | tent | all}"
LEDGER="logs/night2_jobs.txt"
NEWMEMBERS="configs/trial23/newmembers.txt"
mkdir -p logs

# One line per job in the ledger, and on the screen.
record() {
    echo "$STAGE $1 $2" >> "$LEDGER"
    echo "[night2] $1 -> $2"
}

# The dependency option on one job or array (PBS Pro: an array id "123[].server" takes
# the same keywords; the dependency is met when the whole array has ended).
after() {
    echo "-W depend=$1:$2"
}

# A family's chain: training array (lines of configs/trial18/runs.txt) -> predict array
# (lines of newmembers.txt) -> the robust rebuild with the family added. Extra qsub
# options for the training (a walltime, a dependency) come after the four arguments.
chain() {
    local family="$1" train_range="$2" predict_range="$3" include="$4"
    shift 4
    local train predict robust
    train=$(qsub "$@" -J "$train_range" pbs/trial18.pbs)
    record "${family}_train" "$train"
    #afterany: one failed seed must not stop the others' predictions; a predict whose
    #checkpoint is missing fails its preflight in a minute
    # shellcheck disable=SC2046
    predict=$(qsub $(after afterany "$train") -J "$predict_range" -v "LIST=${NEWMEMBERS}" pbs/trial23_predict.pbs)
    record "${family}_predict" "$predict"
    #the rebuild leaves out whatever member is missing or fails the sanity filter
    # shellcheck disable=SC2046
    robust=$(qsub $(after afterany "$predict") -l walltime=03:00:00 \
        -v "STEP=robust,NAME=${family},INCLUDE=${include},SANITY=1,SKIP_MISSING=1" pbs/trial23_cpu.pbs)
    record "${family}_robust" "$robust"
}

case "$STAGE" in
    today)
        pull2=$(qsub -v STEP=robust,NAME=pull2,PULL=2,CHECK=20 pbs/trial23_cpu.pbs)
        record pull2 "$pull2"
        nb6=$(qsub -v STEP=robust,NAME=nb6,NEIGHBOURS=6 pbs/trial23_cpu.pbs)
        record nb6 "$nb6"
        pull2nb6=$(qsub -v STEP=robust,NAME=pull2nb6,PULL=2,NEIGHBOURS=6 pbs/trial23_cpu.pbs)
        record pull2nb6 "$pull2nb6"
        # shellcheck disable=SC2046
        robust8=$(qsub $(after afterok "$pull2") -v STEP=robust,NAME=robust8,MERGE=robust+pull2 pbs/trial23_cpu.pbs)
        record robust8 "$robust8"
        ;;
    configs)
        chain kgq 35-40 40-45 'trial18_kg_|trial18_kq_'
        chain wfixed 41-43 46-48 'trial18_wfixed_'
        chain all0 44-49 49-54 'trial18_all0'
        chain f0fam 50-51 55-56 'trial18_f0fixed' -l walltime=08:00:00
        ;;
    ssl)
        pretrain=$(qsub pbs/trial23_ssl.pbs)
        record ssl_pretrain "$pretrain"
        # shellcheck disable=SC2046
        chain sslg2 52-54 57-59 'trial18_sslg2_' $(after afterok "$pretrain")
        ;;
    tf)
        chain tf 55-57 60-62 'trial18_tf_'
        ;;
    conformer)
        chain conf 58-60 63-65 'trial18_conf_' -l walltime=06:00:00
        ;;
    tent)
        tent=$(qsub -J 1-9 pbs/trial23_tent.pbs)
        record tent_tent "$tent"
        # the adapted members in their parents' place
        # shellcheck disable=SC2046
        robust=$(qsub $(after afterok "$tent") -l walltime=03:00:00 \
            -v 'STEP=robust,NAME=tent,INCLUDE=_tent,EXCLUDE=^trial13_g2_s[0-2]$|^trial18_k2_s[0-2]$|^trial18_k18_s[0-2]$,SANITY=1,SKIP_MISSING=1' \
            pbs/trial23_cpu.pbs)
        record tent_robust "$robust"
        ;;
    all)
        # every predict and tent job recorded so far, whatever their outcome
        ids="$(grep -E ' ([a-z0-9]+_predict|tent_tent) ' "$LEDGER" | awk '{print $3}' | tr '\n' ':' | sed 's/:$//')"
        if [ -z "$ids" ]; then
            echo "no predict or tent jobs in ${LEDGER}: run the other stages first" >&2
            exit 1
        fi
        all=$(qsub -W "depend=afterany:${ids}" -l walltime=04:00:00 \
            -v 'STEP=robust,NAME=all,INCLUDE=trial18_kg_|trial18_kq_|trial18_wfixed_|trial18_all0|trial18_f0fixed|trial18_sslg2_|trial18_tf_|trial18_conf_,SANITY=1,SKIP_MISSING=1' \
            pbs/trial23_cpu.pbs)
        record all_robust "$all"
        ;;
    *)
        echo "unknown stage ${STAGE}" >&2
        exit 1 ;;
esac
echo "[night2] stage ${STAGE} submitted; ledger ${LEDGER}"
