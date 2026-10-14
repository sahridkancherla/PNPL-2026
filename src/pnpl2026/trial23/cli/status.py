"""What is done, queued or running, and still to submit, for trial 23's steps 1-3.

    uv run python -m src.pnpl2026.trial23.cli.status [--pending]

Run on a CX3 login node from the repo root. Each item counts as DONE when
the file its job writes LAST exists (a walltime kill never leaves it); else
QUEUED or RUNNING when a live PBS job will make it (read from `qstat -f`:
the job's name, array index, and its -v variables RUN, LIST, STEP,
RECENTRE, PROTOCOL); else TODO, with the command that submits it.
`--pending` hides the DONE lines. Reads files and qstat only: seconds.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
from pathlib import Path

from ...config import REPO_ROOT

#PBS states of a job that will still run; X and F are finished
LIVE_STATES = {"Q": "QUEUED", "H": "QUEUED", "W": "QUEUED", "B": "RUNNING", "R": "RUNNING", "E": "RUNNING", "S": "QUEUED"}

#the seed wave: (job script, its runs.txt line, the run it trains)
SEED_RUNS = ([("trial13", 13 + i, f"trial13_g1_s{3 + i}") for i in range(3)]
             + [("trial13", 16 + i, f"trial13_g2_s{3 + i}") for i in range(3)]
             + [("trial16", 7 + i, f"trial16_kn_s{3 + i}") for i in range(3)]
             + [("trial18", 25 + i, f"trial18_n_s{3 + i}") for i in range(3)]
             + [("trial18", 28 + i, f"trial18_k2_s{3 + i}") for i in range(3)]
             + [("trial18", 31 + i, f"trial18_k18_s{3 + i}") for i in range(3)]
             + [("trial15", 13 + i, f"trial15_i3_s{3 + i}") for i in range(3)])


def list_lines(path):
    """A list file's non-comment lines, as the job scripts number them.

    ARGS:
        path (str): repo-relative

    RETURNS:
        list of str
    """
    lines = (REPO_ROOT / path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def parse_qstat_full(text):
    """`qstat -f` output as one dict per job; wrapped values rejoined.

    ARGS:
        text (str)

    RETURNS:
        list of dict: id, plus every attribute
    """
    jobs, current, key = [], None, None
    for line in text.splitlines():
        if line.startswith("Job Id:"):
            current = {"id": line.split(":", 1)[1].strip()}
            jobs.append(current)
            key = None
        elif current is not None and " = " in line and not line.startswith("\t"):
            key, value = line.strip().split(" = ", 1)
            current[key] = value
        elif current is not None and key and line.startswith("\t"):
            #a long value wraps onto tab-indented lines
            current[key] += line.strip()
    return jobs


def live_jobs():
    """Every queued or running job of this user, array subjobs one by one.

    RETURNS:
        list of (job name, array index or None, -v variables dict, state word)
    """
    try:
        listing = subprocess.run(["qstat", "-u", os.environ.get("USER", "")], capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        print("(no qstat here: job states unknown)")
        return []
    #the job ids, without the server suffix the listing may truncate
    bases = sorted({match.group(1) + ("[]" if match.group(2) else "") for match in
                    (re.match(r"^(\d+)(\[\])?", line) for line in listing.splitlines()) if match})
    if not bases:
        return []
    full = subprocess.run(["qstat", "-f", "-t", *bases], capture_output=True, text=True, check=False).stdout
    jobs = []
    for job in parse_qstat_full(full):
        state = LIVE_STATES.get(job.get("job_state", ""))
        index = re.search(r"\[(\d+)\]", job["id"])
        #an array's parent stands for its subjobs, listed separately
        if state is None or "[]" in job["id"]:
            continue
        variables = dict(item.split("=", 1) for item in job.get("Variable_List", "").split(",") if "=" in item)
        jobs.append((job.get("Job_Name", ""), int(index.group(1)) if index else None, variables, state))
    return jobs


def job_state(jobs, name, index=None, **wanted):
    """The state of a live job matching a script, a line, and -v variables, or None.

    ARGS:
        jobs (list): from `live_jobs`
        name (str): the job name (#PBS -N)
        index (int or None): the list line: the array index, or -v RUN
        wanted: -v variables that must match; a value None means unset

    RETURNS:
        str or None
    """
    for job_name, array_index, variables, state in jobs:
        line = array_index if array_index is not None else (int(variables["RUN"]) if variables.get("RUN", "").isdigit() else None)
        if job_name != name or (index is not None and line != index):
            continue
        if all(variables.get(key) == value for key, value in wanted.items()):
            return state
    return None


def exists(pattern):
    """Whether a path, or a glob under the home folder or repo, exists.

    ARGS:
        pattern (str)

    RETURNS:
        bool
    """
    expanded = os.path.expanduser(pattern)
    path = expanded if os.path.isabs(expanded) else str(REPO_ROOT / expanded)
    return bool(glob.glob(path))


def report(label, done, state, command, counts):
    """One item's line; counts tallies DONE / live / TODO.

    ARGS:
        label (str), done (bool), state (str or None), command (str), counts (dict)
    """
    word = "DONE" if done else (state or "TODO")
    counts[word] = counts.get(word, 0) + 1
    #--pending: only what is not done yet
    if word == "DONE" and "--pending" in sys.argv:
        return
    print(f"  {word:8} {label}" + (f"   ->  {command}" if word == "TODO" else ""))


def main():
    """Steps 1-3, item by item."""
    jobs = live_jobs()
    hub = os.environ.get("HF_HOME", "~/.cache/huggingface") + "/hub"
    counts = {}
    members, specialists = list_lines("configs/trial23/members.txt"), [line.split()[1] for line in list_lines("configs/trial23/specialists.txt")]

    print("Step 1 -- login node (no jobs):")
    login = [("training scaler", "data/trial23_training_scaler.npz", "uv run python -m src.pnpl2026.trial23.cli.training_scaler"),
             ("testbed check", "data/trial23_testbed_map.npz", "uv run python -m src.pnpl2026.trial23.cli.testbed_check"),
             ("MEG-XL clone", "~/MEG-XL/brainstorm/models/criss_cross_transformer.py",
              'git clone https://github.com/neural-processing-lab/MEG-XL "$HOME/MEG-XL"'),
             ("einops aside", "~/megxllibs/einops", 'uv pip install --python .venv/bin/python --target "$HOME/megxllibs" einops'),
             ("MEG-XL weights", f"{hub}/models--pnpl--MEG-XL/snapshots/*/meg-xl-med-v2.ckpt", "the hf_hub_download line"),
             ("fine calibration", f"{hub}/datasets--pnpl--LibriBrain/snapshots/*/metadata/neo/sss_cal.dat", "the hf_hub_download line"),
             ("GPT-2 medium", f"{hub}/models--openai-community--gpt2-medium/snapshots/*/model.safetensors", "the per-file download loop")]
    for label, path, command in login:
        report(label, exists(path), None, command, counts)

    print("Step 2 -- jobs:")
    for script, line, run in SEED_RUNS:
        report(f"seed {run}", exists(f"runs/{run}/evaluation.json"), job_state(jobs, f"pnpl-{script}", line),
               f"qsub -v RUN={line} pbs/{script}.pbs", counts)
    for index, member in enumerate(members, start=1):
        done = all(exists(f"runs/{member}/embeddings/chains38_{part}.npz") for part in ("validation", "test"))
        state = job_state(jobs, "pnpl-trial23-embed", index, RECENTRE=None, PROTOCOL=None, LIST=None)
        report(f"chains38 dumps {member}", done, state, f"qsub -v RUN={index} pbs/trial23_embed.pbs", counts)
    for index, run in enumerate(specialists, start=1):
        done = exists(f"runs/{run}/finetune.json") and exists(f"submissions/{run}_in_context_ts.csv")
        report(f"P6 specialist {run}", done, job_state(jobs, "pnpl-trial23-finetune", index),
               f"qsub -v RUN={index} pbs/trial23_finetune.pbs", counts)
    report("P5 GPT-2 counts", exists("data/trial23_lm_transitions.npz"), job_state(jobs, "pnpl-trial23-lm"), "qsub pbs/trial23_lm.pbs", counts)
    report("P8b reference", exists("data/trial23_recentre_reference.npz"), job_state(jobs, "pnpl-trial23-cpu", STEP="recentre_reference"),
           "qsub -v STEP=recentre_reference pbs/trial23_cpu.pbs", counts)

    print("Step 3 -- jobs:")
    for step, output in (("testbed", "data/trial23_testbed_settings.json"), ("boundary_diagnose", "runs/trial23_boundary_diagnose.md"),
                         ("sequence_prior_check", "data/trial23_sequence_prior_settings.json")):
        report(step, exists(output), job_state(jobs, "pnpl-trial23-cpu", STEP=step), f"qsub -v STEP={step} pbs/trial23_cpu.pbs", counts)
    for index, member in enumerate(members, start=1):
        done = all(exists(f"runs/{member}/embeddings/chains38_{part}_rc.npz") for part in ("validation", "test"))
        report(f"re-centred dumps {member}", done, job_state(jobs, "pnpl-trial23-embed", index, RECENTRE="1"),
               f"qsub -v RUN={index},RECENTRE=1 pbs/trial23_embed.pbs", counts)

    print("Totals: " + ", ".join(f"{word} {count}" for word, count in sorted(counts.items())))


if __name__ == "__main__":
    main()
