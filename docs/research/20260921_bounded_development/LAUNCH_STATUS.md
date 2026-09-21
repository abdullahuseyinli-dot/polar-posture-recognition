# Bounded phase launch record

Date: 21 September 2026. The hidden, detached queue was launched at **04:22 London
time**. Its actual supervisor PID is **1612**; the Windows venv launcher PID is
4812. This is a launch record, not a claim that the twelve production fits are done.

Run directory: `C:\Users\DELL\polar-posture-recognition\.runs\bounded_adaptation_20260921_0406`.

The [plan](PLAN.md) defines four engineering smokes, twelve fixed production fits,
and one comparison job. Previous results stay read-only and test evaluation is
outside this queue.

| Check | Result |
| --- | --- |
| Full test suite | 214 passed, zero failures/skips; four existing scikit-learn deprecation warnings |
| Lint/style | Pass; no new findings |
| Historical evidence | All 330 protected files unchanged |
| Parent evidence | 21 completed markers and 104 bound artifacts verified; all candidate scores and fusion arithmetic replayed |
| New documentation | 12 local links and all 18-node/18-edge graph endpoints checked |
| Foundation initialization | Original pinned weight/config/processor hashes verified for both models; no missing backbone weights |
| CUDA startup checks | All four model/task optimizer/checkpoint smokes passed; maximum probability replay difference 0.0, with identical image IDs, labels and predictions |
| Current phase at handoff | First full nine-class SigLIP2 fit, seed 42; supervisor job 5/17 |
| Power | Connected, battery 100%; current plan already disables idle sleep; no settings changed |
| Test evaluation | Not performed or included in the queue |

Source and protocol were locked before launch. Canonical queue-plan SHA256:
`09777ded1f308195a3511e77f19c80dabbb33c985f1ead434a54bf76feb80370`.

The supervisor serializes all GPU work, writes durable logs, enforces per-job
deadlines and stops on any failed job. Do not edit locked source or start a second
queue while this one is active. Keep the machine powered and ventilated.

Recheck on request:

```powershell
Get-Content .runs/bounded_adaptation_20260921_0406/queue_status.json
Get-Content .runs/bounded_adaptation_20260921_0406/queue_receipts.json
```

Per-fit progress is under `polar{4,9}/{model}/seed{42,52,62}/progress.json`; logs
are under `logs/`. The final `bounded_summary.json` contains every candidate,
its acceptance checks and either a development finalist or the unchanged incumbent.

Top-level logs: `supervisor.stdout.log` and `supervisor.stderr.log`. Full test
receipt: `validation/pytest.xml`. The machine-readable fixed choices are in
`queue_plan.json` and `parent_evidence_lock.json`, with the public protocol linked
from the plan.

Long production training is left unattended. The next user-requested check can
inspect progress or the final development summary. A separate reviewed final-fit
and test-comparison lock is still required afterward; no SOTA claim is established
by this launch or by selected validation scores.
