# Phase 6 — Paper, Artifacts, Submission

**Ends at:** GATE-S (PI approves submission and arXiv posting).

---

## P6-01 — Paper skeleton and the fabrication guard goes hard

**Branch:** `phase6/paper-skeleton`

1. `paper/` with the target venue's LaTeX style, `main.tex`, `sections/`, `appendix/`,
   `refs.bib`.
2. **Flip `tools/check_no_orphan_numbers.py` from warning to hard failure in CI**
   (see P0-05). From here on, a number in the paper that does not exist in `results/`
   breaks the build. This is the mechanical enforcement of Rule 1 at the moment it
   matters most.
3. Build `tools/render_numbers.py`: a macro system where the paper writes
   `\result{p1_08.predicted_accuracy_150M}` and the tool substitutes the value from the
   results JSON at build time, with the number never typed into the `.tex` at all. Use
   it everywhere. This single tool eliminates the most common failure mode in
   agent-written papers.
4. `make paper` builds a PDF; CI builds it on every PR touching `paper/`.

---

## P6-02 — Narrative outline

**Branch:** `phase6/outline`

Write `paper/outline.md` before writing prose, and get PI sign-off on the outline
rather than on a full draft — it is far cheaper to redirect at this stage.

Proposed structure:

1. **Intro** — the "decide small, deploy big" workflow is universal and has zero
   guarantees; DataDecide showed it is shaky; nobody has explained why; we do, and we
   fix it.
2. **Setup** — BAI where the reward is extrapolated, not observed (P2-01).
3. **Theorem 1** — the bias/variance decomposition, and the corollary that explains
   DataDecide.
4. **Theorem 2** — lower bound and the impossibility phase transition. *Foreground
   this.* It is the answer to "this is just BAI applied".
5. **Theorem 3** — identifiability and design.
6. **Algorithm** — Extrapolation-Track-and-Stop with abstention.
7. **Experiments** — DataDecide re-analysis (the mechanism), offline replay (the
   savings), live runs (confirmation).
8. **A diagnostic for practitioners** — how to estimate your own `sigma^2_extrap` and
   decide whether to trust extrapolation.
9. **Limitations** — honestly written; see P6-06.

**Framing decision, driven by the Phase 1 memo, made explicitly here:**

- If `sigma^2_extrap` is large → lead with "we explain DataDecide" (mechanism paper).
- If `sigma^2_extrap` is small → lead with the allocation speedup (positive method paper).
- If reversals are pervasive → lead with impossibility and the fallback diagnostic
  (cautionary paper).

Record which framing was chosen and why in `docs/decisions.md`.

---

## P6-03 — Write the theory sections
**Branch:** `phase6/write-theory`

Main-text statements with proof sketches; full proofs to the appendix. Every constant
matches the code. Every theorem has a one-sentence plain-English gloss immediately
after it — the venue is an ML conference, not a statistics journal, and the reviewer who
decides your fate may not be a bandit theorist.

## P6-04 — Write the experiment sections
**Branch:** `phase6/write-experiments`

Every claim points to a figure or table; every figure regenerates from `make figures`.
State the negative results as plainly as the positive ones.

## P6-05 — Write intro, abstract, and conclusion
**Branch:** `phase6/write-intro`

Write these **last**, when you know what the paper actually found. The abstract should
survive being read by someone who reads nothing else.

---

## P6-06 — Limitations, ethics, and reproducibility statements

**Branch:** `phase6/limitations`

Write a limitations section that a hostile reviewer could not improve on. It must
include at minimum:

- The guarantee is conditional on the bias bound `eta`, which must be estimated (P3-06).
- DataDecide is one suite, one target scale, one family of tasks.
- The parametric family assumption, and precisely what the misspecification class does
  and does not cover.
- If Phase 4 was declined or reduced: say exactly what was and was not run.
- Any conclusion that flipped under a P5-05 robustness variation.

Reproducibility statement: repo URL, HF artifact URLs, exact commands, compute used
(real GPU-hours from the Phase 4 spend counter, not an estimate).

---

## P6-07 — Internal review round

**Branch:** `phase6/internal-review`

1. Run the `/code-review` skill over the full repo diff since the start.
2. Ask the PI to nominate one or two external readers; give them the PDF plus a
   one-page "what to attack" note.
3. Self-review against the venue's reviewer guidelines and the three anticipated
   objections from P2-07.
4. Verify every citation: correct venue, correct year, correct arXiv id. The source
   document explicitly flags that several references are workshop papers or recent
   preprints whose final venue may have changed, and that Muennighoff et al. appears as
   both 2023 and 2024 in different reference lists. **Check every one against the
   actual arXiv or proceedings page.** Do not trust the reference list in the source PDF
   or your own memory.

---

## P6-08 — Artifact release

**Branch:** `phase6/artifact-release`

- Flip the GitHub repo public (PI approval required).
- Tag a release matching the submission.
- Finalise HF dataset and model cards; cross-link repo, HF artifacts, and paper.
- `docs/reproduce.md`: Phase 1 reproducible from a clean clone with one command; state
  the wall-clock and disk it needs.
- Include `plan/` in the release so the methodology is transparent.

---

## P6-09 — Novelty sweep #2

**Branch:** `phase6/novelty-sweep-2`

Re-run the P0-07 sweep within two weeks of the deadline. Check OpenReview submissions
for the current cycle, recent arXiv listings, and new citations of DataDecide.

If concurrent work exists, assess honestly and report to the PI with options:
(a) proceed and cite as concurrent, (b) re-frame toward whichever of our four
contributions remains uniquely ours, (c) pivot to a runner-up question from the source
document (operator-theoretic model-collapse identifiability; consistent estimation
theory for data valuation with confidence intervals; prediction-powered identifiable
measurement of LLM capability). **The PI decides, not you.**

---

## GATE-S — Submission approval

Post to the PI: the final PDF, a summary of results including negatives, the artifact
URLs, the novelty verdict, the verified deadline, and a submission checklist.

**Ask explicitly for approval to (a) submit and (b) post to arXiv.** These are
public-facing and irreversible; never do either without a clear yes. Do not create an
OpenReview or arXiv account, upload, or click submit on the PI's behalf unless they
direct it — and even then, confirm the exact final file first.

---

## P6-10 — Submit and archive

**Branch:** `phase6/submit`

Only after GATE-S. Then: archive the exact submitted PDF and the commit SHA in the
repo, and write `docs/postmortem.md` — what took longer than planned, what the compute
actually cost against the GATE-C estimate, and what should be done differently. Future
sessions on the rebuttal will thank you.
