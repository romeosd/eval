# Walkthrough script (customer-demo style, ~10 minutes)

Talk about decisions and trade-offs, not code line by line.

1. **Problem (1 min).** What is decided, what an error costs, and why `REFER` is a
   first-class outcome.
2. **Live run (2 min).** `adjudicator run --split dev --cases dev-002,dev-003` and show
   the report.
3. **Hard case (3 min).** `adjudicator show <run> dev-003`. The deciding rule exists only
   in the flowchart image; show the `view_policy_page` call, the citation of the
   diagram, and the validation passing.
4. **Safeguards (2 min).** `show` on `dev-006`: the injection inside the doctor's letter
   is flagged and the image quarantined, and the decision still uses the medical facts.
   Then `dev-007`: the missing receipt forces a referral.
5. **Evidence (1.5 min).** Comparison table, and one change driven by evaluation with
   before/after numbers.
6. **What I'd improve (0.5 min).** Top two limitations and how to fix them.

Prepare for the live deep-dive with: why this model per role, what happens when the
judge is wrong, cost at 10x volume, how you'd onboard a new policy, and which tests
prove the fail-closed paths.
