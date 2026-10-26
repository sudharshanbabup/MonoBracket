"""
splits.py
---------
Tool-level (group) splitting and low-budget label sampling.

Two rules are enforced throughout:

1. No window from a physical tool may appear in both train and test.  Windows
   from the same cutting edge are highly correlated, so a random window split
   would inflate every metric by tens of points.

2. Labels are purchased at the *cut / run* level, not the window level.  In a
   real shop a metrologist stops the machine and measures VB for one pass;
   they never label individual 4 s windows.  The label budget is therefore the
   fraction of measured passes, and all windows of a measured pass inherit it.
"""

from __future__ import annotations

import numpy as np


def group_holdout(groups: np.ndarray, n_test_groups: int, seed: int,
                  stages: np.ndarray | None = None, n_classes: int = 4,
                  max_tries: int = 200):
    """
    Randomly hold out `n_test_groups` whole tools for testing.

    If `stages` is given, the draw is repeated until every wear stage is present
    on both sides of the split; a fold in which a stage is missing from the test
    set makes macro-F1 undefined for that stage and would penalise all methods
    arbitrarily.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    best = None
    for _ in range(max_tries):
        perm = rng.permutation(uniq)
        test_g = perm[:n_test_groups]
        te = np.isin(groups, test_g)
        if stages is None:
            best = (test_g, te)
            break
        ok = (np.bincount(stages[te], minlength=n_classes) > 0).all() and \
             (np.bincount(stages[~te], minlength=n_classes) > 0).all()
        if ok:
            best = (test_g, te)
            break
        if best is None:
            best = (test_g, te)
    test_g, te = best
    return np.flatnonzero(~te), np.flatnonzero(te), test_g


def leave_one_group_out(groups: np.ndarray, held: int):
    te = groups == held
    return np.flatnonzero(~te), np.flatnonzero(te)


def sample_labels(run_ids: np.ndarray, stages: np.ndarray, frac: float,
                  seed: int, min_per_class: int = 1, budget_tol: float = 1.25):
    """
    Choose a subset of *runs* to label so that the resulting labelled window
    fraction is approximately `frac`.

    Runs are drawn uniformly at random -- deliberately not stratified -- so the
    labelled set inherits (and typically amplifies) the natural class imbalance.
    A minimum of `min_per_class` labelled runs per stage is then enforced,
    because a stage with zero labels makes the K-way problem ill-posed and any
    method would score zero on it for trivial reasons.

    Returns boolean mask over windows.
    """
    rng = np.random.default_rng(seed)
    runs = np.unique(run_ids)
    run_stage = np.array([stages[run_ids == r][0] for r in runs])

    n_target = max(1, int(round(frac * len(runs))))
    chosen = set(rng.choice(runs, size=min(n_target, len(runs)), replace=False).tolist())

    for k in np.unique(run_stage):
        have = sum(1 for r in chosen if run_stage[runs == r][0] == k)
        if have < min_per_class:
            pool = [r for r in runs[run_stage == k] if r not in chosen]
            rng.shuffle(pool)
            for r in pool[: min_per_class - have]:
                chosen.add(int(r))

    # Enforce the budget in *windows*, not only in passes.  Cutting passes are
    # not all the same length, so a draw can occasionally land on an unusually
    # long pass and blow past the stated budget.  We therefore drop the largest
    # chosen passes until the labelled window fraction is back inside
    # `budget_tol` x frac, never removing the last pass of a stage.
    def _mask(ch):
        return np.isin(run_ids, np.array(sorted(ch)))

    limit = budget_tol * frac
    while len(chosen) > 1 and _mask(chosen).mean() > limit:
        sizes = {r: int((run_ids == r).sum()) for r in chosen}
        per_class = {}
        for r in chosen:
            k = int(run_stage[runs == r][0])
            per_class[k] = per_class.get(k, 0) + 1
        droppable = [r for r in chosen
                     if per_class[int(run_stage[runs == r][0])] > min_per_class]
        if not droppable:
            break
        chosen.discard(max(droppable, key=lambda r: sizes[r]))

    return _mask(chosen)


def describe(counts: np.ndarray) -> str:
    counts = np.asarray(counts)
    ir = counts.max() / max(1, counts[counts > 0].min())
    return f"{counts.tolist()} (IR={ir:.2f})"
