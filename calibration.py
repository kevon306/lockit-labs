"""
calibration.py  -  the real "learning loop": nudges model probabilities toward what
the locked prediction ledger has actually shown, once there's enough of it to trust.

This never touches the walk-forward projection models themselves (model.py,
team_model.py stay exactly as they are) and never touches the locked predictions
table - it's a small, transparent correction layer applied on top of a freshly
computed probability, using Platt scaling (fit a 2-parameter logistic curve mapping
raw probability -> empirically observed win rate): the standard, well-understood way
to recalibrate a model's stated probabilities against its own real track record,
not a from-scratch retrain.

Gated hard behind MIN_DECIDED_FOR_CALIBRATION: below that many real decided (won/lost)
predictions, fit() returns None and apply() is a strict no-op, passing probabilities
through completely unchanged. A 2-parameter fit "trained" on a handful of outcomes
would just be fitting noise and dressing it up as learning - exactly what this project
has refused to do everywhere else (see model.py's MIN_HISTORY, team_model.py's
MIN_HISTORY). The threshold is deliberately a round, conservative number, not tuned to
make anything "turn on" sooner.

One global curve, not split by market or confidence tier - with how little graded
history exists today, splitting further would guarantee "not enough data" for a long
time. Model Performance already shows per-market win rates for a human to eyeball
separately; this file can grow a per-market version once there's genuinely enough
volume to support it.

Recomputed fresh every call (no model version, no table of its own) - cheap at this
scale, and more honest than caching a stale fit: every newly graded game should be
able to move it immediately, not just whenever someone remembers to retrain a
versioned model. lockit_app.py wraps the entry point in its own short-TTL cache for
performance, the same way it caches everything else derived from the live database.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import db

MIN_DECIDED_FOR_CALIBRATION = 100  # see module docstring for why this isn't lower


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit(p_over_pct, outcomes):
    """Fits (a, b) so sigmoid(a * logit(p/100) + b) best predicts `outcomes` (1.0 for
    WON, 0.0 for LOST) by maximum likelihood, given the raw model probabilities
    (0-100 scale) that were actually logged for those same predictions.

    Returns None - not a fit - if there are fewer than MIN_DECIDED_FOR_CALIBRATION
    real decided predictions, or if the optimizer doesn't converge. A caller must
    treat None as "no correction available," never fall back to guessing a=1, b=0
    silently - apply() already does this for you."""
    p_over_pct = np.asarray(p_over_pct, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    n = len(p_over_pct)
    if n < MIN_DECIDED_FOR_CALIBRATION:
        return None
    z = _logit(p_over_pct / 100)

    def neg_log_lik(params):
        a, b = params
        p = np.clip(_sigmoid(a * z + b), 1e-6, 1 - 1e-6)
        return -(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)).sum()

    result = minimize(neg_log_lik, x0=[1.0, 0.0], method="Nelder-Mead")
    if not result.success:
        return None
    return float(result.x[0]), float(result.x[1])


def apply(p_over_pct, params):
    """Applies a fitted (a, b) correction to a raw model probability (0-100 scale,
    scalar or array/Series alike). params=None (not enough data yet, see fit()) means
    this is a pure no-op - the input comes back completely unchanged, which is the
    honest default this app shipped with for its entire history before this file
    existed."""
    if params is None:
        return p_over_pct
    a, b = params
    return 100 * _sigmoid(a * _logit(p_over_pct / 100) + b)


def current_calibration():
    """Loads every decided (WON/LOST) prediction from the locked ledger right now and
    fits a fresh curve from scratch. Returns a dict:
      params            (a, b) tuple, or None if there isn't enough data yet
      n                 how many decided predictions this was computed from
      raw_brier         Brier score of the probabilities as originally logged
      calibrated_brier  Brier score after applying this same fit - None if params is
                        None

    raw_brier and calibrated_brier are both measured on the SAME sample the curve was
    fit on, not a held-out set - with this little data, a real train/test split would
    leave too little of either to mean anything. That makes calibrated_brier an
    optimistic, in-sample number, not proof the correction generalizes - shown for
    transparency (so a human can see the fit isn't doing anything bizarre), not as
    evidence it's actually improving future predictions.

    Fits against raw_model_p, not p_over - p_over can now be blended with the market
    for a market that hasn't beaten its own closing-line backtest yet (see
    lockit_app.py's shown_probability()), and fitting a correction against a
    market-blended number would be learning to correct the market, not the model.
    COALESCE falls back to p_over for rows logged before raw_model_p existed - back
    then p_over always was the raw model value, so the fallback is exact, not a guess."""
    with db.connect() as conn:
        preds = pd.read_sql("SELECT id, COALESCE(raw_model_p, p_over) AS raw_model_p FROM predictions", conn)
        results = pd.read_sql("SELECT prediction_id, result FROM prediction_results", conn)
    merged = preds.merge(results, left_on="id", right_on="prediction_id")
    decided = merged[merged["result"].isin(["WON", "LOST"])]
    n = len(decided)
    if n == 0:
        return {"params": None, "n": 0, "raw_brier": None, "calibrated_brier": None}

    outcomes = (decided["result"] == "WON").to_numpy(dtype=float)
    p_pct = (decided["raw_model_p"] * 100).to_numpy()
    params = fit(p_pct, outcomes)
    raw_brier = float(np.mean((p_pct / 100 - outcomes) ** 2))
    calibrated_brier = None
    if params is not None:
        calibrated_brier = float(np.mean((apply(p_pct, params) / 100 - outcomes) ** 2))
    return {"params": params, "n": n, "raw_brier": raw_brier, "calibrated_brier": calibrated_brier}
