"""Point-in-time guarantees: nothing known after the application may leak into features or tokens."""
import numpy as np
import pandas as pd

from scoring.config import OOT_START
from scoring.features import build_features, pit_recompute
from scoring.sequences import build_tokens, make_store


def test_history_features_equal_recompute_on_truncated_history(small_data):
    d = small_data
    feats = build_features(d.applications, d.loans, d.events, d.external).set_index("application_id")
    ids = d.applications.sample(40, random_state=0)["application_id"].tolist()
    re = pit_recompute(ids, d.applications, d.loans, d.events).set_index("application_id")
    batch = feats.loc[ids, re.columns]
    mismatch = ((re - batch).abs() > 1e-9) & ~(re.isna() & batch.isna())
    assert int(mismatch.values.sum()) == 0, mismatch.sum()[mismatch.sum() > 0]


def test_at_least_300_features(small_data):
    d = small_data
    feats = build_features(d.applications, d.loans, d.events, d.external)
    assert feats.shape[1] - 1 >= 300


def test_tokens_strictly_before_application(small_data):
    d = small_data
    meta, tok = build_tokens(d.events, d.loans)
    store = make_store(meta, tok, d.applications, OOT_START)
    has = store.lengths > 0
    assert has.mean() > 0.99                       # repeat borrowers always have a history
    assert (store.t[store.seq_end[has] - 1] < store.app_t[has]).all()
    # the next token of the same client (if any) is not before the application
    nxt = store.seq_end[has]
    same_client = nxt < len(meta)
    same_client[same_client] &= (meta["client_id"].to_numpy()[nxt[same_client]] ==
                                 d.applications["client_id"].to_numpy()[has][same_client])
    assert (store.t[nxt[same_client]] >= store.app_t[has][same_client]).all()


def test_events_of_scored_loan_are_after_application(small_data):
    d = small_data
    ev = d.events.merge(d.applications[["application_id", "application_date"]], left_on="loan_id", right_on="application_id")
    assert (ev["event_time"] >= ev["application_date"]).all()
    assert pd.api.types.is_datetime64_any_dtype(ev["event_time"])
    assert np.isfinite(d.applications["npl30d90"]).all()
