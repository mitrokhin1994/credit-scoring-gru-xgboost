"""The numpy re-implementation used for deployment reproduces the torch model."""
import numpy as np
import torch

from scoring.gru import GRUConfig, GruNumpyScorer, SeqModel


def test_numpy_scorer_matches_torch():
    torch.manual_seed(0)
    cfg = GRUConfig(d=16, hidden=12)
    models = [SeqModel(n_feat=7, n_tasks=4, cfg=cfg).eval() for _ in range(3)]
    scorer = GruNumpyScorer(models, max_len=50)
    rng = np.random.default_rng(0)
    for T in [1, 5, 23]:
        x = rng.normal(size=(T, 8)).astype(np.float32)
        seg = int(rng.integers(0, 3))
        with torch.no_grad():
            ref = np.mean([m(torch.from_numpy(x[None]), torch.tensor([T]), torch.tensor([seg]))[0][0, 0].item() for m in models])
        assert abs(scorer.logit(x, seg) - ref) < 1e-5


def test_empty_history_handled():
    cfg = GRUConfig(d=8, hidden=6)
    m = SeqModel(n_feat=3, n_tasks=1, cfg=cfg).eval()
    scorer = GruNumpyScorer([m], max_len=10)
    with torch.no_grad():
        ref = m(torch.zeros(1, 1, 4), torch.tensor([0]), torch.tensor([1]))[0][0, 0].item()
    assert abs(scorer.logit(np.zeros((0, 4)), 1) - ref) < 1e-6
