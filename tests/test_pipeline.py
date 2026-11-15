import numpy as np

from dgam.evaluate import evaluate
from dgam.method import fit_concept, get_balanced_accuracy
from dgam.synthetic import main as make_synthetic


def test_well_separated_concept_gets_a_good_threshold():
    rng = np.random.default_rng(1)
    absent = [(x, 0) for x in rng.normal(0.20, 0.012, 2000)]
    present = [(x, 1) for x in rng.normal(0.27, 0.012, 600)]
    data = sorted(absent + present)
    score, threshold = fit_concept(data)
    assert 0.22 < threshold < 0.25
    accuracy = get_balanced_accuracy({1: data}, {1: threshold}, {1: [1, score]}, 1)[0]
    assert accuracy > 0.95


def test_concept_selection_beats_using_all_concepts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["synthetic", "--out-dir", str(tmp_path), "--n-concepts", "30", "--n-images", "1500"],
    )
    make_synthetic()
    result = evaluate("SYNTHETIC", str(tmp_path), cutoff=10)
    assert result["n_concepts"] == 30
    assert result["balanced_accuracy_at_cutoff"] > result["balanced_accuracy_all_concepts"]
