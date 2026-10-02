from io import BytesIO
import numpy as np
import pandas as pd
import pytest

from financepaper.data.schema import FEATURE_NAMES, TARGET
from financepaper.data.split import split_indices
from financepaper.data.taiwan import WORKBOOK_NAME, download_taiwan, load_taiwan


def write_fixture(path, X, y):
    frame = X.copy()
    frame.insert(0, "ID", X.index)
    frame[TARGET] = y
    frame.to_csv(path, index=False)


def test_loader_schema_and_repeatability(tmp_path, credit_frame):
    X, y = credit_frame
    path = tmp_path / "fixture.csv"
    write_fixture(path, X, y)
    a, b = load_taiwan(path), load_taiwan(path)
    assert a.sha256 == b.sha256
    assert a.X.columns.tolist() == list(FEATURE_NAMES)
    pd.testing.assert_frame_equal(a.X, b.X)
    np.testing.assert_allclose(a.X, X)
    assert "ID" not in a.X and TARGET not in a.X
    assert a.X.index.equals(X.index)
    assert -2 in a.X.PAY_0.values


@pytest.mark.parametrize("fault", ["missing", "target", "duplicate_id", "extra", "fractional_code", "duplicate_column"])
def test_loader_rejects_invalid_source(tmp_path, credit_frame, fault):
    X, y = credit_frame
    path = tmp_path / "bad.csv"
    write_fixture(path, X, y)
    frame = pd.read_csv(path)
    if fault == "missing":
        frame.loc[0, "LIMIT_BAL"] = np.nan
    elif fault == "target":
        frame.loc[0, TARGET] = 2
    elif fault == "duplicate_id":
        frame.loc[0, "ID"] = frame.loc[1, "ID"]
    elif fault == "extra":
        frame["label_copy"] = y.to_numpy()
    elif fault == "fractional_code":
        frame.loc[0, "PAY_0"] = 0.5
    elif fault == "duplicate_column":
        columns = list(frame.columns)
        columns[1] = columns[2]
        frame.columns = columns
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_taiwan(path)


def test_split_disjoint_exhaustive_stratified():
    y = np.tile([0, 0, 0, 0, 1], 6000)
    split = split_indices(y, 42)
    assert [len(v) for v in split.values()] == [15000, 3000, 3000, 4500, 4500]
    all_indices = np.concatenate(list(split.values()))
    assert len(np.unique(all_indices)) == len(y)
    assert set(all_indices) == set(range(len(y)))
    for name, positions in split.items():
        np.testing.assert_array_equal(positions, split_indices(y, 42)[name])
        assert y[positions].mean() == pytest.approx(0.2)
    assert not np.array_equal(split["train"], split_indices(y, 43)["train"])


def test_download_rejects_bad_archive_before_writing(tmp_path, monkeypatch):
    import financepaper.data.taiwan as module
    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: BytesIO(b"not the official bytes"))
    with pytest.raises(ValueError, match="SHA256"):
        download_taiwan(tmp_path)
    assert not (tmp_path / WORKBOOK_NAME).exists()


def test_cached_download_validates_without_network(tmp_path, monkeypatch):
    import financepaper.data.taiwan as module
    content = b"synthetic download fixture"
    from hashlib import sha256
    monkeypatch.setattr(module, "WORKBOOK_SHA256", sha256(content).hexdigest())
    path = tmp_path / WORKBOOK_NAME
    path.write_bytes(content)
    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: pytest.fail("network call"))
    assert download_taiwan(tmp_path) == path
