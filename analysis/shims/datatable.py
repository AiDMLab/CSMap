"""Minimal datatable shim so original CSMap code can read CSV/TSV via pandas."""

from __future__ import annotations

import pandas as pd


class _Frame:
    def __init__(self, df: pd.DataFrame):
        self._df = df.reset_index(drop=True)

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            row_slice, col_sel = key
            sub = self._df.iloc[row_slice, col_sel]
            return _Column(sub)
        return _Column(self._df[key])

    def to_pandas(self) -> pd.DataFrame:
        return self._df.copy()


class _Column:
    def __init__(self, data):
        self._data = data

    def to_pandas(self):
        if isinstance(self._data, pd.DataFrame):
            return self._data.copy()
        return self._data.to_frame() if isinstance(self._data, pd.Series) else self._data


def fread(file_path: str, header: bool = True) -> _Frame:
    delim = "," if file_path.lower().endswith(".csv") else "\t"
    df = pd.read_csv(file_path, sep=delim)
    return _Frame(df)
