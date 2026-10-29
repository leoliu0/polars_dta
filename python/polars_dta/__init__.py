"""Write Polars DataFrames to Stata .dta files."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping

import polars as pl

from polars_dta._polars_dta import write_dta as _write_dta

__all__ = ["write_dta"]

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def write_dta(
    df: pl.DataFrame | pl.LazyFrame,
    file: str | os.PathLike[str],
    *,
    variable_labels: Mapping[str, str] | None = None,
    value_labels: Mapping[str, Mapping[int, str]] | None = None,
    data_label: str | None = None,
) -> None:
    """
    Write a DataFrame to a Stata .dta file (format 118, or 119 above 32,767 columns).

    The file is written next to ``file`` under a temporary name and renamed into place, so a
    failed write leaves any existing file untouched.

    Parameters
    ----------
    df
        Data to write. A LazyFrame is collected first.
    file
        Path of the .dta file.
    variable_labels
        Column name to variable label (at most 80 characters).
    value_labels
        Integer or Boolean column name to a ``{value: label}`` mapping.
    data_label
        Dataset label (at most 80 characters).

    Notes
    -----
    Column types map to Stata types as follows; nulls become missing (``.``), or ``""`` in
    string columns, which have no missing value in Stata.

    - Integers: the smallest of ``byte``, ``int``, ``long`` holding every value, starting from
      the type matching the column's width (Int64 starts at ``long``); ``double`` when no integer
      type fits. Values beyond ±2^53 raise an error, as a double would round them.
    - Boolean: ``byte`` 0/1.
    - Float32: ``float`` (``double`` when a value exceeds Stata's float range); Float64:
      ``double``. NaN becomes missing; infinity raises an error.
    - String: ``str#`` up to 2045 bytes, ``strL`` beyond (or when a value contains NUL).
    - Enum and Categorical: integer codes 1, 2, ... in category order, with a value label of the
      same name holding the categories.
    - Date: ``long`` with format ``%td``. Datetime: ``double`` with format ``%tc``, in
      milliseconds; time-zone-aware values are written as local time in their zone. Time:
      ``double`` with format ``%tcHH:MM:SS``.
    - Null: ``byte``, all missing.

    Other types (Duration, Decimal, Binary, nested types) raise an error; cast them first.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_dta import write_dta
    >>> df = pl.DataFrame({"id": [1, 2], "sex": pl.Series(["m", "f"], dtype=pl.Enum(["m", "f"]))})
    >>> write_dta(df, "out.dta", variable_labels={"sex": "Sex of respondent"})  # doctest: +SKIP
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    casts = []
    for name, dtype in df.schema.items():
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
            casts.append(pl.col(name).dt.replace_time_zone(None))
        elif dtype == pl.Null:
            # arrow-rs rejects the buffer Polars exports for Null arrays.
            casts.append(pl.col(name).cast(pl.Int8))
    if casts:
        df = df.with_columns(casts)
    now = time.localtime()
    _write_dta(
        df,
        os.fspath(file),
        variable_labels=dict(variable_labels) if variable_labels else None,
        value_labels=(
            {name: dict(labels) for name, labels in value_labels.items()} if value_labels else None
        ),
        data_label=data_label,
        timestamp=(
            f"{now.tm_mday:02d} {_MONTHS[now.tm_mon - 1]} {now.tm_year} "
            f"{now.tm_hour:02d}:{now.tm_min:02d}"
        ),
    )


@pl.api.register_dataframe_namespace("dta")
class DtaNameSpace:
    """Stata .dta output, as ``df.dta.write(...)``."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def write(
        self,
        file: str | os.PathLike[str],
        *,
        variable_labels: Mapping[str, str] | None = None,
        value_labels: Mapping[str, Mapping[int, str]] | None = None,
        data_label: str | None = None,
    ) -> None:
        """Write the DataFrame to a Stata .dta file; see :func:`polars_dta.write_dta`."""
        write_dta(
            self._df,
            file,
            variable_labels=variable_labels,
            value_labels=value_labels,
            data_label=data_label,
        )
