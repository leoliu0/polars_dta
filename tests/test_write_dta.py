from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from polars_dta import write_dta


def roundtrip(df: pl.DataFrame, path: Path, **kwargs) -> pd.DataFrame:
    write_dta(df, path, **kwargs)
    return pd.read_stata(path, convert_categoricals=False)


def variable_types(path: Path) -> list[int]:
    """Stata type codes of a release 118 file: 1-2045 str#, 32768 strL, 65526-65530 numeric."""
    data = path.read_bytes()
    start = data.index(b"<variable_types>") + len(b"<variable_types>")
    end = data.index(b"</variable_types>")
    return np.frombuffer(data[start:end], dtype="<u2").tolist()


@pytest.mark.parametrize(
    ("dtype", "values", "stata_dtype"),
    [
        (pl.Int8, [-127, 100], "int8"),
        (pl.Int8, [101], "int16"),
        (pl.Int8, [-128], "int16"),
        (pl.Int16, [-32_767, 32_740], "int16"),
        (pl.Int16, [32_741], "int32"),
        (pl.UInt8, [255], "int16"),
        (pl.Int32, [-2_147_483_647, 2_147_483_620], "int32"),
        (pl.Int32, [2_147_483_621], "float64"),
        (pl.Int64, [1, 2], "int32"),
        (pl.Int64, [-(2**53), 2**53], "float64"),
        (pl.UInt64, [2**53], "float64"),
    ],
)
def test_integer_storage(
    tmp_path: Path, dtype: pl.DataType, values: list[int], stata_dtype: str
) -> None:
    out = roundtrip(pl.DataFrame({"v": pl.Series(values, dtype=dtype)}), tmp_path / "a.dta")
    assert str(out["v"].dtype) == stata_dtype
    assert out["v"].tolist() == values


@pytest.mark.parametrize(("dtype", "value"), [(pl.Int64, 2**53 + 1), (pl.UInt64, 2**64 - 1)])
def test_integer_beyond_double_precision_raises(
    tmp_path: Path, dtype: pl.DataType, value: int
) -> None:
    with pytest.raises(ValueError, match=r"±2\^53"):
        write_dta(pl.DataFrame({"v": pl.Series([value], dtype=dtype)}), tmp_path / "a.dta")


def test_nulls_and_nan_become_missing(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "i": pl.Series([1, None], dtype=pl.Int16),
            "b": [None, True],
            "x": [float("nan"), None],
            "d": [None, dt.date(2000, 1, 1)],
            "s": [None, "a"],
            "n": [None, None],
        }
    )
    out = roundtrip(df, tmp_path / "a.dta")
    assert out["i"].isna().tolist() == [False, True]
    assert out["b"].isna().tolist() == [True, False]
    assert out["x"].isna().tolist() == [True, True]
    assert out["d"].isna().tolist() == [True, False]
    assert out["s"].tolist() == ["", "a"]
    assert out["n"].isna().all()


def test_floats(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "f32": pl.Series([1.5, -2.25], dtype=pl.Float32),
            "f32_wide": pl.Series([3e38, 1.0], dtype=pl.Float32),
            "f64": [1e300, -1e-300],
        }
    )
    out = roundtrip(df, tmp_path / "a.dta")
    assert out.dtypes.astype(str).tolist() == ["float32", "float64", "float64"]
    assert out["f32"].tolist() == [1.5, -2.25]
    assert out["f32_wide"].tolist() == [np.float32(3e38), 1.0]
    assert out["f64"].tolist() == [1e300, -1e-300]


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
def test_infinity_raises(tmp_path: Path, dtype: pl.DataType) -> None:
    with pytest.raises(ValueError, match="infinity"):
        write_dta(
            pl.DataFrame({"x": pl.Series([1.0, float("-inf")], dtype=dtype)}), tmp_path / "a.dta"
        )


def test_strings(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "fixed": ["é" * 1022 + "x", "日本語"],  # 2045 bytes: the widest str#
            "strl": ["é" * 1023, ""],  # 2046 bytes
            "nul": ["a\x00b", "c"],
        }
    )
    path = tmp_path / "a.dta"
    out = roundtrip(df, path)
    assert out["fixed"].tolist() == df["fixed"].to_list()
    assert out["strl"].tolist() == df["strl"].to_list()
    assert variable_types(path) == [2045, 32768, 32768]
    # A value holding NUL is a binary GSO (type 129): length 3, then the bytes as they are.
    assert b"\x81\x03\x00\x00\x00a\x00b" in path.read_bytes()


def test_enum_and_categorical_get_value_labels(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "grade": pl.Series(["hi", None, "lo"], dtype=pl.Enum(["lo", "mid", "hi"])),
            "region": pl.Series(["west", "east", "west"], dtype=pl.Categorical),
        }
    )
    path = tmp_path / "a.dta"
    out = roundtrip(df, path)
    assert out["grade"].tolist()[0::2] == [3, 1]
    assert np.isnan(out["grade"][1])
    with pd.io.stata.StataReader(path) as reader:
        assert reader.value_labels() == {
            "grade": {1: "lo", 2: "mid", 3: "hi"},
            "region": {1: "west", 2: "east"},
        }
    labelled = pd.read_stata(path)
    assert labelled["grade"].tolist()[0::2] == ["hi", "lo"]
    assert labelled["region"].tolist() == ["west", "east", "west"]


def test_multiple_chunks(tmp_path: Path) -> None:
    first = pl.DataFrame(
        {"c": pl.Series(["a", "b"], dtype=pl.Categorical), "s": ["x" * 3000, ""], "i": [1, 2]}
    )
    second = pl.DataFrame(
        {"c": pl.Series(["z", "a"], dtype=pl.Categorical), "s": ["", "y" * 2500], "i": [3, 4]}
    )
    df = pl.concat([first, second], rechunk=False)
    assert df.n_chunks() == 2
    path = tmp_path / "a.dta"
    write_dta(df, path)
    out = pd.read_stata(path)
    assert out["c"].tolist() == ["a", "b", "z", "a"]
    assert out["s"].tolist() == df["s"].to_list()
    assert out["i"].tolist() == [1, 2, 3, 4]


def test_temporal(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "d": [dt.date(1900, 1, 1), dt.date(2024, 2, 29)],
            "ts_us": [
                dt.datetime(1959, 12, 31, 23, 59, 59, 999_999),
                dt.datetime(2024, 1, 1, 12, 0, 0, 123_456),
            ],
            "ts_ns": pl.Series([dt.datetime(1970, 1, 1), dt.datetime(2000, 6, 30, 1, 2, 3)]).cast(
                pl.Datetime("ns")
            ),
            "ts_tz": pl.Series([dt.datetime(2024, 7, 1, 12), dt.datetime(2024, 1, 1, 12)])
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("America/New_York"),
            "t": [dt.time(0), dt.time(23, 59, 59, 999_999)],
        }
    )
    out = roundtrip(df, tmp_path / "a.dta")
    assert out["d"].tolist() == [pd.Timestamp("1900-01-01"), pd.Timestamp("2024-02-29")]
    assert out["ts_us"].tolist() == [
        pd.Timestamp("1959-12-31 23:59:59.999"),
        pd.Timestamp("2024-01-01 12:00:00.123"),
    ]
    assert out["ts_ns"].tolist() == [
        pd.Timestamp("1970-01-01"),
        pd.Timestamp("2000-06-30 01:02:03"),
    ]
    # Wall-clock time in the column's zone.
    assert out["ts_tz"].tolist() == [
        pd.Timestamp("2024-07-01 08:00"),
        pd.Timestamp("2024-01-01 07:00"),
    ]
    assert out["t"].tolist() == [
        pd.Timestamp("1960-01-01"),
        pd.Timestamp("1960-01-01 23:59:59.999"),
    ]


def test_labels(tmp_path: Path) -> None:
    df = pl.DataFrame({"sex": [1, 2, 1], "ok": [True, False, True]})
    path = tmp_path / "a.dta"
    write_dta(
        df,
        path,
        variable_labels={"sex": "Sex of respondent", "ok": "Ünïcode ✓"},
        value_labels={"sex": {2: "Female", 1: "Male"}, "ok": {0: "no", 1: "yes"}},
        data_label="Survey wave 1",
    )
    with pd.io.stata.StataReader(path) as reader:
        assert reader.data_label == "Survey wave 1"
        assert reader.variable_labels() == {"sex": "Sex of respondent", "ok": "Ünïcode ✓"}
        assert reader.value_labels() == {"sex": {1: "Male", 2: "Female"}, "ok": {0: "no", 1: "yes"}}
    assert pd.read_stata(path)["sex"].tolist() == ["Male", "Female", "Male"]


def test_empty_frame(tmp_path: Path) -> None:
    df = pl.DataFrame(schema={"i": pl.Int64, "s": pl.String, "e": pl.Enum(["a", "b"])})
    out = roundtrip(df, tmp_path / "a.dta")
    assert out.columns.tolist() == ["i", "s", "e"]
    assert len(out) == 0


def test_more_than_32767_columns_uses_release_119(tmp_path: Path) -> None:
    df = pl.DataFrame({f"v{i}": [i % 100] for i in range(32_768)})
    path = tmp_path / "a.dta"
    write_dta(df, path)
    assert path.read_bytes().startswith(b"<stata_dta><header><release>119</release>")
    out = pd.read_stata(path)
    assert out.shape == (1, 32_768)
    assert out["v32767"].tolist() == [67]


@pytest.mark.parametrize("name", ["1st", "has space", "x" * 33, "_n", "str10", "byte", "a-b"])
def test_invalid_column_name_raises(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match=name if name.isidentifier() else None):
        write_dta(pl.DataFrame({name: [1]}), tmp_path / "a.dta")


def test_unicode_column_name(tmp_path: Path) -> None:
    out = roundtrip(pl.DataFrame({"größe": [1]}), tmp_path / "a.dta")
    assert out.columns.tolist() == ["größe"]


@pytest.mark.parametrize(
    "series",
    [
        pl.Series("v", [dt.timedelta(seconds=1)]),
        pl.Series("v", [Decimal("1.5")], dtype=pl.Decimal(10, 2)),
        pl.Series("v", [[1, 2]]),
        pl.Series("v", [b"\x00"]),
    ],
)
def test_unsupported_type_raises(tmp_path: Path, series: pl.Series) -> None:
    with pytest.raises(ValueError, match='column "v": Stata has no type'):
        write_dta(series.to_frame(), tmp_path / "a.dta")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"variable_labels": {"missing": "x"}}, 'no column "missing"'),
        ({"variable_labels": {"i": "x" * 81}}, "at most 80 characters"),
        ({"value_labels": {"s": {1: "x"}}}, "not an integer or boolean column"),
        ({"value_labels": {"i": {2_147_483_621: "x"}}}, "outside"),
        ({"data_label": "x" * 81}, "at most 80 characters"),
    ],
)
def test_invalid_options_raise(tmp_path: Path, kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        write_dta(pl.DataFrame({"i": [1], "s": ["a"]}), tmp_path / "a.dta", **kwargs)


def test_failed_write_leaves_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "a.dta"
    path.write_bytes(b"previous")
    with pytest.raises(ValueError, match="infinity"):
        write_dta(pl.DataFrame({"x": [1.0] * 100_000 + [float("inf")]}), path)
    assert path.read_bytes() == b"previous"
    assert [p.name for p in tmp_path.iterdir()] == ["a.dta"]
