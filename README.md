# polars-dta

Write Polars DataFrames to Stata `.dta` files from Rust, without pandas or pyarrow.

```sh
pip install polars-dta
```

```python
import polars as pl
import polars_dta  # also registers df.dta

df = pl.DataFrame(
    {
        "id": [1, 2, 3],
        "grade": pl.Series(["lo", "hi", "lo"], dtype=pl.Enum(["lo", "mid", "hi"])),
        "income": [52_000.0, None, 61_500.5],
    }
)

df.dta.write("survey.dta", variable_labels={"income": "Annual income"}, data_label="Wave 1")
polars_dta.write_dta(df.lazy(), "survey.dta")  # same; a LazyFrame is collected first
```

Files are release 118, which Stata 14 and later read, or 119 (Stata 15+ MP) above 32,767
columns. The data reaches Rust through the Arrow C stream interface without copying. A failed
write leaves any existing file untouched.

## Types

| Polars | Stata |
| --- | --- |
| Int8 … Int64, UInt8 … UInt64 | smallest of `byte`/`int`/`long` holding the values, from the column's width up; `double` beyond `long`; values beyond ±2^53 raise an error |
| Boolean | `byte` 0/1 |
| Float32 / Float64 | `float` / `double`; NaN becomes `.`; infinity raises an error |
| String | `str1`–`str2045`, `strL` for longer values or values holding NUL |
| Enum, Categorical | integer codes 1, 2, … in category order plus a value label of the same name |
| Date | `long`, `%td` |
| Datetime | `double`, `%tc` (milliseconds); zoned values are written as local time in their zone |
| Time | `double`, `%tcHH:MM:SS` |
| Null | `byte`, all missing |

Nulls become `.`, except in strings, where Stata has no missing value and they become `""`.
Duration, Decimal, Binary and nested types raise an error; cast them first.

Column names must be valid Stata names (1–32 letters, digits or underscores, not starting with a
digit, not reserved such as `_n` or `byte`); invalid names raise an error rather than being
renamed.

Options: `variable_labels={col: label}`, `value_labels={int_or_bool_col: {value: label}}` and
`data_label`. Labels are at most 80 characters.

Requires Polars 1.3 or later.

## Development

```sh
uv venv && uv pip install polars numpy pandas pytest ruff maturin
maturin develop --release
pytest
cargo clippy --all-targets && cargo fmt --check && ruff check && ruff format --check
```

Releases: bump `version` in `Cargo.toml`, then push a tag `v<version>`; the CI workflow builds and
tests the wheels and publishes them to PyPI.
