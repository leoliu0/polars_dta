from typing import Any

def write_dta(
    data: Any,
    path: str,
    *,
    variable_labels: dict[str, str] | None = None,
    value_labels: dict[str, dict[int, str]] | None = None,
    data_label: str | None = None,
    timestamp: str | None = None,
) -> None: ...
