"""Reading, writing and contract-checking the two data sides.

Any of xlsx / csv / json / jsonl goes in; everything downstream sees the same
JSON shape -- a list of dicts. The column contract is fixed by convention:

    input side     level_1, level_2, ... level_n   (ascending; the last one may
                                                    be free text such as an SKU name)
    taxonomy side  cat_1,   cat_2,   ... cat_n

English only -- there is no translation step.
"""

import json
import re
from pathlib import Path

import pandas as pd

Record = dict[str, object]


def read_records(path: str | Path) -> list[Record]:
    """Load xlsx / csv / json / jsonl into the common JSON shape."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path}: json input must be a list of objects")
        return data
    else:
        raise ValueError(
            f"unsupported input format: {path.suffix} (use xlsx, csv, json or jsonl)"
        )

    return df.to_dict(orient="records")


def write_records(records: list[Record], path: str | Path) -> None:
    """Write results out; format follows the extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    elif suffix == ".jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    elif suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2, default=str)
    else:
        raise ValueError(f"unsupported output format: {path.suffix}")


def level_columns(records: list[Record], prefix: str) -> list[str]:
    """Return `prefix_1 .. prefix_n` and check the contract.

    The numbers must start at 1 and be contiguous. Columns that don't match the
    pattern are left alone -- they ride through to the output untouched.
    """
    if not records:
        raise ValueError("no records found")

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    numbers = sorted(
        {
            int(m.group(1))
            for key in records[0]
            if (m := pattern.match(str(key)))
        }
    )

    if not numbers:
        found = ", ".join(str(k) for k in records[0]) or "(none)"
        raise ValueError(
            f"no {prefix}_1 ... {prefix}_n columns found. columns present: {found}"
        )
    expected = list(range(1, len(numbers) + 1))
    if numbers != expected:
        raise ValueError(
            f"{prefix}_* columns must be numbered 1..n with no gaps, got: "
            + ", ".join(f"{prefix}_{i}" for i in numbers)
        )

    return [f"{prefix}_{i}" for i in numbers]


def clean(value: object) -> str | None:
    """Strings get stripped; anything else (NaN, numbers, None) becomes None."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def load_input(path: str | Path) -> tuple[list[Record], list[str]]:
    records = read_records(path)
    return records, level_columns(records, "level")


def load_taxonomy(path: str | Path) -> tuple[list[Record], list[str]]:
    records = read_records(path)
    return records, level_columns(records, "cat")
