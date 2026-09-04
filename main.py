"""Batch entry point: read config.yaml, match, write the output file.

Everything it does is available as a module too -- see `catmatch.Matcher`.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

from catmatch import Config, Matcher, read_records, write_records  # noqa: E402


def main() -> None:
    config = Config.from_yaml("config.yaml")
    for name in ("input_path", "taxonomy_path", "output_path"):
        if not getattr(config, name):
            raise ValueError(f"config.yaml is missing {name.split('_')[0]}.path")

    print("reading the input files...")
    records = read_records(config.input_path)
    taxonomy = read_records(config.taxonomy_path)
    print(f"{len(records)} record(s), {len(taxonomy)} taxonomy row(s)\n")

    results = Matcher(config).match(records, taxonomy)

    write_records(results, config.output_path)
    print(f"\n*** done -- result written to {config.output_path} ***\n")


if __name__ == "__main__":
    main()
