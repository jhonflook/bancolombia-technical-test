"""Escanea inf values en los parquets train/test/oot columna por columna (bajo RAM)."""
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
from pathlib import Path

ARTIFACTS = Path("data/artifacts")

for fname in ["train", "test", "oot"]:
    path = ARTIFACTS / f"{fname}.parquet"
    if not path.exists():
        print(f"\n{fname}: archivo no encontrado ({path})")
        continue

    pf = pq.ParquetFile(path)
    inf_report = {}

    for field in pf.schema_arrow:
        if pa.types.is_floating(field.type):
            col = pf.read(columns=[field.name]).column(field.name)
            n = pc.sum(pc.is_inf(col)).as_py() or 0
            if n > 0:
                inf_report[field.name] = n

    if inf_report:
        print(f"\n{fname}: {len(inf_report)} columna(s) con inf")
        for col, n in sorted(inf_report.items(), key=lambda x: -x[1]):
            print(f"  {col}: {n}")
    else:
        print(f"\n{fname}: sin inf values")
