from pathlib import Path
import re

import pandas as pd

from dataloader import build_records


CAI_DIR = Path(
    "dataset/"
    "Datasets on CFRP specimens subjected to compression after impact tests/"
    "5_Compression after impact strength"
)


def normalize_specimen_name(name):
    """
    Examples:
        C8-1.jpg  -> c8-1
        Q24_12    -> q24-12
        C24-10t   -> c24-10t
    """
    name = str(name).strip().lower()

    # 去扩展名
    name = Path(name).stem

    # 统一各种分隔符
    name = name.replace("_", "-")
    name = name.replace(" ", "")

    return name


# 匹配：
# c8-1
# c16-31
# c24-10t
# q8-25
# q16-3
# q24-50
SPECIMEN_PATTERN = re.compile(
    r"^(?:c|q)(?:8|16|24)-\d+t?$",
    re.IGNORECASE,
)


# ============================================================
# 1. 读取当前 C-scan dataset
# ============================================================

records, _ = build_records()

cscan_records = {
    normalize_specimen_name(Path(r.image_path).name): r
    for r in records
    if "astm" not in Path(r.image_path).stem.lower()
}

print()
print("=" * 60)
print("C-SCAN DATASET")
print("=" * 60)
print("Non-ASTM C-scans:", len(cscan_records))


# ============================================================
# 2. 扫描 CAI strength 文件
# ============================================================

files = []

for pattern in ("*.xlsx", "*.xls", "*.csv"):
    files.extend(CAI_DIR.rglob(pattern))

print()
print("=" * 60)
print("CAI FILES")
print("=" * 60)

for file in files:
    print(file)

print("Number of files:", len(files))


# ============================================================
# 3. 从所有表格单元格中寻找 specimen name
# ============================================================

cai_specimens = set()

for file in files:

    print()
    print(f"[reading] {file}")

    try:
        if file.suffix.lower() == ".csv":
            sheets = {
                "csv": pd.read_csv(
                    file,
                    header=None,
                )
            }

        else:
            sheets = pd.read_excel(
                file,
                sheet_name=None,
                header=None,
            )

    except Exception as e:
        print(f"[skip] {file}: {e}")
        continue

    for sheet_name, df in sheets.items():

        sheet_found = set()

        for value in df.astype(str).to_numpy().ravel():

            value = normalize_specimen_name(value)

            if SPECIMEN_PATTERN.match(value):
                sheet_found.add(value)
                cai_specimens.add(value)

        if sheet_found:
            print(
                f"  sheet={sheet_name}: "
                f"{len(sheet_found)} specimen names"
            )


# ============================================================
# 4. 求交集
# ============================================================

cscan_names = set(cscan_records.keys())

intersection = cscan_names & cai_specimens

missing_cai = cscan_names - cai_specimens

cai_without_cscan = cai_specimens - cscan_names


print()
print("=" * 60)
print("INTERSECTION RESULT")
print("=" * 60)

print("Non-ASTM C-scan :", len(cscan_names))
print("CAI specimens   :", len(cai_specimens))
print("Intersection    :", len(intersection))
print("Missing CAI     :", len(missing_cai))


print()
print("=" * 60)
print("C-SCANS WITHOUT CAI LABEL")
print("=" * 60)

for name in sorted(missing_cai):

    r = cscan_records[name]

    print(
        f"{Path(r.image_path).name:25s}",
        f"E={r.impact_energy_j:7.3f} J",
        f"family={r.family}",
        f"condition={r.condition_id}",
    )


print()
print("=" * 60)
print("CAI SPECIMENS WITHOUT C-SCAN")
print("=" * 60)

for name in sorted(cai_without_cscan):
    print(name)