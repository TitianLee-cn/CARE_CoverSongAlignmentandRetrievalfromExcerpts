import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = Path(os.getenv("CARE_ROOT", PROJECT/"data_weights")).expanduser().resolve()
OUTPUT = Path(os.getenv("CARE_OUTPUT", PROJECT/"outputs/group_b")).expanduser().resolve()
MANIFESTS = Path(os.getenv("CARE_MANIFESTS", OUTPUT/"manifests")).expanduser().resolve()
NAMES = {"query":"covers80_queries", "reference":"covers80_references", "jamendo":"jamendo_inputs"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(exist_ok=True,parents=True)
    with NamedTemporaryFile("w", encoding="utf-8",dir=path.parent,delete=False) as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False)
    os.replace(f.name,path)


def cache_path(dataset, stage, row, output=OUTPUT):
    # The recording ID remains in the JSON. Hashing avoids platform-specific filename rules.
    key=hashlib.sha256(row["recording_id"].encode()).hexdigest()[:24]
    return output/dataset/stage/(key+".json")


def load_stage(dataset,stage,row,output=OUTPUT):
    data=read(cache_path(dataset,stage,row,output))
    if data["identity"] != identity(row):
        raise ValueError(f"Cache and active manifest disagree for {row['recording_id']}")
    return data


def identity(row):
    return dict(recording_id=row["recording_id"],sha256=row["sha256"],path=row["path"],
                source_start_sec=row.get("start_sec",0.),source_end_sec=row.get("end_sec",row["duration_sec"]))
