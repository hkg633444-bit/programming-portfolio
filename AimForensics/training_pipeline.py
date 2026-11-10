"""Versioned temporal datasets and source-disjoint evaluation."""
import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path

import numpy as np

SCHEMA = 1
WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 1.0
SAMPLE_HZ = 30
FEATURE_NAMES = ["relative_x", "relative_y", "vx", "vy", "speed", "acceleration"]
DATA_PATH = Path("saved_training_data/temporal_v1.npz")
_SAVE_LOCK = threading.RLock()


def video_id(data):
    return hashlib.sha256(data).hexdigest()


def temporal_windows(positions, timestamps):
    """Resample to 30 Hz; no padding or interpolation across missing intervals."""
    p, t = np.asarray(positions, float), np.asarray(timestamps, float)
    if len(t) < 2 or p.shape != (len(t), 2) or not np.isfinite(p).all():
        raise ValueError("軌跡資料不足或無效")
    gaps = np.diff(t)
    if not np.isfinite(t).all() or np.any(gaps <= 0):
        raise ValueError("時間戳必須有限且嚴格遞增")
    if np.max(gaps) > 1 / SAMPLE_HZ + 1e-6:
        raise ValueError("訓練需至少 30 Hz 的有效取樣率；請使用逐幀提取及 ≥30 FPS 影片")
    dt = 1 / SAMPLE_HZ
    grid = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * SAMPLE_HZ + 1e-6)) + 1) * dt
    points = np.column_stack([np.interp(grid, t, p[:, axis]) for axis in (0, 1)])
    length, stride = int(WINDOW_SECONDS * SAMPLE_HZ), int(STRIDE_SECONDS * SAMPLE_HZ)
    sequences, starts = [], []
    for start in range(0, len(points) - length + 1, stride):
        q = points[start:start + length]
        velocity = np.gradient(q, dt, axis=0)
        acceleration = np.linalg.norm(np.gradient(velocity, dt, axis=0), axis=1)
        features = np.column_stack([
            q - q[0], velocity, np.linalg.norm(velocity, axis=1), acceleration
        ])
        sequences.append(features)
        starts.append(float(grid[start]))
    if not sequences:
        raise ValueError("影片不足一個完整的 2 秒視窗（60 個 30 Hz 取樣點）")
    return np.asarray(sequences, np.float32), starts


def grouped_split(labels, groups, three_way=True, test_ratio=0.2):
    """Stratify source IDs, never individual windows. Fail on insufficient sources."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    if not np.isfinite(test_ratio) or not 0 < test_ratio < 1:
        raise ValueError("測試比例必須介於 0 與 1 之間")
    if labels.ndim != 1 or groups.ndim != 1:
        raise ValueError("標籤與來源必須是一維陣列")
    if len(labels) != len(groups) or len(labels) == 0:
        raise ValueError("缺少逐樣本來源分組")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("需要 Human 與 Aimbot 兩個類別")
    source_labels = {}
    for group in np.unique(groups):
        if not str(group):
            raise ValueError("來源分組不可空白")
        values = np.unique(labels[groups == group])
        if len(values) != 1:
            raise ValueError("同一來源分組不能包含相衝突的標籤")
        source_labels[group] = values[0]
    rng = np.random.default_rng(42)
    assigned = {"train": [], "validation": [], "test": []}
    for label in (0, 1):
        ids = sorted(g for g, value in source_labels.items() if value == label)
        minimum = 3 if three_way else 2
        if len(ids) < minimum:
            raise ValueError(f"每類至少需要 {minimum} 個獨立來源；類別 {label} 只有 {len(ids)} 個")
        ids = np.asarray(ids)
        rng.shuffle(ids)
        n_test = min(len(ids) - (2 if three_way else 1),
                     max(1, int(np.ceil(len(ids) * test_ratio))))
        n_val = min(len(ids) - n_test - 1, max(1, int(np.ceil(len(ids) * .2)))) if three_way else 0
        assigned["test"].extend(ids[:n_test])
        assigned["validation"].extend(ids[n_test:n_test + n_val])
        assigned["train"].extend(ids[n_test + n_val:])
    parts = {key: np.flatnonzero(np.isin(groups, ids)) for key, ids in assigned.items()}
    assert not (set(assigned["train"]) & set(assigned["test"]))
    assert not (set(assigned["train"]) & set(assigned["validation"]))
    assert not (set(assigned["validation"]) & set(assigned["test"]))
    return parts


def empty_dataset():
    return {"x": np.empty((0, 60, 6), np.float32), "y": np.empty(0, np.int64),
            "groups": np.empty(0, dtype=str), "videos": []}


def load_dataset(path=DATA_PATH):
    if not Path(path).exists():
        return empty_dataset()
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata["schema"] != SCHEMA:
            raise ValueError("訓練資料版本不相容，請勿混用舊影像或摘要特徵資料")
        result = {key: data[key].copy() for key in ("x", "y", "groups")}
        result["videos"] = metadata["videos"]
    validate_dataset(result)
    return result


def validate_dataset(result):
    n = len(result["y"])
    if result["x"].shape != (n, 60, 6) or result["y"].shape != (n,) or result["groups"].shape != (n,):
        raise ValueError("儲存的訓練資料形狀不符")
    if not np.isfinite(result["x"]).all() or not set(result["y"].tolist()).issubset({0, 1}):
        raise ValueError("訓練資料含非有限數值或未知標籤")
    offset, ids = 0, set()
    for video in result["videos"]:
        size = video["windows"]
        if not isinstance(size, int) or size <= 0 or not video["group"] or video["id"] in ids:
            raise ValueError("影片來源中繼資料無效或重複")
        ids.add(video["id"])
        if (len(video["starts_seconds"]) != size or offset + size > n
                or not np.all(result["groups"][offset:offset + size] == video["group"])
                or not np.all(result["y"][offset:offset + size] == video["label"])):
            raise ValueError("影片來源與序列對應不一致")
        offset += size
    if offset != n:
        raise ValueError("缺少序列對應的影片來源")


def save_dataset(dataset, path=DATA_PATH, expected_ids=None):
    validate_dataset(dataset)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _SAVE_LOCK:
        if expected_ids is not None:
            current = {v["id"] for v in load_dataset(path)["videos"]}
            if current != set(expected_ids):
                raise ValueError("資料集已被另一個操作更新，請重新整理後再加入，避免覆蓋資料")
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
                tmp = f.name
                np.savez_compressed(f, x=dataset["x"], y=dataset["y"], groups=dataset["groups"],
                                    metadata=json.dumps({"schema": SCHEMA, "videos": dataset["videos"]}))
            os.replace(tmp, path)
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)


def append_video(dataset, content_id, group, label, name, sequences, starts):
    if label not in (0, 1) or not content_id or not group:
        raise ValueError("缺少影片識別、來源分組或有效標籤")
    if (sequences.ndim != 3 or sequences.shape[1:] != (60, 6) or len(sequences) == 0
            or not np.isfinite(sequences).all() or len(starts) != len(sequences)
            or not np.isfinite(starts).all()):
        raise ValueError("影片序列或時間戳無效")
    if any(v["id"] == content_id for v in dataset["videos"]):
        raise ValueError("相同影片內容已加入；重新命名也不能重複加入")
    if any(v["group"] == group and v["label"] != label for v in dataset["videos"]):
        raise ValueError("同一錄影來源的標籤不一致")
    n = len(sequences)
    return {
        "x": np.concatenate([dataset["x"], sequences]),
        "y": np.concatenate([dataset["y"], np.full(n, label, np.int64)]),
        "groups": np.concatenate([dataset["groups"], np.full(n, group)]),
        "videos": dataset["videos"] + [{"id": content_id, "group": group, "label": label,
                                       "filename": name, "windows": n, "starts_seconds": starts}],
    }


def split_manifest(dataset, parts):
    report = {"schema": SCHEMA, "window_seconds": WINDOW_SECONDS,
              "stride_seconds": STRIDE_SECONDS, "sample_hz": SAMPLE_HZ,
              "sequence_shape": [60, 6], "feature_names": FEATURE_NAMES,
              "seed": 42, "total_windows": len(dataset["y"]),
              "total_videos": len(dataset["videos"]), "splits": {}}
    sets = {}
    for name, indices in parts.items():
        groups = set(dataset["groups"][indices].tolist())
        sets[name] = groups
        videos = [v for v in dataset["videos"] if v["group"] in groups]
        report["splits"][name] = {
            "source_groups": sorted(groups), "videos": videos, "video_count": len(videos),
            "window_count": len(indices),
            "human_windows": int(np.sum(dataset["y"][indices] == 0)),
            "aimbot_windows": int(np.sum(dataset["y"][indices] == 1))}
    report["group_intersections"] = {
        f"{a}/{b}": sorted(sets[a] & sets[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    return report


def standardize(x, train_indices):
    mean = x[train_indices].mean(axis=(0, 1), keepdims=True)
    scale = x[train_indices].std(axis=(0, 1), keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return ((x - mean) / scale).astype(np.float32), mean, scale
