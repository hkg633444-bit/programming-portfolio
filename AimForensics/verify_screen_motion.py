"""
驗證畫面位移軌跡。放在 app.py、screen_motion.py 同一個資料夾執行：

    python verify_screen_motion.py

1. 用固定亂數種子產生一張紋理圖，切出兩支答案已知的測試影片（320×240、30 fps、120 格、FFV1 無損）：
   static.avi：畫面完全不動；right5.avi：畫面內容每格往右移 5 像素。
2. 比較修正前（中心法）與修正後（畫面位移）的軌跡與可疑度（預設靈敏度 0.5、原始權重）。
3. 用 cProfile 量修正後流程（解碼＋位移估計＋特徵與評分）的時間。
結果寫到 verify_screen_motion.json。
"""
import cProfile
import json
import pstats

import cv2
import numpy as np

import app
from screen_motion import extract_screen_motion

W, H, FPS, N, STEP, SEED = 320, 240, 30, 120, 5, 20261018


def make_videos():
    rng = np.random.default_rng(SEED)
    tex = (rng.random((H + 40, W + STEP * N + 40, 3)) * 255).astype(np.uint8)
    tex = cv2.GaussianBlur(tex, (0, 0), 2)
    for name, step in [("static.avi", 0), ("right5.avi", STEP)]:
        vw = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"FFV1"), FPS, (W, H))
        x0 = STEP * N  # 視窗每格往左移 step 像素 → 畫面內容往右移 step 像素
        for t in range(N):
            x = x0 - step * t
            vw.write(tex[20:20 + H, x:x + W])
        vw.release()


def suspicion(positions, fps):
    ana = app.TrajectoryAnalyzer(fps=fps)
    ad = app.AnomalyDetector(sensitivity=0.5)
    ts = np.arange(len(positions)) / fps
    segs = ana.segment_trajectory(positions, ts, window_size=60, overlap=30)
    return float(np.mean([ad.analyze_segment(s).overall_suspicion for s in segs]))


def old_positions(path):
    traj = app.CrosshairDetector().extract_trajectory_from_video(path, method="center")
    return np.array([[p.x, p.y] for p in traj])


def new_pipeline(path):
    pos, ts, _, fps = extract_screen_motion(path)
    return pos, suspicion(pos, fps)


def main():
    make_videos()
    out = []
    for name, truth in [("static.avi", (0.0, 0.0)), ("right5.avi", (float(STEP), 0.0))]:
        prof = cProfile.Profile()
        prof.enable()
        pos, new_s = new_pipeline(name)
        prof.disable()
        secs = pstats.Stats(prof).total_tt
        d = np.diff(pos, axis=0)
        out.append({
            "video": name, "frames": N, "resolution": [W, H],
            "mean_dx_dy": d.mean(axis=0).tolist(),
            "max_component_error_px": float(np.abs(d - np.array(truth)).max()),
            "old_suspicion": suspicion(old_positions(name), FPS),
            "new_suspicion": new_s,
            "profile_seconds": secs,
            "pipeline_frames_per_second": N / secs,
        })
    with open("verify_screen_motion.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
