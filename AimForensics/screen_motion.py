"""
畫面位移軌跡（v2 的預設軌跡擷取方法）

FPS 遊戲的準心固定在畫面中央，玩家瞄準時真正會動的是畫面。
這裡用相鄰兩格的相位相關（cv2.phaseCorrelate）估計整個畫面平移了多少，
再把每格的位移累加起來，當作瞄準軌跡交給原本的 TrajectoryAnalyzer。

限制：只估計「整個畫面的平移」，不是真實滑鼠輸入或視角角度；
3D 旋轉、視差、移動中的敵人、固定的 HUD、後座力與切換鏡頭都會干擾；
累加也會讓誤差慢慢漂移。
"""
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

MIN_RESPONSE = 0.05     # 相位相關的峰值太低代表兩格對不上（例如切鏡）
MIN_TEXTURE_STD = 1.0   # 灰階標準差太低代表畫面幾乎沒有紋理，無法估計位移


def _to_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return gray.astype(np.float32)


def extract_screen_motion(video_path: str,
                          frame_interval: int = 1,
                          progress_callback: Optional[Callable[[float], None]] = None,
                          start_frame: int = 0,
                          end_frame: Optional[int] = None,
                          ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], float]:
    """
    回傳 (positions, timestamps, sample_frames, effective_fps)。

    positions：形狀 (N, 2)，第一格為 (0, 0)，之後為累加的畫面位移（像素）；
               正 x 代表畫面內容往右移。
    每一格都算位移再累加，最後才依 frame_interval 抽樣，
    所以抽樣不會漏掉中間的移動；effective_fps = fps / frame_interval。
    """
    if frame_interval < 1:
        raise ValueError("frame_interval 必須 ≥ 1")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("無法開啟影片檔案（編碼不支援或檔案損毀）")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("影片 FPS 無效")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    stop = end_frame if end_frame is not None else (total if total > 0 else None)

    positions = [(0.0, 0.0)]
    samples: List[np.ndarray] = []
    prev = None
    window = None
    x = y = 0.0
    idx = start_frame
    sample_every = max(1, ((stop - start_frame) if stop else 600) // 10)

    while stop is None or idx < stop:
        ok, frame = cap.read()
        if not ok:
            break
        gray = _to_gray(frame)
        if prev is None:
            if gray.std() < MIN_TEXTURE_STD:
                cap.release()
                raise ValueError("畫面幾乎沒有紋理，無法估計位移")
            window = cv2.createHanningWindow(gray.shape[::-1], cv2.CV_32F)
        else:
            if gray.shape != prev.shape:
                cap.release()
                raise ValueError("影格尺寸改變，無法估計位移")
            # 傳入副本：有些 OpenCV 版本會在原陣列上直接套用視窗函數，下一格再用就會出錯
            (dx, dy), response = cv2.phaseCorrelate(prev.copy(), gray.copy(), window)
            if response < MIN_RESPONSE:
                cap.release()
                raise ValueError(f"第 {idx} 格與前一格對不上（相關峰值 {response:.3f}），可能是切鏡或畫面無紋理")
            x += dx
            y += dy
            positions.append((x, y))
        if len(samples) < 10 and (idx - start_frame) % sample_every == 0:
            samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prev = gray
        idx += 1
        if progress_callback and stop:
            progress_callback(min(1.0, (idx - start_frame) / max(1, stop - start_frame)))
    cap.release()

    if len(positions) < 2:
        raise ValueError("可用影格太少")
    pos = np.array(positions, dtype=float)[::frame_interval]
    ts = (start_frame + np.arange(len(positions))[::frame_interval]) / fps
    return pos, ts, samples, fps / frame_interval
