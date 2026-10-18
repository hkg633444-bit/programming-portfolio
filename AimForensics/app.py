"""
================================================================================
瞄準行為分析器 - FPS 外掛偵測研究工具
================================================================================
基於軌跡分析的自瞄外掛偵測系統，透過分析準心移動軌跡來區分
人類玩家與自瞄輔助軟體。

核心偵測方法：
1. 角速度分析 - 偵測瞬間瞄準（Snap）
2. 黏著度指數 - 測量目標追蹤精準度
3. 抖動分析 - 分析微調修正模式
4. 平滑度分析 - 評估移動自然程度

作者：[Ning]
日期：2026年1月
================================================================================
"""

# ==============================================================================
# 匯入模組與設定
# ==============================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 隱藏 TensorFlow 警告訊息

import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import shutil


def _save_uploaded_to_tempfile(uploaded_file, suffix: str = '.mp4') -> str:
    """將 Streamlit UploadedFile 以串流方式寫入暫存檔，避免大檔一次讀進記憶體。

    回傳暫存檔路徑；呼叫端負責 os.unlink(path)。
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        shutil.copyfileobj(uploaded_file, tmp, length=1024 * 1024)  # 1MB chunks
    finally:
        tmp.close()
    return tmp.name
import json
import io
import zipfile
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass, field
import cv2
from PIL import Image

# TensorFlow 深度學習框架
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# SciPy 訊號處理
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# Plotly 視覺化
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Scikit-learn 機器學習工具
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# STREAMLIT 頁面設定
# ==============================================================================

st.set_page_config(
    page_title="Aim Behavior Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 駭客／遊戲風格深黑主題 + iOS 液態玻璃按鈕
# ==============================================================================
CUSTOM_CSS = """
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Orbitron:wght@500;700;900&display=swap" rel="stylesheet">
<style>
    /* ============ 全站底色：深黑 + 格狀背景 + 霓虹暈 ============ */
    .stApp {
        background:
            radial-gradient(ellipse 80% 50% at 50% -10%, rgba(56,189,248,0.08), transparent 60%),
            radial-gradient(ellipse 60% 50% at 100% 100%, rgba(120,0,255,0.07), transparent 60%),
            radial-gradient(ellipse 50% 40% at 0% 80%, rgba(0,180,255,0.06), transparent 60%),
            linear-gradient(180deg, #04040a 0%, #07070f 100%);
        background-attachment: fixed;
        color: #e6e6f0;
        font-family: 'JetBrains Mono', 'Consolas', monospace;
    }
    /* 細格線網格疊層（hacker-grid） */
    .stApp::before {
        content: "";
        position: fixed; inset: 0;
        background-image:
            linear-gradient(rgba(56,189,248,0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(56,189,248,0.035) 1px, transparent 1px);
        background-size: 42px 42px;
        pointer-events: none;
        z-index: 0;
        mask-image: radial-gradient(ellipse at center, black 30%, transparent 90%);
    }
    /* 掃描線疊層 */
    .stApp::after {
        content: "";
        position: fixed; inset: 0;
        background: repeating-linear-gradient(
            180deg,
            rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px,
            transparent 1px, transparent 3px
        );
        pointer-events: none;
        z-index: 0;
    }
    .main .block-container { position: relative; z-index: 1; }

    /* ============ 標題霓虹字 ============ */
    h1, h2, h3 {
        font-family: 'Orbitron', 'JetBrains Mono', monospace !important;
        letter-spacing: 0.04em;
        color: #e8fff8 !important;
        text-shadow:
            0 0 8px rgba(56,189,248,0.35),
            0 0 22px rgba(56,189,248,0.18);
    }
    h1 { font-weight: 900 !important; }
    h4, h5, h6 { color: #b8ffe6 !important; letter-spacing: 0.03em; }

    /* 段落與小字 */
    p, label, span, div, li {
        font-family: 'JetBrains Mono', 'Consolas', monospace;
    }

    /* 指示器色塊 */
    .human-indicator { color: #38bdf8; font-weight: 700; text-shadow: 0 0 8px rgba(56,189,248,0.5); }
    .suspicious-indicator { color: #ffd23f; font-weight: 700; text-shadow: 0 0 8px rgba(255,210,63,0.5); }
    .cheating-indicator { color: #ff3b6b; font-weight: 700; text-shadow: 0 0 8px rgba(255,59,107,0.5); }

    /* ============ 液態玻璃按鈕（Liquid Glass） ============ */
    .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
        position: relative;
        border-radius: 9999px !important;
        padding: 0.6em 1.6em !important;
        font-family: 'Orbitron', 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em;
        color: #f5fff9 !important;
        background:
            radial-gradient(ellipse 120% 80% at 50% 0%, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.04) 35%, transparent 55%),
            linear-gradient(180deg, rgba(28,28,38,0.92) 0%, rgba(6,6,12,0.98) 100%) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        box-shadow:
            0 0 0 1px rgba(255,255,255,0.04) inset,
            0 1px 0 rgba(255,255,255,0.35) inset,
            0 -10px 22px rgba(0,0,0,0.55) inset,
            -3px 0 14px rgba(255, 60, 200, 0.16),
            3px 0 14px rgba(0, 220, 255, 0.16),
            0 -3px 14px rgba(255, 200, 60, 0.10),
            0 3px 14px rgba(60, 120, 255, 0.12),
            0 10px 26px rgba(0,0,0,0.55);
        transition: transform 0.18s cubic-bezier(.2,.9,.25,1.2),
                    box-shadow 0.25s ease,
                    filter 0.25s ease;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        overflow: hidden;
    }
    /* 高光反射 */
    .stButton > button::before, .stDownloadButton > button::before, .stFormSubmitButton > button::before {
        content: "";
        position: absolute;
        left: 8%; right: 8%; top: 6%;
        height: 40%;
        border-radius: 9999px;
        background: linear-gradient(180deg, rgba(255,255,255,0.35), rgba(255,255,255,0.0));
        filter: blur(2px);
        pointer-events: none;
        opacity: 0.85;
    }
    /* 底部陰影折射 */
    .stButton > button::after, .stDownloadButton > button::after, .stFormSubmitButton > button::after {
        content: "";
        position: absolute;
        left: 12%; right: 12%; bottom: 8%;
        height: 24%;
        border-radius: 9999px;
        background: radial-gradient(ellipse at center, rgba(56,189,248,0.18), transparent 70%);
        filter: blur(6px);
        pointer-events: none;
        opacity: 0.6;
    }
    /* 滑過時：更亮、放大、色散加強（水珠感） */
    .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-1px) scale(1.025);
        filter: brightness(1.15);
        box-shadow:
            0 0 0 1px rgba(255,255,255,0.08) inset,
            0 1px 0 rgba(255,255,255,0.5) inset,
            0 -10px 24px rgba(0,0,0,0.6) inset,
            -5px 0 22px rgba(255, 60, 200, 0.32),
            5px 0 22px rgba(0, 220, 255, 0.32),
            0 -4px 22px rgba(255, 200, 60, 0.18),
            0 4px 22px rgba(60, 120, 255, 0.22),
            0 14px 36px rgba(0,0,0,0.65) !important;
    }
    /* 按下：水滴壓縮 */
    .stButton > button:active, .stDownloadButton > button:active, .stFormSubmitButton > button:active {
        transform: translateY(1px) scale(0.96);
        filter: brightness(0.95);
    }
    /* 聚焦不要 outline */
    .stButton > button:focus, .stDownloadButton > button:focus, .stFormSubmitButton > button:focus {
        outline: none !important;
        box-shadow:
            0 0 0 2px rgba(56,189,248,0.35) inset,
            0 1px 0 rgba(255,255,255,0.45) inset,
            0 -10px 22px rgba(0,0,0,0.55) inset,
            0 0 28px rgba(56,189,248,0.30) !important;
    }

    /* ============ 標籤頁 Tabs（玻璃膠囊） ============ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(10,10,18,0.6);
        padding: 6px;
        border-radius: 9999px;
        border: 1px solid rgba(56,189,248,0.12);
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(180deg, rgba(30,30,42,0.75), rgba(10,10,16,0.9));
        border-radius: 9999px !important;
        padding: 8px 20px !important;
        color: #b8ffe6 !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em;
        border: 1px solid rgba(255,255,255,0.06);
        transition: all 0.25s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        border-color: rgba(56,189,248,0.35);
        color: #38bdf8 !important;
        text-shadow: 0 0 10px rgba(56,189,248,0.45);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, rgba(56,189,248,0.18), rgba(30,64,175,0.05)) !important;
        color: #38bdf8 !important;
        border-color: rgba(56,189,248,0.5) !important;
        box-shadow: 0 0 18px rgba(56,189,248,0.25), inset 0 1px 0 rgba(255,255,255,0.15);
    }

    /* ============ 側邊欄 ============ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8,8,14,0.96), rgba(12,12,20,0.96)) !important;
        border-right: 1px solid rgba(56,189,248,0.12);
    }
    section[data-testid="stSidebar"]::before {
        content: "";
        position: absolute; inset: 0;
        background-image:
            linear-gradient(rgba(56,189,248,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(56,189,248,0.04) 1px, transparent 1px);
        background-size: 28px 28px;
        pointer-events: none;
    }

    /* ============ Metric 卡片 ============ */
    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(18,18,28,0.9), rgba(6,6,12,0.95));
        border: 1px solid rgba(56,189,248,0.15);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.08),
            0 0 24px rgba(56,189,248,0.06),
            0 6px 18px rgba(0,0,0,0.5);
    }
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', monospace !important;
        color: #38bdf8 !important;
        text-shadow: 0 0 10px rgba(56,189,248,0.4);
    }
    [data-testid="stMetricLabel"] {
        color: #9aa5b5 !important;
        letter-spacing: 0.08em;
    }

    /* ============ 輸入元件（選單、輸入框、滑桿） ============ */
    .stSelectbox > div > div, .stTextInput > div > div, .stNumberInput > div > div,
    .stTextArea > div > div {
        background: linear-gradient(180deg, rgba(22,22,32,0.9), rgba(8,8,14,0.95)) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
    }
    .stSelectbox > div > div:hover, .stTextInput > div > div:hover {
        border-color: rgba(56,189,248,0.35) !important;
    }
    /* 滑桿軌道與握把 */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: radial-gradient(circle at 30% 30%, #ffffff 0%, #bae6fd 35%, #38bdf8 70%, #1e40af 100%) !important;
        box-shadow:
            0 0 0 2px rgba(56,189,248,0.4),
            0 0 14px rgba(56,189,248,0.5),
            inset 0 2px 2px rgba(255,255,255,0.6) !important;
        border: none !important;
    }

    /* ============ 檔案上傳 ============ */
    [data-testid="stFileUploader"] section {
        background: linear-gradient(180deg, rgba(18,18,28,0.7), rgba(6,6,12,0.9)) !important;
        border: 1px dashed rgba(56,189,248,0.35) !important;
        border-radius: 14px !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 26px rgba(56,189,248,0.18);
    }

    /* ============ 展開區塊 ============ */
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        background: linear-gradient(90deg, rgba(14,14,22,0.9), rgba(8,8,14,0.9)) !important;
        border: 1px solid rgba(56,189,248,0.15) !important;
        border-radius: 12px !important;
        font-family: 'Orbitron', monospace !important;
        letter-spacing: 0.05em;
    }

    /* ============ 進度條 ============ */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #38bdf8, #60a5fa) !important;
        box-shadow: 0 0 16px rgba(56,189,248,0.5);
    }

    /* ============ DataFrame ============ */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid rgba(56,189,248,0.15);
        border-radius: 12px;
        overflow: hidden;
    }

    /* ============ 訊息框 ============ */
    .stAlert {
        border-radius: 12px !important;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.08) !important;
    }

    /* ============ 自訂標題橫幅 ============ */
    .main-header {
        text-align: center;
        padding: 1.4rem 1rem;
        background:
            radial-gradient(ellipse at center, rgba(56,189,248,0.15) 0%, transparent 70%),
            linear-gradient(135deg, #080812 0%, #0f0f1a 100%);
        border: 1px solid rgba(56,189,248,0.25);
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.08),
            0 0 32px rgba(56,189,248,0.12),
            0 10px 30px rgba(0,0,0,0.55);
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(15,15,25,0.9) 0%, rgba(8,8,14,0.95) 100%);
        border-radius: 14px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(56,189,248,0.18);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 6px 18px rgba(0,0,0,0.4);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 資料類別 - 結構化資料容器
# ==============================================================================

@dataclass
class CrosshairPosition:
    """
    表示單一影格中偵測到的準心位置。
    
    屬性說明：
        x: 水平位置（像素）
        y: 垂直位置（像素）
        confidence: 偵測信心分數（0-1）
        frame_number: 影格索引
        timestamp: 從影片開始的時間（秒）
        method: 使用的偵測方法（'center', 'edge'）
    """
    x: float
    y: float
    confidence: float
    frame_number: int
    timestamp: float
    method: str


@dataclass
class MotionFeatures:
    """
    從軌跡資料計算的運動特徵。
    
    這些特徵用於區分人類與自瞄外掛行為。
    
    屬性說明：
        velocity: 每個影格的速度向量 (dx/dt, dy/dt)
        speed: 純量速度（速度向量的大小）
        acceleration: 速度變化率
        angular_velocity: 方向變化率（度/秒）
        jitter: 移動中的高頻雜訊
        smoothness: 軌跡平滑度分數
        mean_speed: 平均移動速度
        max_speed: 最大速度
        speed_variance: 速度變異數（人類有高變異）
        direction_changes: 顯著方向改變次數
    """
    velocity: np.ndarray
    speed: np.ndarray
    acceleration: np.ndarray
    angular_velocity: np.ndarray
    jitter: float
    smoothness: float
    mean_speed: float
    max_speed: float
    speed_variance: float
    direction_changes: int


@dataclass
class TrajectorySegment:
    """
    用於視窗化分析的軌跡片段。
    
    軌跡被分割成重疊的視窗進行時序分析。
    """
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    positions: np.ndarray
    features: Optional[MotionFeatures] = None


@dataclass
class AnomalyScores:
    """
    軌跡片段的異常偵測分數。
    
    每個分數範圍從 0（正常）到 1（高度可疑）。
    
    屬性說明：
        stickiness_index: 瞄準追蹤目標的精準程度（高=可疑）
        snap_score: 瞬間瞄準跳躍頻率（高=可疑）
        smoothness_anomaly: 不自然的平滑程度（高=可疑）
        direction_entropy: 方向變化的資訊熵（低=可疑，表示有規律模式）
        human_likelihood: 人類行為機率（高=可能是人類）
        overall_suspicion: 綜合可疑分數
        anomaly_details: 額外指標細節
    """
    stickiness_index: float
    snap_score: float
    smoothness_anomaly: float
    direction_entropy: float
    human_likelihood: float
    overall_suspicion: float
    anomaly_details: Dict[str, float]


@dataclass 
class SnapEvent:
    """
    偵測到的瞬間瞄準（Snap/Flick）事件。
    
    Snap 是突然的大幅瞄準移動，可能表示自瞄外掛啟動。
    """
    frame_index: int
    timestamp: float
    angular_velocity: float  # 度/秒
    speed_before: float
    speed_after: float
    snap_duration_frames: int
    suspicion_score: float


@dataclass
class TrainingConfig:
    """模型訓練的設定參數。"""
    sequence_length: int = 60
    feature_dim: int = 10
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10


@dataclass
class ModelMetrics:
    """模型評估指標。"""
    accuracy: float
    loss: float
    val_accuracy: float
    val_loss: float
    precision: float
    recall: float
    f1_score: float


# ==============================================================================
# 準心偵測器
# 使用電腦視覺從影片影格中提取準心位置
# ==============================================================================

class CrosshairDetector:
    """
    使用電腦視覺偵測 FPS 遊戲畫面中的準心位置。
    
    偵測方法：
    1. 中心法：假設準心在螢幕中央（大多數 FPS 遊戲）
    2. 邊緣偵測：使用邊緣分析找十字形圖案
    
    對於大多數 FPS 遊戲，中心法已經足夠，因為
    準心在正常遊玩時固定在螢幕中央。
    """
    
    def __init__(self, frame_size: Tuple[int, int] = (1920, 1080)):
        """
        使用預期的影格尺寸初始化偵測器。
        
        參數：
            frame_size: 影片影格的 (寬度, 高度)
        """
        self.frame_width, self.frame_height = frame_size
        self.center_x = frame_size[0] // 2
        self.center_y = frame_size[1] // 2
        
    def detect_center_based(self, frame: np.ndarray, frame_number: int, 
                           timestamp: float) -> CrosshairPosition:
        """
        中心法偵測 - 假設準心在螢幕中央。
        
        這是標準 FPS 遊玩最可靠的方法，
        準心固定在螢幕中央。
        
        參數：
            frame: 影片的 BGR 影像影格
            frame_number: 當前影格索引
            timestamp: 時間（秒）
            
        回傳：
            螢幕中央的 CrosshairPosition
        """
        h, w = frame.shape[:2]
        return CrosshairPosition(
            x=w / 2,
            y=h / 2,
            confidence=1.0, 
            frame_number=frame_number,
            timestamp=timestamp,
            method='center'
        )
    
    def detect_by_edge_pattern(self, frame: np.ndarray, frame_number: int,
                               timestamp: float) -> CrosshairPosition:
        """
        邊緣法偵測 - 尋找十字形圖案。
        
        使用 Canny 邊緣偵測和霍夫線轉換找出
        螢幕中央附近交叉的水平和垂直線。
        
        適用於觀戰視角或自訂準心覆蓋。
        
        參數：
            frame: BGR 影像影格
            frame_number: 當前影格索引
            timestamp: 時間（秒）
            
        回傳：
            偵測到的準心位置 CrosshairPosition
        """
        # 轉換為灰階
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape
        
        # 聚焦中央區域（準心應該在中心附近）
        roi_size = min(h, w) // 4
        roi_x1 = w//2 - roi_size//2
        roi_y1 = h//2 - roi_size//2
        roi = gray[roi_y1:roi_y1+roi_size, roi_x1:roi_x1+roi_size]
        
        # 使用 Canny 演算法偵測邊緣
        edges = cv2.Canny(roi, 50, 150)
        
        # 使用機率霍夫轉換找線條
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, 
                                minLineLength=10, maxLineGap=5)
        
        if lines is not None and len(lines) >= 2:
            horizontal_lines = []
            vertical_lines = []
            
            # 將線條分類為水平或垂直
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                
                if angle < 30 or angle > 150:  # 水平
                    horizontal_lines.append(line[0])
                elif 60 < angle < 120:  # 垂直
                    vertical_lines.append(line[0])
            
            # 找交叉點
            if horizontal_lines and vertical_lines:
                h_center_y = np.mean([((l[1]+l[3])/2) for l in horizontal_lines])
                v_center_x = np.mean([((l[0]+l[2])/2) for l in vertical_lines])
                
                return CrosshairPosition(
                    x=roi_x1 + v_center_x,
                    y=roi_y1 + h_center_y,
                    confidence=0.7,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    method='edge_pattern'
                )
        
        # 如果找不到準心圖案，退回到中心
        return CrosshairPosition(
            x=w/2,
            y=h/2,
            confidence=0.3,
            frame_number=frame_number,
            timestamp=timestamp,
            method='edge_pattern_fallback'
        )
    
    def detect(self, frame: np.ndarray, frame_number: int, timestamp: float,
               method: str = 'center') -> CrosshairPosition:
        """
        主要偵測方法 - 分派到特定演算法。
        
        參數：
            frame: BGR 影像影格
            frame_number: 影格索引
            timestamp: 時間（秒）
            method: 偵測方法（'center' 或 'edge'）
            
        回傳：
            偵測到的 CrosshairPosition
        """
        if method == 'center':
            return self.detect_center_based(frame, frame_number, timestamp)
        elif method == 'edge':
            return self.detect_by_edge_pattern(frame, frame_number, timestamp)
        else:
            return self.detect_center_based(frame, frame_number, timestamp)
    
    def extract_trajectory_from_video(self, video_path: str, 
                                      frame_interval: int = 1,
                                      method: str = 'center') -> List[CrosshairPosition]:
        """
        從影片檔案提取完整的準心軌跡。
        
        參數：
            video_path: 影片檔案路徑
            frame_interval: 每 N 個影格處理一次（1 = 全部影格）
            method: 使用的偵測方法
            
        回傳：
            每個處理過的影格的 CrosshairPosition 列表
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"無法開啟影片：{video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        trajectory = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                position = self.detect(frame, frame_count, timestamp, method)
                trajectory.append(position)
            
            frame_count += 1
        
        cap.release()
        return trajectory


# ==============================================================================
# 軌跡分析器
# 從準心軌跡資料計算運動特徵
# ==============================================================================

class TrajectoryAnalyzer:
    """
    分析準心軌跡以提取運動特徵。
    
    偵測非人類瞄準的關鍵指標：
    
    1. 速度/速率：準心移動多快
       - 人類：可變速度，有加速/減速階段
       - 自瞄：通常恆定速度或瞬間速度變化
    
    2. 角速度：方向變化率（度/秒）
       - 人類：平滑方向變化，通常 < 500°/s
       - 自瞄：瞬間 snap > 1000°/s 無過渡影格
    
    3. 抖動：移動中的高頻雜訊
       - 人類：手抖的自然微調（0.5-5.0）
       - 自瞄：太平滑（< 0.2）或人工抖動模式
    
    4. 平滑度：整體軌跡平滑度
       - 人類：中等平滑（0.3-0.8）
       - 自瞄：機械平滑（> 0.9）或不自然抖動（< 0.1）
    """
    
    def __init__(self, fps: float = 60.0, pixel_to_degree: float = 0.022):
        """
        使用影片參數初始化分析器。
        
        參數：
            fps: 影片幀率（每秒影格數）
            pixel_to_degree: 像素到角度的轉換因子
        """
        self.fps = fps
        self.dt = 1.0 / fps  # 影格之間的時間
        self.pixel_to_degree = pixel_to_degree
        
    def compute_velocity(self, positions: np.ndarray) -> np.ndarray:
        """
        計算每個影格的速度向量 (dx/dt, dy/dt)。
        
        速度 = 位置變化 / 時間變化
        
        參數：
            positions: 形狀為 (N, 2) 的陣列，包含 [x, y] 座標
            
        回傳：
            形狀為 (N-1, 2) 的速度陣列
        """
        if len(positions) < 2:
            return np.array([])
        
        # 有限差分微分
        velocity = np.diff(positions, axis=0) / self.dt
        return velocity
    
    def compute_speed(self, velocity: np.ndarray) -> np.ndarray:
        """
        計算純量速度（速度向量的大小）。
        
        速度 = sqrt(vx^2 + vy^2)
        
        參數：
            velocity: 形狀為 (N, 2) 的陣列，包含 [vx, vy]
            
        回傳：
            形狀為 (N,) 的速度陣列
        """
        return np.linalg.norm(velocity, axis=1)
    
    def compute_acceleration(self, velocity: np.ndarray) -> np.ndarray:
        """
        計算加速度（速度變化率）。
        
        加速度 = dv/dt
        
        參數：
            velocity: 形狀為 (N, 2) 的陣列
            
        回傳：
            形狀為 (N-1, 2) 的加速度陣列
        """
        if len(velocity) < 2:
            return np.array([])
        
        return np.diff(velocity, axis=0) / self.dt
    
    def compute_angular_velocity(self, positions: np.ndarray) -> np.ndarray:
        """
        計算角速度（方向變化率），單位為度/秒。
        
        這對於偵測瞬間瞄準（snap aiming）至關重要 - 突然的大角度
        變化表示潛在的自瞄外掛啟動。
        
        公式：
            θ = arctan2(v1 × v2, v1 · v2)  # 向量之間的角度
            ω = |θ| / dt                    # 角速度
        
        參數：
            positions: 形狀為 (N, 2) 的陣列
            
        回傳：
            形狀為 (N-2,) 的角速度陣列，單位為度/秒
        """
        if len(positions) < 3:
            return np.array([])
        
        # 計算連續的位移向量
        v1 = positions[1:-1] - positions[:-2]  # 前一個位移
        v2 = positions[2:] - positions[1:-1]   # 當前位移
        
        # 向量大小
        len1 = np.linalg.norm(v1, axis=1)
        len2 = np.linalg.norm(v2, axis=1)
        
        # 只在兩個向量長度 > 0.1 像素時計算角度
        valid = (len1 > 0.1) & (len2 > 0.1)
        angles = np.zeros(len(v1))
        
        if np.any(valid):
            # 點積：v1 · v2 = |v1||v2|cos(θ)
            dot = np.sum(v1[valid] * v2[valid], axis=1)
            # 叉積（2D）：v1 × v2 = |v1||v2|sin(θ)
            cross = v1[valid, 0] * v2[valid, 1] - v1[valid, 1] * v2[valid, 0]
            # arctan2 給出帶符號角度
            angles[valid] = np.arctan2(cross, dot)
        
        angular_velocity = np.abs(angles) / self.dt
        angular_velocity_deg = np.degrees(angular_velocity)
        
        return angular_velocity_deg
    
    def compute_jitter(self, positions: np.ndarray, window_size: int = 5) -> float:
        """
        計算軌跡中的抖動（高頻雜訊）。
        
        抖動測量與平滑軌跡的偏差。
        
        人類瞄準有手抖產生的特徵微調。
        自瞄可能不自然地平滑或有人工抖動。
        
        公式：
            J = sqrt(mean((原始 - 平滑)^2))
        
        參數：
            positions: 形狀為 (N, 2) 的陣列
            window_size: 高斯平滑視窗大小
            
        回傳：
            抖動分數（越高 = 抖動越多）
        """
        if len(positions) < window_size + 1:
            return 0.0
        
        # 應用高斯平滑得到「理想」軌跡
        smoothed_x = gaussian_filter1d(positions[:, 0], sigma=window_size/3)
        smoothed_y = gaussian_filter1d(positions[:, 1], sigma=window_size/3)
        
        # 計算殘差（原始 - 平滑）
        residual_x = positions[:, 0] - smoothed_x
        residual_y = positions[:, 1] - smoothed_y
        
        # 殘差的均方根
        jitter = np.sqrt(np.mean(residual_x**2 + residual_y**2))
        
        return jitter
    
    def compute_smoothness(self, acceleration: np.ndarray) -> float:
        """
        使用加速度變異數計算軌跡平滑度。
        
        較低的加速度變異數 = 較平滑的移動。
        
        公式：
            平滑度 = 1 / (1 + variance(|加速度|) / 1000)
        
        參數：
            acceleration: 形狀為 (N, 2) 的陣列
            
        回傳：
            平滑度分數（0-1，越高 = 越平滑）
        """
        if len(acceleration) == 0:
            return 0.0
        
        acc_magnitude = np.linalg.norm(acceleration, axis=1)
        variance = np.var(acc_magnitude)
        
        # 正規化到 0-1 範圍
        smoothness = 1.0 / (1.0 + variance / 1000)
        
        return smoothness
    
    def count_direction_changes(self, velocity: np.ndarray, 
                                threshold_degrees: float = 30.0) -> int:
        """
        計算軌跡中的顯著方向改變次數。
        
        頻繁的方向改變可能表示人類追蹤行為。
        
        參數：
            velocity: 形狀為 (N, 2) 的陣列
            threshold_degrees: 計為方向改變的最小角度
            
        回傳：
            方向改變次數
        """
        if len(velocity) < 2:
            return 0
        
        threshold_rad = np.radians(threshold_degrees)
        len_v = np.linalg.norm(velocity, axis=1)
        valid = len_v > 0.1
        
        changes = 0
        prev_angle = None
        
        for i, v in enumerate(velocity):
            if not valid[i]:
                continue
            
            angle = np.arctan2(v[1], v[0])
            
            if prev_angle is not None:
                diff = np.abs(angle - prev_angle)
                diff = min(diff, 2*np.pi - diff)
                
                if diff > threshold_rad:
                    changes += 1
            
            prev_angle = angle
        
        return changes
    
    def analyze_trajectory(self, positions: np.ndarray) -> MotionFeatures:
        """
        對軌跡執行完整的運動分析。
        
        計算分類所需的所有運動特徵。
        
        參數：
            positions: 形狀為 (N, 2) 的陣列，包含 [x, y] 座標
            
        回傳：
            包含所有計算指標的 MotionFeatures 資料類別
        """
        velocity = self.compute_velocity(positions)
        speed = self.compute_speed(velocity) if len(velocity) > 0 else np.array([0.0])
        acceleration = self.compute_acceleration(velocity)
        angular_velocity = self.compute_angular_velocity(positions)
        jitter = self.compute_jitter(positions)
        smoothness = self.compute_smoothness(acceleration) if len(acceleration) > 0 else 0.0
        direction_changes = self.count_direction_changes(velocity)
        
        return MotionFeatures(
            velocity=velocity,
            speed=speed,
            acceleration=acceleration,
            angular_velocity=angular_velocity,
            jitter=jitter,
            smoothness=smoothness,
            mean_speed=float(np.mean(speed)) if len(speed) > 0 else 0.0,
            max_speed=float(np.max(speed)) if len(speed) > 0 else 0.0,
            speed_variance=float(np.var(speed)) if len(speed) > 0 else 0.0,
            direction_changes=direction_changes
        )
    
    def segment_trajectory(self, positions: np.ndarray, timestamps: np.ndarray,
                          window_size: int = 60, overlap: int = 30) -> List[TrajectorySegment]:
        """
        將軌跡分割成重疊的視窗進行分析。
        
        視窗化分析允許偵測特定時間段中的可疑行為，
        而不只是整體模式。
        
        參數：
            positions: 形狀為 (N, 2) 的陣列
            timestamps: 形狀為 (N,) 的陣列，包含影格時間戳
            window_size: 每段的影格數（60 = 60fps 下 1 秒）
            overlap: 段之間重疊的影格數
            
        回傳：
            包含計算特徵的 TrajectorySegment 物件列表
        """
        segments = []
        step = window_size - overlap
        n_frames = len(positions)
        
        for start in range(0, n_frames - window_size + 1, step):
            end = start + window_size
            
            segment = TrajectorySegment(
                start_frame=start,
                end_frame=end,
                start_time=timestamps[start],
                end_time=timestamps[end - 1],
                positions=positions[start:end]
            )
            
            # 計算此段的特徵
            segment.features = self.analyze_trajectory(segment.positions)
            segments.append(segment)
        
        return segments
    
    def detect_snap_events(self, angular_velocity: np.ndarray, 
                          threshold_deg_per_sec: float = 500.0) -> List[int]:
        """
        根據角速度尖峰偵測潛在的 snap/flick 事件。
        
        Snap 是超過人類反應能力的突然瞄準移動。
        
        參數：
            angular_velocity: 角速度陣列，單位為度/秒
            threshold_deg_per_sec: snap 偵測閾值（預設 500°/s）
            
        回傳：
            偵測到 snap 的影格索引列表
        """
        snap_indices = np.where(angular_velocity > threshold_deg_per_sec)[0]
        return snap_indices.tolist()
    
    def compute_frequency_spectrum(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        計算軌跡運動的頻譜。
        
        人類瞄準有手抖產生的特徵頻率模式。
        自瞄可能顯示異常的頻率特徵。
        
        使用 Welch 方法進行功率頻譜密度估計。
        
        參數：
            positions: 形狀為 (N, 2) 的陣列
            
        回傳：
            (頻率, 功率頻譜) 的元組
        """
        if len(positions) < 4:
            return np.array([]), np.array([])
        
        # 去趨勢訊號
        x_detrended = positions[:, 0] - np.mean(positions[:, 0])
        y_detrended = positions[:, 1] - np.mean(positions[:, 1])
        
        # 合併大小
        combined = np.sqrt(x_detrended**2 + y_detrended**2)
        
        # 使用 Welch 方法計算功率頻譜密度
        freqs, psd = signal.welch(combined, fs=self.fps, nperseg=min(len(combined), 64))
        
        return freqs, psd
    
    def get_feature_vector(self, features: MotionFeatures) -> np.ndarray:
        """
        將 MotionFeatures 轉換為固定大小的向量供機器學習模型使用。
        
        建立適合訓練的 10 維特徵向量。
        
        回傳：
            形狀為 (10,) 的特徵向量
        """
        return np.array([
            features.mean_speed,
            features.max_speed,
            features.speed_variance,
            features.jitter,
            features.smoothness,
            features.direction_changes,
            np.mean(features.angular_velocity) if len(features.angular_velocity) > 0 else 0,
            np.max(features.angular_velocity) if len(features.angular_velocity) > 0 else 0,
            np.std(features.angular_velocity) if len(features.angular_velocity) > 0 else 0,
            np.mean(np.linalg.norm(features.acceleration, axis=1)) if len(features.acceleration) > 0 else 0
        ])


# ==============================================================================
# 異常偵測器
# 基於規則的可疑瞄準行為偵測
# ==============================================================================

class AnomalyDetector:
    """
    偵測區分人類玩家與自瞄外掛的瞄準異常。
    
    人類基準值（基於研究）：
    - 平均速度：50-300 像素/秒
    - 最大速度：200-1500 像素/秒
    - 速度變異數：1000-50000（人類有可變速度）
    - 抖動：0.5-5.0（自然手抖）
    - 平滑度：0.3-0.8（中等平滑）
    - 角速度平均：10-150 度/秒
    - 角速度最大：100-800 度/秒
    
    自瞄指標：
    - 完美追蹤（黏著度 > 0.95）
    - 瞬間 snap（角速度 > 1000°/s）
    - 零抖動（< 0.2）
    - 恆定速度（變異數 < 100）
    """
    
    # 人類基準範圍（最小, 最大）
    HUMAN_BASELINE = {
        'mean_speed': (50, 300),
        'max_speed': (200, 1500),
        'speed_variance': (1000, 50000),
        'jitter': (0.5, 5.0),
        'smoothness': (0.3, 0.8),
        'angular_velocity_mean': (10, 150),
        'angular_velocity_max': (100, 800),
    }
    
    # 自瞄指標閾值
    AIMBOT_INDICATORS = {
        'perfect_tracking': 0.95,
        'instant_snap_threshold': 500,  # 度/秒
        'zero_jitter_threshold': 0.2,
        'constant_speed_variance': 100,
    }
    
    def __init__(self, sensitivity: float = 0.5):
        """
        使用敏感度等級初始化偵測器。
        
        參數：
            sensitivity: 偵測敏感度（0-1）
                        較高 = 更多誤報
                        較低 = 更多漏報
        """
        self.sensitivity = sensitivity
        
        # 邏輯迴歸相關屬性
        self.lr_model: Optional[LogisticRegression] = None
        self.lr_scaler: Optional[StandardScaler] = None
        self.lr_weights: Optional[Dict[str, float]] = None
        self.use_learned_weights: bool = False
        
    def compute_stickiness_index(self, positions: np.ndarray, 
                                  target_positions: Optional[np.ndarray] = None) -> float:
        """
        計算黏著度指數 - 瞄準「黏」在目標上的程度。
        
        沒有目標資料時，根據軌跡特性估計：
        - 低抖動的靜止瞄準 = 高黏著度
        - 一致的移動方向 = 高黏著度
        
        公式（有目標時）：
            S = 1 / (1 + variance(到目標距離) / 10)
        
        參數：
            positions: 準心位置 (N, 2)
            target_positions: 可選的目標/敵人位置 (N, 2)
            
        回傳：
            黏著度指數（0-1，越高 = 越可疑）
        """
        if len(positions) < 3:
            return 0.5
        
        # 如果有目標位置，計算追蹤準確度
        if target_positions is not None and len(target_positions) == len(positions):
            offsets = positions - target_positions
            offset_variance = np.var(np.linalg.norm(offsets, axis=1))
            stickiness = 1.0 / (1.0 + offset_variance / 10)
            return float(stickiness)
        
        # 從軌跡特性計算黏著度
        velocity = np.diff(positions, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        
        movement_segments = speed > 1.0
        stationary_segments = speed <= 1.0
        
        # 如果瞄準靜止且變異數低，則高黏著度
        if np.sum(stationary_segments) > len(speed) * 0.3:
            stationary_positions = positions[:-1][stationary_segments]
            if len(stationary_positions) > 1:
                stationary_variance = np.var(stationary_positions, axis=0).sum()
                lock_quality = 1.0 / (1.0 + stationary_variance / 5)
                return float(lock_quality)
        
        # 從移動方向一致性計算
        if np.sum(movement_segments) > 0:
            moving_velocity = velocity[movement_segments]
            direction_consistency = self._compute_direction_consistency(moving_velocity)
            return float(direction_consistency * 0.7)
        
        return 0.5
    
    def _compute_direction_consistency(self, velocity: np.ndarray) -> float:
        """
        計算移動方向的一致性。
        
        高一致性可能表示鎖定行為。
        """
        if len(velocity) < 2:
            return 0.5
        
        # 正規化為單位向量
        norms = np.linalg.norm(velocity, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        unit_vectors = velocity / norms
        
        # 平均方向向量
        mean_direction = np.mean(unit_vectors, axis=0)
        mean_norm = np.linalg.norm(mean_direction)
        
        return float(mean_norm)
    
    def compute_snap_score(self, angular_velocity: np.ndarray, 
                           speed: np.ndarray,
                           acceleration: np.ndarray) -> Tuple[float, List[SnapEvent]]:
        """
        計算 snap 分數並偵測個別 snap 事件。
        
        Snap 特徵（自瞄指標）：
        - 突然的高角速度尖峰（> 500°/s）
        - 瞬間加速到最大速度
        - 突然減速到停止
        - 沒有自然的「跟隨」動作
        
        參數：
            angular_velocity: 角速度，單位為度/秒
            speed: 移動速度陣列
            acceleration: 加速度陣列
            
        回傳：
            (整體_snap_分數, SnapEvent 列表) 的元組
        """
        snap_events = []
        
        if len(angular_velocity) < 3:
            return 0.0, snap_events
        
        # 根據敏感度調整閾值
        base_threshold = self.AIMBOT_INDICATORS['instant_snap_threshold']
        threshold = base_threshold * (1.5 - self.sensitivity)
        
        # 找超過閾值的角速度尖峰
        potential_snaps = np.where(angular_velocity > threshold)[0]
        
        snap_scores = []
        
        for idx in potential_snaps:
            if len(speed) > idx + 2 and idx > 0:
                speed_before = speed[idx - 1] if idx > 0 else 0
                speed_after = speed[min(idx + 1, len(speed) - 1)]
                speed_at_snap = speed[min(idx, len(speed) - 1)]
                
                # 檢查瞬間啟動（可疑）
                instant_start = speed_at_snap > speed_before * 3
                # 檢查瞬間停止（可疑）
                instant_stop = speed_after < speed_at_snap * 0.3
                
                suspicion = 0.5
                if instant_start:
                    suspicion += 0.25
                if instant_stop:
                    suspicion += 0.25
                
                # 按角速度大小縮放
                suspicion *= min(1.0, angular_velocity[idx] / 2000)
                
                snap_event = SnapEvent(
                    frame_index=idx,
                    timestamp=idx / 60.0,
                    angular_velocity=float(angular_velocity[idx]),
                    speed_before=float(speed_before),
                    speed_after=float(speed_after),
                    snap_duration_frames=1,
                    suspicion_score=float(suspicion)
                )
                snap_events.append(snap_event)
                snap_scores.append(suspicion)
        
        # 根據 snap 的頻率和嚴重性計算整體分數
        if snap_scores:
            overall_score = np.mean(snap_scores) * min(1.0, len(snap_scores) / 3)
        else:
            overall_score = 0.0
        
        return float(overall_score), snap_events
    
    def compute_smoothness_anomaly(self, features: MotionFeatures) -> float:
        """
        偵測軌跡平滑度的異常。
        
        人類瞄準有特徵性的微調。
        自瞄通常有不自然的模式。
        
        參數：
            features: 軌跡分析的 MotionFeatures
            
        回傳：
            異常分數（0-1，越高 = 越可疑）
        """
        jitter_min, jitter_max = self.HUMAN_BASELINE['jitter']
        smooth_min, smooth_max = self.HUMAN_BASELINE['smoothness']
        
        # 檢查抖動異常
        jitter_anomaly = 0.0
        if features.jitter < jitter_min * 0.5:
            # 太平滑（可疑）
            jitter_anomaly = 1.0 - (features.jitter / (jitter_min * 0.5))
        elif features.jitter > jitter_max * 2:
            # 過度抖動（可能人工）
            jitter_anomaly = min(1.0, (features.jitter - jitter_max) / jitter_max)
        
        # 檢查平滑度異常
        smoothness_anomaly = 0.0
        if features.smoothness > smooth_max:
            # 太平滑（可疑）
            smoothness_anomaly = min(1.0, (features.smoothness - smooth_max) / 0.2)
        elif features.smoothness < smooth_min * 0.5:
            # 太抖（可能人工）
            smoothness_anomaly = min(1.0, (smooth_min - features.smoothness) / smooth_min)
        
        # 合併分數
        combined = (jitter_anomaly * 0.6 + smoothness_anomaly * 0.4)
        
        return float(combined)
    
    def compute_direction_entropy(self, angular_velocity: np.ndarray) -> float:
        """
        計算方向變化的資訊熵。
        
        原理：
        - 資訊熵（Shannon Entropy）衡量隨機性/不確定性
        - 人類瞄準的方向變化是隨機的 → 高熵值
        - 自瞄外掛可能有固定模式 → 低熵值
        
        公式：H = -Σ p(x) * log2(p(x))
        
        參數：
            angular_velocity: 角速度序列
            
        回傳：
            熵值（0-1 正規化後，越高 = 越隨機 = 越像人類）
        """
        if len(angular_velocity) < 10:
            return 0.5  # 資料不足，回傳中性值
        
        # 將角速度離散化成 20 個區間
        # 這樣可以計算每個區間的機率分布
        bins = 20
        hist, _ = np.histogram(angular_velocity, bins=bins)
        
        # 轉換為機率分布
        total = hist.sum()
        if total == 0:
            return 0.5
        
        probs = hist / total
        
        # 排除零機率（log(0) 未定義）
        probs = probs[probs > 0]
        
        # 計算熵值：H = -Σ p * log2(p)
        entropy = -np.sum(probs * np.log2(probs))
        
        # 正規化到 0-1 範圍
        # 最大熵值 = log2(bins)，當所有區間機率相等時
        max_entropy = np.log2(bins)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return float(normalized_entropy)
    
    def compute_human_likelihood(self, features: MotionFeatures) -> float:
        """
        計算行為是人類的整體可能性。
        
        基於區分人類與機器人瞄準的多個因素。
        
        參數：
            features: 軌跡分析的 MotionFeatures
            
        回傳：
            人類可能性（0-1，越高 = 越可能是人類）
        """
        scores = []
        
        # 檢查抖動（人類有自然手抖）
        jitter_min, jitter_max = self.HUMAN_BASELINE['jitter']
        if jitter_min <= features.jitter <= jitter_max:
            scores.append(1.0)
        elif features.jitter < jitter_min:
            scores.append(features.jitter / jitter_min)
        else:
            scores.append(max(0, 1 - (features.jitter - jitter_max) / jitter_max))
        
        # 檢查平滑度
        smooth_min, smooth_max = self.HUMAN_BASELINE['smoothness']
        if smooth_min <= features.smoothness <= smooth_max:
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        # 檢查速度變異數（人類有可變速度）
        var_min, var_max = self.HUMAN_BASELINE['speed_variance']
        if var_min <= features.speed_variance <= var_max:
            scores.append(1.0)
        elif features.speed_variance < var_min:
            scores.append(0.3)  # 太恆定
        else:
            scores.append(0.7)
        
        # 檢查角速度（人類有極限）
        if len(features.angular_velocity) > 0:
            mean_av = np.mean(features.angular_velocity)
            av_min, av_max = self.HUMAN_BASELINE['angular_velocity_mean']
            if av_min <= mean_av <= av_max:
                scores.append(1.0)
            elif mean_av > av_max:
                scores.append(max(0, 1 - (mean_av - av_max) / av_max))
            else:
                scores.append(mean_av / av_min)
        
        return float(np.mean(scores))
    
    def analyze_segment(self, segment: TrajectorySegment) -> AnomalyScores:
        """
        對軌跡片段執行完整的異常分析。
        
        參數：
            segment: 包含計算特徵的 TrajectorySegment
            
        回傳：
            包含所有偵測指標的 AnomalyScores
        """
        if segment.features is None:
            raise ValueError("片段必須有計算過的特徵")
        
        features = segment.features
        
        # 計算個別分數
        stickiness = self.compute_stickiness_index(segment.positions)
        
        snap_score, snap_events = self.compute_snap_score(
            features.angular_velocity,
            features.speed,
            features.acceleration
        )
        
        smoothness_anomaly = self.compute_smoothness_anomaly(features)
        direction_entropy = self.compute_direction_entropy(features.angular_velocity)
        human_likelihood = self.compute_human_likelihood(features)
        
        # 熵值異常分數：低熵值表示有規律模式（可疑）
        # 人類熵值通常 > 0.6，自瞄可能 < 0.4
        entropy_anomaly = max(0, 0.6 - direction_entropy) / 0.6 if direction_entropy < 0.6 else 0
        
        # 計算整體可疑度（加權組合）
        if self.use_learned_weights and self.lr_weights is not None:
            # 使用邏輯迴歸學習到的權重（有科學根據）
            w = self.lr_weights
            overall_suspicion = (
                stickiness * w.get('stickiness', 0.25) +
                snap_score * w.get('snap_score', 0.40) +
                smoothness_anomaly * w.get('smoothness_anomaly', 0.20) +
                entropy_anomaly * w.get('entropy_anomaly', 0.15)
            ) * (1.0 + self.sensitivity * 0.5)
        else:
            # 使用預設經驗權重
            # 權重設定理由：
            # - snap_score (0.40): 瞬間轉向是自瞄最明確的特徵，人類反應時間限制無法產生
            # - stickiness (0.25): 完美鎖定需要持續精準追蹤，高度依賴機械輔助
            # - smoothness_anomaly (0.20): 機械平滑與人類自然抖動有可區分差異
            # - entropy_anomaly (0.15): 低熵值表示有規律的機械模式
            overall_suspicion = (
                stickiness * 0.25 +
                snap_score * 0.40 +
                smoothness_anomaly * 0.20 +
                entropy_anomaly * 0.15
            ) * (1.0 + self.sensitivity * 0.5)
        
        overall_suspicion = min(1.0, overall_suspicion)
        
        return AnomalyScores(
            stickiness_index=stickiness,
            snap_score=snap_score,
            smoothness_anomaly=smoothness_anomaly,
            direction_entropy=direction_entropy,
            human_likelihood=human_likelihood,
            overall_suspicion=overall_suspicion,
            anomaly_details={
                'jitter': features.jitter,
                'smoothness': features.smoothness,
                'mean_speed': features.mean_speed,
                'max_speed': features.max_speed,
                'speed_variance': features.speed_variance,
                'direction_changes': features.direction_changes,
                'snap_events_count': len(snap_events),
                'max_angular_velocity': float(np.max(features.angular_velocity)) if len(features.angular_velocity) > 0 else 0,
                'entropy_anomaly': entropy_anomaly
            }
        )
    
    def train_logistic_regression(self, feature_data: List[Dict[str, float]], 
                                    labels: List[int],
                                    test_size: float = 0.2) -> Dict:
        """
        使用邏輯迴歸從標註資料中學習最佳特徵權重。
        
        邏輯迴歸模型：
            P(aimbot|X) = σ(w₁·stickiness + w₂·snap + w₃·smoothness + w₄·entropy + b)
            其中 σ(z) = 1/(1+e^(-z)) 為 sigmoid 函數
        
        學習完成後，將迴歸係數正規化為加權評分公式的權重，
        使其總和為 1.0，可直接替換經驗權重。
        
        參數：
            feature_data: 特徵字典列表，每個字典包含：
                - stickiness: 黏著度指數
                - snap_score: Snap 分數
                - smoothness_anomaly: 平滑度異常
                - entropy_anomaly: 熵值異常
            labels: 標籤列表（0=人類, 1=自瞄）
            test_size: 測試集比例（預設 0.2）
            
        回傳：
            包含訓練結果的字典：
            - weights: 正規化後的權重
            - raw_coefficients: 原始迴歸係數
            - accuracy: 測試集準確率
            - classification_report: 分類報告
            - confusion_matrix: 混淆矩陣
            - model_params: 模型參數資訊
        """
        # 特徵名稱（對應加權公式的四個指標）
        feature_names = ['stickiness', 'snap_score', 'smoothness_anomaly', 'entropy_anomaly']
        
        # 組裝特徵矩陣 X 和標籤向量 y
        X = np.array([[d.get(f, 0.0) for f in feature_names] for d in feature_data])
        y = np.array(labels)
        
        # 資料量檢查
        if len(X) < 10:
            raise ValueError(f"資料量不足：至少需要 10 筆樣本，目前只有 {len(X)} 筆")
        
        # 類別平衡檢查
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            raise ValueError("標籤必須包含至少兩個類別（0=人類, 1=自瞄）")
        
        # 劃分訓練集與測試集（分層抽樣確保類別比例一致）
        if len(X) < 5 or min(counts) < 2:
            # 資料太少時不分割，使用全部資料訓練
            X_train, X_test, y_train, y_test = X, X, y, y
            split_warning = "資料量不足以分割測試集，使用全部資料進行訓練與評估"
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            split_warning = None
        
        # 特徵標準化（Z-score 正規化）
        # 確保各特徵在相同尺度上比較，避免數值範圍差異影響係數大小
        self.lr_scaler = StandardScaler()
        X_train_scaled = self.lr_scaler.fit_transform(X_train)
        X_test_scaled = self.lr_scaler.transform(X_test)
        
        # 訓練邏輯迴歸模型
        # - C=1.0: 正則化強度（L2 懲罰），防止過擬合
        # - solver='lbfgs': 適合小型資料集的最佳化演算法
        # - max_iter=1000: 最大迭代次數，確保收斂
        # - class_weight='balanced': 自動調整類別權重，處理不平衡資料
        self.lr_model = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        self.lr_model.fit(X_train_scaled, y_train)
        
        # 提取迴歸係數（標準化空間的值）
        standardized_coefficients = self.lr_model.coef_[0]
        intercept = self.lr_model.intercept_[0]
        
        # 將標準化空間的係數轉換回原始特徵空間
        # 標準化公式：x_scaled = (x - mean) / std
        # 因此原始空間係數：w_raw = w_scaled / std
        # 這確保權重反映各特徵在「原始尺度」上的實際影響力
        raw_coefficients = standardized_coefficients / self.lr_scaler.scale_
        
        # 將原始空間係數轉換為正規化權重
        # 取絕對值後正規化，使權重總和為 1.0
        # 絕對值是因為我們關心的是「影響力大小」而非方向
        abs_coefficients = np.abs(raw_coefficients)
        total = np.sum(abs_coefficients)
        
        if total > 0:
            normalized_weights = abs_coefficients / total
        else:
            normalized_weights = np.array([0.25, 0.25, 0.25, 0.25])
        
        # 建立權重字典
        self.lr_weights = {
            feature_names[i]: float(normalized_weights[i])
            for i in range(len(feature_names))
        }
        
        # 計算測試集指標
        y_pred = self.lr_model.predict(X_test_scaled)
        y_prob = self.lr_model.predict_proba(X_test_scaled)
        accuracy = float(np.mean(y_pred == y_test))
        
        # 建立分類報告
        report = classification_report(
            y_test, y_pred, 
            target_names=['Human', 'Aimbot'],
            output_dict=True
        )
        
        # 建立混淆矩陣
        cm = confusion_matrix(y_test, y_pred)
        
        # 組裝結果
        results = {
            'weights': self.lr_weights.copy(),
            'standardized_coefficients': {
                feature_names[i]: float(standardized_coefficients[i])
                for i in range(len(feature_names))
            },
            'raw_coefficients': {
                feature_names[i]: float(raw_coefficients[i])
                for i in range(len(feature_names))
            },
            'intercept': float(intercept),
            'normalized_weights': {
                feature_names[i]: float(normalized_weights[i])
                for i in range(len(feature_names))
            },
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'model_params': {
                'C': 1.0,
                'solver': 'lbfgs',
                'class_weight': 'balanced',
                'n_samples_train': len(X_train),
                'n_samples_test': len(X_test),
                'n_human': int(np.sum(y == 0)),
                'n_aimbot': int(np.sum(y == 1)),
            },
            'feature_names': feature_names,
            'split_warning': split_warning
        }
        
        return results
    
    def apply_learned_weights(self, enable: bool = True):
        """
        啟用或停用邏輯迴歸學習到的權重。
        
        參數：
            enable: True=使用學習權重, False=使用預設經驗權重
        """
        if enable and self.lr_weights is None:
            raise ValueError("尚未訓練邏輯迴歸模型，請先呼叫 train_logistic_regression()")
        self.use_learned_weights = enable
    
    def predict_with_lr(self, feature_dict: Dict[str, float]) -> Tuple[int, float]:
        """
        使用訓練好的邏輯迴歸模型進行預測。
        
        參數：
            feature_dict: 包含四個特徵的字典
            
        回傳：
            (預測類別, 自瞄機率) 的元組
        """
        if self.lr_model is None or self.lr_scaler is None:
            raise ValueError("模型尚未訓練")
        
        feature_names = ['stickiness', 'snap_score', 'smoothness_anomaly', 'entropy_anomaly']
        X = np.array([[feature_dict.get(f, 0.0) for f in feature_names]])
        X_scaled = self.lr_scaler.transform(X)
        
        prediction = self.lr_model.predict(X_scaled)[0]
        probability = self.lr_model.predict_proba(X_scaled)[0]
        
        return int(prediction), float(probability[1])  # probability[1] = P(aimbot)
    
    def save_lr_model(self, filepath: str = 'saved_models/logistic_regression.pkl'):
        """
        儲存邏輯迴歸模型與相關參數。
        
        參數：
            filepath: 儲存路徑
        """
        import pickle
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.lr_model,
            'scaler': self.lr_scaler,
            'weights': self.lr_weights,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_lr_model(self, filepath: str = 'saved_models/logistic_regression.pkl'):
        """
        載入已儲存的邏輯迴歸模型。
        
        參數：
            filepath: 模型檔案路徑
        """
        import pickle
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到模型檔案：{filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.lr_model = model_data['model']
        self.lr_scaler = model_data['scaler']
        self.lr_weights = model_data['weights']
    
    def classify_behavior(self, scores: AnomalyScores) -> Tuple[str, float]:
        """
        根據異常分數分類瞄準行為。
        
        分類：
        - 'legitimate'：正常人類行為
        - 'suspicious'：偵測到一些異常
        - 'likely_cheating'：強烈的自瞄指標
        
        參數：
            scores: 分析的 AnomalyScores
            
        回傳：
            (分類, 信心度) 的元組
        """
        suspicion = scores.overall_suspicion
        
        if suspicion < 0.3:
            return 'legitimate', 1.0 - suspicion
        elif suspicion < 0.6:
            return 'suspicious', 0.5 + (suspicion - 0.3) * 1.67
        else:
            return 'likely_cheating', min(0.99, suspicion)


# ==============================================================================
# 序列模型
# 用於時序模式學習的 CNN+LSTM 架構
# ==============================================================================

class SequenceModel:
    """
    用於時序瞄準分析的 CNN+LSTM 模型。
    
    架構：
    1. 1D CNN 層提取局部時序模式
    2. 雙向 LSTM 捕捉長程依賴
    3. 全連接層進行分類
    
    模型學習區分：
    - 人類瞄準模式（自然加速、微調）
    - 自瞄模式（瞬間 snap、完美追蹤）
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.model: Optional[Model] = None
        self.history: Optional[Dict] = None
        self.is_trained: bool = False
        
    def build_model(self) -> Model:
        """
        建立 CNN+LSTM 架構。
        
        輸入：(batch_size, sequence_length, feature_dim)
        輸出：(batch_size, 2) - [人類機率, 自瞄機率]
        """
        inputs = layers.Input(shape=(self.config.sequence_length, self.config.feature_dim))
        
        # 卷積層用於局部模式提取
        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.3)(x)
        
        # 雙向 LSTM 用於時序依賴
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)
        x = layers.Dropout(0.3)(x)
        
        # 全連接層用於分類
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # 輸出層
        outputs = layers.Dense(2, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='AimAnalyzer_CNN_LSTM')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """預測序列的自瞄機率。"""
        if self.model is None:
            raise ValueError("模型未建立。請先呼叫 build_model()")
        return self.model.predict(sequences, verbose=0)
    
    def predict_single(self, sequence: np.ndarray) -> Tuple[float, float]:
        """預測單一序列。"""
        if sequence.ndim == 2:
            sequence = np.expand_dims(sequence, axis=0)
        pred = self.predict(sequence)[0]
        return float(pred[0]), float(pred[1])
    
    def save_model(self, filepath: str) -> None:
        """儲存訓練好的模型到檔案。"""
        if self.model is None:
            raise ValueError("沒有模型可儲存。")
        self.model.save(filepath)
    
    def load_model(self, filepath: str) -> None:
        """從檔案載入模型。"""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True


# ==============================================================================
# 學術視覺化器
# 建立研究品質的視覺化
# ==============================================================================

class AcademicVisualizer:
    """
    建立適合出版的瞄準分析視覺化。
    
    視覺化類型：
    1. 帶異常高亮的軌跡圖
    2. 隨時間的速度/速率圖
    3. 帶 snap 偵測的角速度圖
    4. 統計分佈和比較
    5. 訓練歷史曲線
    """
    
    COLORS = {
        'human': '#4CAF50',      # 綠色
        'suspicious': '#FFC107', # 黃色
        'cheating': '#F44336',   # 紅色
        'primary': '#2196F3',    # 藍色
        'secondary': '#9C27B0',  # 紫色
        'background': '#1a1a2e',
        'text': '#FFFFFF',
        'grid': '#333355'
    }
    
    def __init__(self, dark_theme: bool = True):
        self.dark_theme = dark_theme
        self.template = 'plotly_dark' if dark_theme else 'plotly_white'
        
    def _get_layout(self, title: str, xaxis_title: str = '', 
                    yaxis_title: str = '') -> Dict:
        """取得標準佈局設定。"""
        return dict(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            template=self.template,
            paper_bgcolor=self.COLORS['background'] if self.dark_theme else 'white',
            plot_bgcolor=self.COLORS['background'] if self.dark_theme else 'white',
            font=dict(color=self.COLORS['text'] if self.dark_theme else 'black')
        )
    
    def plot_trajectory_2d(self, positions: np.ndarray, 
                           timestamps: Optional[np.ndarray] = None,
                           anomaly_scores: Optional[np.ndarray] = None,
                           title: str = "Crosshair Trajectory") -> go.Figure:
        """
        繪製帶可選異常高亮的 2D 準心軌跡。
        """
        fig = go.Figure()
        
        if anomaly_scores is not None:
            colors = anomaly_scores[:len(positions)]
            colorscale = [[0, self.COLORS['human']], 
                         [0.5, self.COLORS['suspicious']], 
                         [1, self.COLORS['cheating']]]
            
            fig.add_trace(go.Scatter(
                x=positions[:, 0],
                y=positions[:, 1],
                mode='markers+lines',
                marker=dict(
                    size=6,
                    color=colors,
                    colorscale=colorscale,
                    colorbar=dict(title='Suspicion'),
                    showscale=True
                ),
                line=dict(width=1, color='rgba(255,255,255,0.3)'),
                name='Trajectory',
                hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>Suspicion: %{marker.color:.2f}'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=positions[:, 0],
                y=positions[:, 1],
                mode='markers+lines',
                marker=dict(size=4, color=self.COLORS['primary']),
                line=dict(width=1, color=self.COLORS['primary']),
                name='Trajectory'
            ))
        
        # 標記起點和終點
        fig.add_trace(go.Scatter(
            x=[positions[0, 0]], y=[positions[0, 1]],
            mode='markers',
            marker=dict(size=15, color='lime', symbol='circle'),
            name='Start'
        ))
        fig.add_trace(go.Scatter(
            x=[positions[-1, 0]], y=[positions[-1, 1]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='x'),
            name='End'
        ))
        
        fig.update_layout(**self._get_layout(title, 'X Position (pixels)', 'Y Position (pixels)'))
        fig.update_yaxes(autorange='reversed')
        
        return fig
    
    def plot_velocity_profile(self, features: MotionFeatures,
                              timestamps: Optional[np.ndarray] = None,
                              title: str = "Velocity Profile") -> go.Figure:
        """繪製隨時間的速度和速度分量。"""
        fig = make_subplots(rows=3, cols=1, 
                           subplot_titles=['Speed', 'X Velocity', 'Y Velocity'],
                           vertical_spacing=0.1)
        
        x = timestamps[1:] if timestamps is not None else np.arange(len(features.speed))
        
        fig.add_trace(go.Scatter(
            x=x, y=features.speed,
            mode='lines', name='Speed',
            line=dict(color=self.COLORS['primary'])
        ), row=1, col=1)
        
        if len(features.velocity) > 0:
            fig.add_trace(go.Scatter(
                x=x[:len(features.velocity)], y=features.velocity[:, 0],
                mode='lines', name='Vx',
                line=dict(color=self.COLORS['human'])
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=x[:len(features.velocity)], y=features.velocity[:, 1],
                mode='lines', name='Vy',
                line=dict(color=self.COLORS['secondary'])
            ), row=3, col=1)
        
        fig.update_layout(**self._get_layout(title))
        fig.update_xaxes(title_text='Time (s)', row=3, col=1)
        
        return fig
    
    def plot_angular_velocity(self, angular_velocity: np.ndarray,
                              snap_threshold: float = 500,
                              title: str = "Angular Velocity Analysis") -> go.Figure:
        """繪製帶 snap 偵測閾值的角速度。"""
        fig = go.Figure()
        
        x = np.arange(len(angular_velocity))
        
        fig.add_trace(go.Scatter(
            x=x, y=angular_velocity,
            mode='lines', name='Angular Velocity',
            line=dict(color=self.COLORS['primary'])
        ))
        
        # 閾值線
        fig.add_hline(y=snap_threshold, 
                     line_dash='dash', 
                     line_color=self.COLORS['cheating'],
                     annotation_text=f'Snap Threshold ({snap_threshold}°/s)')
        
        # 標記 snap 事件
        snap_indices = np.where(angular_velocity > snap_threshold)[0]
        if len(snap_indices) > 0:
            fig.add_trace(go.Scatter(
                x=snap_indices,
                y=angular_velocity[snap_indices],
                mode='markers',
                marker=dict(size=12, color=self.COLORS['cheating'], symbol='triangle-up'),
                name='Potential Snaps'
            ))
        
        fig.update_layout(**self._get_layout(title, 'Frame', 'Angular Velocity (°/s)'))
        
        return fig
    
    def plot_speed_histogram(self, speed: np.ndarray,
                             human_baseline: Tuple[float, float] = (50, 300),
                             title: str = "Speed Distribution") -> go.Figure:
        """繪製帶人類基準覆蓋的速度分佈。"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=speed, nbinsx=50, name='Observed',
            marker_color=self.COLORS['primary'], opacity=0.7
        ))
        
        # 人類基準區域
        fig.add_vrect(
            x0=human_baseline[0], x1=human_baseline[1],
            fillcolor=self.COLORS['human'], opacity=0.2,
            line_width=0, annotation_text='Human Baseline',
            annotation_position='top'
        )
        
        fig.update_layout(**self._get_layout(title, 'Speed (pixels/s)', 'Frequency'))
        
        return fig
    
    def plot_feature_comparison(self, human_features: Dict[str, float],
                                sample_features: Dict[str, float],
                                title: str = "Feature Comparison") -> go.Figure:
        """雷達圖比較樣本特徵與人類基準。"""
        categories = list(human_features.keys())
        
        # 正規化值
        human_max = max(human_features.values()) or 1
        sample_max = max(sample_features.values()) or 1
        max_val = max(human_max, sample_max)
        
        human_normalized = [human_features.get(c, 0) / max_val for c in categories]
        sample_normalized = [sample_features.get(c, 0) / max_val for c in categories]
        
        # 閉合多邊形
        human_normalized.append(human_normalized[0])
        sample_normalized.append(sample_normalized[0])
        categories_closed = categories + [categories[0]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=human_normalized, theta=categories_closed,
            fill='toself', name='Human Baseline',
            line_color=self.COLORS['human'],
            fillcolor='rgba(76, 175, 80, 0.3)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=sample_normalized, theta=categories_closed,
            fill='toself', name='Sample',
            line_color=self.COLORS['primary'],
            fillcolor='rgba(33, 150, 243, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
                bgcolor=self.COLORS['background'] if self.dark_theme else 'white'
            ),
            **self._get_layout(title)
        )
        
        return fig
    
    def plot_training_history(self, history: Dict,
                              title: str = "Model Training History") -> go.Figure:
        """繪製訓練損失和準確率曲線。"""
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Loss', 'Accuracy'])
        
        epochs = range(1, len(history.get('loss', [])) + 1)
        
        fig.add_trace(go.Scatter(
            x=list(epochs), y=history.get('loss', []),
            mode='lines+markers', name='Training Loss',
            line=dict(color=self.COLORS['primary'])
        ), row=1, col=1)
        
        if 'val_loss' in history:
            fig.add_trace(go.Scatter(
                x=list(epochs), y=history['val_loss'],
                mode='lines+markers', name='Validation Loss',
                line=dict(color=self.COLORS['secondary'])
            ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=list(epochs), y=history.get('accuracy', []),
            mode='lines+markers', name='Training Accuracy',
            line=dict(color=self.COLORS['human'])
        ), row=1, col=2)
        
        if 'val_accuracy' in history:
            fig.add_trace(go.Scatter(
                x=list(epochs), y=history['val_accuracy'],
                mode='lines+markers', name='Validation Accuracy',
                line=dict(color=self.COLORS['cheating'])
            ), row=1, col=2)
        
        fig.update_layout(**self._get_layout(title))
        fig.update_xaxes(title_text='Epoch', row=1, col=1)
        fig.update_xaxes(title_text='Epoch', row=1, col=2)
        
        return fig
    
    def create_analysis_dashboard(self, positions: np.ndarray,
                                  features: MotionFeatures,
                                  scores: AnomalyScores,
                                  title: str = "Aim Analysis Dashboard") -> go.Figure:
        """建立綜合分析儀表板。"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Crosshair Trajectory', 'Speed Profile',
                'Angular Velocity', 'Anomaly Scores',
                'Speed Distribution', 'Feature Radar'
            ],
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'histogram'}, {'type': 'scatterpolar'}]
            ],
            vertical_spacing=0.1, horizontal_spacing=0.1
        )
        
        # 軌跡
        fig.add_trace(go.Scatter(
            x=positions[:, 0], y=positions[:, 1],
            mode='lines+markers', marker=dict(size=3),
            line=dict(width=1), name='Trajectory'
        ), row=1, col=1)
        
        # 速度
        fig.add_trace(go.Scatter(
            x=np.arange(len(features.speed)), y=features.speed,
            mode='lines', name='Speed'
        ), row=1, col=2)
        
        # 角速度
        if len(features.angular_velocity) > 0:
            fig.add_trace(go.Scatter(
                x=np.arange(len(features.angular_velocity)),
                y=features.angular_velocity,
                mode='lines', name='Angular Velocity'
            ), row=2, col=1)
        
        # 異常分數（注意：熵值顯示原始值，低熵=可疑）
        anomaly_names = ['Stickiness', 'Snap', 'Smoothness', 'Entropy']
        anomaly_values = [scores.stickiness_index, scores.snap_score, 
                         scores.smoothness_anomaly, scores.direction_entropy]
        colors = [self.COLORS['cheating'] if v > 0.5 else self.COLORS['human'] 
                 for v in anomaly_values]
        
        fig.add_trace(go.Bar(
            x=anomaly_names, y=anomaly_values,
            marker_color=colors, name='Anomaly Scores'
        ), row=2, col=2)
        
        # 速度直方圖
        fig.add_trace(go.Histogram(
            x=features.speed, nbinsx=30, name='Speed Dist'
        ), row=3, col=1)
        
        # 特徵雷達
        categories = ['Mean Speed', 'Jitter', 'Smoothness', 'Dir Changes']
        values = [
            features.mean_speed / 500,
            features.jitter / 5,
            features.smoothness,
            features.direction_changes / 20
        ]
        values = [min(1, max(0, v)) for v in values]
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself', name='Features'
        ), row=3, col=2)
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            template=self.template,
            showlegend=False,
            height=900
        )
        
        return fig


# ==============================================================================
# SESSION STATE 初始化
# ==============================================================================

def _generate_mock_training_data():
    import random as _rnd
    _rnd.seed(77)
    _m = {'sequences': [], 'labels': [], 'video_info': []}
    for i in range(30):
        s = _rnd.randint(3, 12)
        for _ in range(s):
            _m['sequences'].append([_rnd.uniform(80,200),_rnd.uniform(100,400),_rnd.uniform(500,20000),_rnd.uniform(1,5),_rnd.uniform(0.3,0.7),_rnd.randint(5,25),_rnd.uniform(50,150),_rnd.uniform(0.5,3.0),_rnd.uniform(1,4),_rnd.uniform(0.3,0.8)])
            _m['labels'].append(0)
        _m['video_info'].append({'filename':f'human_gameplay_{i+1:02d}.mp4','label':'human','segments':s,'timestamp':f'2026-02-23T{10+i//6}:{(i%6)*10:02d}:00'})
    for i in range(30):
        s = _rnd.randint(3, 12)
        for _ in range(s):
            _m['sequences'].append([_rnd.uniform(200,400),_rnd.uniform(300,800),_rnd.uniform(500,10000),_rnd.uniform(0.1,0.5),_rnd.uniform(0.85,0.98),_rnd.randint(1,6),_rnd.uniform(200,600),_rnd.uniform(0.1,0.5),_rnd.uniform(0.5,1.5),_rnd.uniform(0.85,0.99)])
            _m['labels'].append(1)
        _m['video_info'].append({'filename':f'aimbot_sample_{i+1:02d}.mp4','label':'aimbot','segments':s,'timestamp':f'2026-02-23T{14+i//6}:{(i%6)*10:02d}:00'})
    return _m

def init_session_state():
    """初始化 Streamlit session state 變數。"""
    defaults = {
        'crosshair_detector': CrosshairDetector(),
        'trajectory_analyzer': TrajectoryAnalyzer(fps=60.0),
        'anomaly_detector': AnomalyDetector(sensitivity=0.5),
        'sequence_model': SequenceModel(),
        'visualizer': AcademicVisualizer(dark_theme=True),
        'current_trajectory': None,
        'current_features': None,
        'current_anomaly_scores': None,
        'training_data': {'sequences': [], 'labels': [], 'video_info': []},
        'analysis_results': None,
        'model_trained': False,
        'training_history': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ==============================================================================
# 影片處理函數
# ==============================================================================

def extract_trajectory_from_video(video_path: str, frame_interval: int = 1,
                                   detection_method: str = 'phase',
                                   progress_callback=None) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    從影片檔案提取準心軌跡。
    
    參數：
        video_path: 影片檔案路徑
        frame_interval: 每 N 個影格處理一次
        detection_method: 'center' 或 'edge'
        progress_callback: 進度回呼函數 (0-1)
        
    回傳：
        (positions, timestamps, sample_frames) 的元組
    """
    if detection_method == 'phase':
        from screen_motion import extract_screen_motion
        positions, timestamps, samples, effective_fps = extract_screen_motion(
            video_path, frame_interval, progress_callback)
        st.session_state.trajectory_analyzer.fps = effective_fps
        st.session_state.trajectory_analyzer.dt = 1.0 / effective_fps
        return positions, timestamps, samples
    detector = st.session_state.crosshair_detector

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("無法開啟影片檔案（編碼不支援或檔案損毀）")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 防呆：壞掉的 metadata
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0  # 合理預設
    if total_frames <= 0:
        # 部分影片 metadata 報 0，改為「邊讀邊算」
        total_frames = 0

    # 更新分析器 FPS
    st.session_state.trajectory_analyzer.fps = fps / frame_interval
    st.session_state.trajectory_analyzer.dt = frame_interval / fps

    positions = []
    timestamps = []
    sample_frames = []
    frame_count = 0

    sample_mod = max(1, (total_frames // 10) + 1) if total_frames > 0 else 60

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            # 主偵測；若 edge 法意外失敗，自動退回 center
            try:
                position = detector.detect(frame, frame_count, timestamp, method=detection_method)
            except Exception:
                position = detector.detect_center_based(frame, frame_count, timestamp)

            positions.append([position.x, position.y])
            timestamps.append(timestamp)

            # 預覽用的樣本影格
            if len(sample_frames) < 10 and frame_count % sample_mod == 0:
                sample_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if progress_callback and total_frames > 0:
                progress_callback(min(1.0, frame_count / total_frames))

        frame_count += 1

    cap.release()

    return np.array(positions), np.array(timestamps), sample_frames


def analyze_trajectory(positions: np.ndarray, timestamps: np.ndarray) -> Dict:
    """
    執行完整的軌跡分析。
    
    回傳包含特徵、片段和異常分數的字典。
    """
    analyzer = st.session_state.trajectory_analyzer
    anomaly_detector = st.session_state.anomaly_detector

    # 防呆：軌跡為空或太短無法做時序分析
    if positions is None or len(positions) < 10 or len(timestamps) < 10:
        return {
            'status': 'insufficient',
            'reason': '影片中可用影格太少（<10），無法進行軌跡分析。請檢查影片長度 / 編碼。',
            'features': None, 'segments': [], 'segment_scores': [],
            'overall_scores': None, 'classification': 'unknown', 'confidence': 0.0,
            'positions': np.asarray(positions) if positions is not None else np.empty((0, 2)),
            'timestamps': np.asarray(timestamps) if timestamps is not None else np.empty((0,)),
        }

    # 計算整體特徵
    features = analyzer.analyze_trajectory(positions)
    
    # 將軌跡分割成片段進行時序分析
    segments = analyzer.segment_trajectory(positions, timestamps, window_size=60, overlap=30)
    
    # 分析每個片段
    segment_scores = []
    for segment in segments:
        if segment.features is not None:
            scores = anomaly_detector.analyze_segment(segment)
            segment_scores.append(scores)
    
    # 計算整體分數
    if segment_scores:
        overall_scores = AnomalyScores(
            stickiness_index=np.mean([s.stickiness_index for s in segment_scores]),
            snap_score=np.mean([s.snap_score for s in segment_scores]),
            smoothness_anomaly=np.mean([s.smoothness_anomaly for s in segment_scores]),
            direction_entropy=np.mean([s.direction_entropy for s in segment_scores]),
            human_likelihood=np.mean([s.human_likelihood for s in segment_scores]),
            overall_suspicion=np.mean([s.overall_suspicion for s in segment_scores]),
            anomaly_details={}
        )
    else:
        overall_scores = anomaly_detector.analyze_segment(
            TrajectorySegment(0, len(positions), timestamps[0], timestamps[-1], positions, features)
        )
    
    classification, confidence = anomaly_detector.classify_behavior(overall_scores)
    
    return {
        'features': features,
        'segments': segments,
        'segment_scores': segment_scores,
        'overall_scores': overall_scores,
        'classification': classification,
        'confidence': confidence,
        'positions': positions,
        'timestamps': timestamps
    }


# ==============================================================================
# UI 渲染函數
# ==============================================================================

def render_sidebar():
    """渲染側邊欄設定和控制。"""
    with st.sidebar:
        st.markdown("## Settings")

        # ---- 背景圖上傳（替代格子背景） ----
        st.markdown("### Background")
        bg_file = st.file_uploader(
            "上傳背景圖（取代格網）",
            type=['png', 'jpg', 'jpeg', 'webp'],
            key='bg_image_upload',
            help="上傳後會覆蓋預設的格網背景；清除後恢復原樣"
        )
        if bg_file is not None:
            import base64 as _b64
            raw = bg_file.getvalue()
            st.session_state['bg_image_b64'] = _b64.b64encode(raw).decode('ascii')
            st.session_state['bg_image_mime'] = bg_file.type or 'image/png'
        col_bg1, col_bg2 = st.columns(2)
        with col_bg1:
            st.slider("暗化程度", 0.0, 0.9,
                      st.session_state.get('bg_dim_level', 0.55), 0.05,
                      key='bg_dim_level',
                      help="數值越大越暗，讓內容更易閱讀")
        with col_bg2:
            if st.button("清除背景圖", use_container_width=True, key='bg_clear_btn'):
                st.session_state.pop('bg_image_b64', None)
                st.session_state.pop('bg_image_mime', None)
                st.rerun()

        st.markdown("### Video Processing")
        frame_interval = st.slider(
            "Frame Sampling Interval",
            min_value=1, max_value=10, value=1,
            help="Process every Nth frame. 1 = most accurate, higher = faster"
        )
        
        detection_method = st.selectbox(
            "Trajectory Extraction Method",
            options=['phase', 'center', 'edge'],
            format_func=lambda x: {
                'phase': 'Screen Motion (Phase Correlation)',
                'center': 'Screen Center (Legacy Comparison Only)',
                'edge': 'Edge Detection (Legacy Comparison Only)'
            }[x]
        )
        
        st.markdown("### Anomaly Detection")
        sensitivity = st.slider(
            "Detection Sensitivity",
            min_value=0.1, max_value=1.0, value=0.5,
            help="Higher = more false positives, Lower = more false negatives"
        )
        st.session_state.anomaly_detector.sensitivity = sensitivity
        
        st.markdown("### Model Status")
        if st.session_state.model_trained:
            st.success("Model Trained")
        else:
            st.info("Using Rule-Based Detection")
        
        return {
            'frame_interval': frame_interval,
            'detection_method': detection_method,
            'sensitivity': sensitivity
        }


def render_analyze_tab(settings: Dict):
    """渲染影片分析頁籤。"""
    st.markdown("## Video Analysis")
    st.markdown("Upload FPS gameplay footage to estimate accumulated screen-motion trajectories.")
    st.info("畫面位移估計不是真實滑鼠軌跡或視角角度。角色移動、HUD、後座力和切鏡可能干擾結果；現有可疑度僅供研究比較，不是作弊機率。舊中心法僅保留作修正前對照。")
    
    uploaded_file = st.file_uploader(
        "Upload Gameplay Video",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        key='analyze_upload'
    )
    
    if uploaded_file:
        # 偵測是否上傳了新影片，若是則清空舊的分析結果，避免匯出頁顯示 stale 資料
        _sig = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.get('_analyze_last_sig') != _sig:
            st.session_state.analysis_results = None
            st.session_state['_analyze_last_sig'] = _sig

        video_path = _save_uploaded_to_tempfile(uploaded_file, suffix='.mp4')

        try:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.video(uploaded_file)

            with col2:
                st.markdown("### Video Info")
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frames / fps if fps > 0 else 0
                cap.release()

                st.metric("FPS", f"{fps:.1f}")
                st.metric("Total Frames", frames)
                st.metric("Duration", f"{duration:.1f} sec")

                if frames <= 0 or fps <= 0:
                    st.warning("⚠ 影片 metadata 異常，可能無法分析；會嘗試以預設 30 FPS 讀取。")

            if st.button("Start Analysis", type="primary", use_container_width=True):
                try:
                    with st.spinner("Extracting trajectory..."):
                        progress_bar = st.progress(0)

                        def update_progress(p):
                            progress_bar.progress(p)

                        positions, timestamps, sample_frames = extract_trajectory_from_video(
                            video_path,
                            frame_interval=settings['frame_interval'],
                            detection_method=settings['detection_method'],
                            progress_callback=update_progress
                        )
                        progress_bar.progress(1.0)

                    with st.spinner("Analyzing trajectory features..."):
                        results = analyze_trajectory(positions, timestamps)
                        st.session_state.analysis_results = results

                    if results.get('status') == 'insufficient':
                        st.error(f"❌ 分析中止：{results.get('reason', '資料不足')}")
                    else:
                        st.success("Analysis Complete!")
                except ValueError as e:
                    st.error(f"❌ 無法分析：{e}")
                except Exception as e:
                    st.error(f"❌ 分析失敗（未預期錯誤）：{type(e).__name__}: {e}")
                    st.exception(e)
        finally:
            try:
                os.unlink(video_path)
            except Exception:
                pass
    
    if st.session_state.analysis_results:
        render_analysis_results(st.session_state.analysis_results)

    render_duration_sensitivity_section(settings)


def extract_trajectory_window(video_path: str, start_frame: int, end_frame: int,
                              frame_interval: int = 1,
                              detection_method: str = 'phase') -> Tuple[np.ndarray, np.ndarray]:
    """從影片的指定影格範圍提取準心軌跡。"""
    if detection_method == 'phase':
        from screen_motion import extract_screen_motion
        positions, timestamps, _, effective_fps = extract_screen_motion(
            video_path, frame_interval, start_frame=start_frame, end_frame=end_frame)
        st.session_state.trajectory_analyzer.fps = effective_fps
        st.session_state.trajectory_analyzer.dt = 1.0 / effective_fps
        return positions, timestamps
    detector = st.session_state.crosshair_detector

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("無法開啟影片檔案")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("影片 FPS 無效")
    st.session_state.trajectory_analyzer.fps = fps / frame_interval
    st.session_state.trajectory_analyzer.dt = frame_interval / fps

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    positions: List[List[float]] = []
    timestamps: List[float] = []
    frame_count = start_frame

    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_count - start_frame) % frame_interval == 0:
            ts = frame_count / fps if fps > 0 else 0.0
            position = detector.detect(frame, frame_count, ts, method=detection_method)
            positions.append([position.x, position.y])
            timestamps.append(ts)
        frame_count += 1

    cap.release()
    return np.array(positions), np.array(timestamps)


def run_duration_sensitivity_test(video_path: str, durations: List[int],
                                   trials_per_duration: int,
                                   settings: Dict,
                                   progress_callback=None) -> pd.DataFrame:
    """將影片切成指定長度的多個視窗，對每個視窗執行偵測流程。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("無法開啟影片檔案")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    video_duration = total_frames / fps if fps > 0 else 0.0
    rows: List[Dict] = []

    sorted_durations = sorted(durations)
    total_planned = 0
    plan: List[Tuple[int, List[Tuple[int, int]]]] = []
    for dur in sorted_durations:
        window_frames = int(dur * fps)
        if window_frames <= 0 or window_frames > total_frames:
            plan.append((dur, []))
            continue
        max_windows = max(1, total_frames // window_frames)
        n_windows = min(trials_per_duration, max_windows)
        windows = [(i * window_frames, (i + 1) * window_frames) for i in range(n_windows)]
        plan.append((dur, windows))
        total_planned += n_windows

    analyzer = st.session_state.trajectory_analyzer
    anomaly_detector = st.session_state.anomaly_detector

    completed = 0
    for dur, windows in plan:
        for trial_idx, (start_f, end_f) in enumerate(windows, start=1):
            try:
                positions, timestamps = extract_trajectory_window(
                    video_path, start_f, end_f,
                    frame_interval=settings['frame_interval'],
                    detection_method=settings['detection_method']
                )
                if len(positions) < 5:
                    confidence = float('nan')
                    classification = 'insufficient_data'
                else:
                    features = analyzer.analyze_trajectory(positions)
                    segment = TrajectorySegment(
                        start_frame=0,
                        end_frame=len(positions),
                        start_time=float(timestamps[0]),
                        end_time=float(timestamps[-1]),
                        positions=positions,
                        features=features,
                    )
                    scores = anomaly_detector.analyze_segment(segment)
                    classification, conf = anomaly_detector.classify_behavior(scores)
                    confidence = float(conf)
            except Exception as exc:
                confidence = float('nan')
                classification = f'error: {exc}'

            rows.append({
                'duration_sec': dur,
                'trial_id': trial_idx,
                'start_frame': start_f,
                'end_frame': end_f,
                'start_time_sec': round(start_f / fps, 3) if fps > 0 else 0.0,
                'confidence': confidence,
                'confidence_pct': confidence * 100 if not np.isnan(confidence) else float('nan'),
                'classification': classification,
            })

            completed += 1
            if progress_callback and total_planned > 0:
                progress_callback(completed / total_planned)

    df = pd.DataFrame(rows)
    df.attrs['fps'] = fps
    df.attrs['video_duration'] = video_duration
    return df


def plot_duration_sensitivity(df: pd.DataFrame) -> "go.Figure":
    """繪製不同切片長度下的信心分佈盒鬚圖。"""
    fig = go.Figure()

    durations_sorted_desc = sorted(df['duration_sec'].dropna().unique(), reverse=True)
    n = max(1, len(durations_sorted_desc))
    palette = []
    for i in range(n):
        t = i / max(1, n - 1) if n > 1 else 0.0
        r = int(70 + (220 - 70) * t)
        g = int(140 + (120 - 140) * t)
        b = int(220 + (100 - 220) * t)
        palette.append(f'rgba({r},{g},{b},0.55)')

    for color, dur in zip(palette, durations_sorted_desc):
        sub = df[df['duration_sec'] == dur].dropna(subset=['confidence_pct'])
        label = f"{int(dur)}s"
        fig.add_trace(go.Box(
            y=sub['confidence_pct'],
            name=label,
            marker_color=color,
            line=dict(color='#222'),
            boxpoints='all',
            jitter=0.5,
            pointpos=0,
            marker=dict(color='crimson', size=6, opacity=0.85,
                       line=dict(width=0.5, color='#660000')),
            fillcolor=color,
            showlegend=False,
        ))

    fig.add_hline(y=70, line_dash='dash', line_color='crimson',
                 annotation_text='Cheating threshold (70%)',
                 annotation_position='top left',
                 annotation_font_color='crimson')
    fig.add_hline(y=50, line_dash='dash', line_color='darkorange',
                 annotation_text='Suspicious threshold (50%)',
                 annotation_position='bottom left',
                 annotation_font_color='darkorange')

    n_trials = len(df.dropna(subset=['confidence_pct']))
    fig.update_layout(
        title=dict(
            text=f"<b>Confidence Score Distribution Across Clip Durations</b><br>"
                 f"<sup>1 video × {n_trials} trials total</sup>",
            x=0.5, xanchor='center'
        ),
        xaxis_title='Clip Duration',
        yaxis_title='Detection Confidence (%)',
        yaxis=dict(range=[0, 100], gridcolor='#eee'),
        plot_bgcolor='white',
        height=520,
        margin=dict(l=60, r=30, t=80, b=60),
    )
    return fig


def render_duration_sensitivity_section(settings: Dict):
    """渲染「切片長度敏感度測試」區段。"""
    st.markdown("---")
    st.markdown("## Duration Sensitivity Test")
    st.markdown(
        "Upload a single gameplay clip, choose which window lengths to evaluate, "
        "and watch how detection confidence varies as the analysed slice gets shorter. "
        "This reproduces the minimum-duration experiment interactively."
    )

    sens_file = st.file_uploader(
        "Upload Video for Sensitivity Test",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        key='duration_sens_upload'
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        durations_text = st.text_input(
            "Clip Durations to Test (seconds, 1–60, 以逗號分隔)",
            value="3, 5, 10, 15, 30",
            help="輸入想擷取的秒數（整數），用逗號分隔，例如：2, 4, 7, 12, 25",
            key='dur_sens_durations_text'
        )
        # 解析輸入並驗證 1-60
        durations = []
        invalid_tokens = []
        for tok in durations_text.replace('，', ',').split(','):
            tok = tok.strip()
            if not tok:
                continue
            try:
                v = int(float(tok))
                if 1 <= v <= 60:
                    durations.append(v)
                else:
                    invalid_tokens.append(tok)
            except ValueError:
                invalid_tokens.append(tok)
        durations = sorted(set(durations))
        if invalid_tokens:
            st.warning(f"已忽略超出範圍或無法解析的值：{', '.join(invalid_tokens)}（限 1–60 整數）")
        if durations:
            st.caption(f"將測試時長：{durations}（共 {len(durations)} 種）")
    with col_b:
        trials_per_duration = st.slider(
            "Trials per Duration",
            min_value=1, max_value=20, value=5,
            help="Maximum number of non-overlapping windows sampled for each duration",
            key='dur_sens_trials'
        )

    run_disabled = sens_file is None or len(durations) == 0
    if sens_file is None:
        st.caption("⬆ 請先上傳影片以啟用測試")
    elif len(durations) == 0:
        st.caption("⬆ 請在上方「Clip Durations to Test」輸入至少一個 1–60 的秒數（例：3, 5, 10, 15, 30）")
    if st.button("Run Duration Sensitivity Test",
                 type="primary", use_container_width=True,
                 disabled=run_disabled, key='dur_sens_run'):
        st.session_state.pop('duration_sensitivity_df', None)
        tmp_path = _save_uploaded_to_tempfile(sens_file, suffix='.mp4')

        try:
            progress_bar = st.progress(0.0)
            status = st.empty()
            status.info("Slicing video and running detector on each window...")

            def _update(p):
                progress_bar.progress(min(1.0, max(0.0, p)))

            df = run_duration_sensitivity_test(
                tmp_path, durations, trials_per_duration,
                settings, progress_callback=_update
            )
            progress_bar.progress(1.0)
            status.empty()

            if df.empty or df['confidence_pct'].dropna().empty:
                st.warning("No valid windows could be analysed. The video may be too short for the chosen durations.")
            else:
                st.session_state.duration_sensitivity_df = df
                st.success(f"Completed {len(df)} trials across {df['duration_sec'].nunique()} durations.")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    df = st.session_state.get('duration_sensitivity_df')
    if df is not None and not df.empty:
        valid = df.dropna(subset=['confidence_pct'])
        if not valid.empty:
            fig = plot_duration_sensitivity(df)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Summary by Duration")
            summary = (valid.groupby('duration_sec')['confidence_pct']
                       .agg(['count', 'mean', 'std', 'min', 'max'])
                       .reset_index()
                       .rename(columns={
                           'duration_sec': 'Duration (s)',
                           'count': 'Trials',
                           'mean': 'Mean (%)',
                           'std': 'Std (%)',
                           'min': 'Min (%)',
                           'max': 'Max (%)',
                       }))
            for c in ['Mean (%)', 'Std (%)', 'Min (%)', 'Max (%)']:
                summary[c] = summary[c].round(2)
            st.dataframe(summary, use_container_width=True, hide_index=True)

            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Trial-Level CSV",
                data=csv_bytes,
                file_name='duration_sensitivity_trials.csv',
                mime='text/csv',
                use_container_width=True,
            )


def render_analysis_results(results: Dict):
    """渲染分析結果和視覺化。"""
    st.markdown("---")
    st.markdown("## Analysis Results")

    # 防呆：軌跡資料不足或 overall_scores 為空時，顯示友善訊息並提早結束
    if results.get('status') == 'insufficient' or results.get('overall_scores') is None:
        st.warning(
            results.get('reason', '本次分析沒有產生有效的軌跡資料。')
            + "\n\n可能原因：影片太短、編碼不相容、或 Frame Sampling Interval 設得太大。"
        )
        return

    scores = results['overall_scores']
    classification = results['classification']
    confidence = results['confidence']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        color_class = {
            'legitimate': 'human-indicator',
            'suspicious': 'suspicious-indicator',
            'likely_cheating': 'cheating-indicator'
        }.get(classification, '')
        
        label = {
            'legitimate': 'Legitimate Player',
            'suspicious': 'Suspicious Behavior',
            'likely_cheating': 'Likely Cheating'
        }.get(classification, classification)
        
        st.markdown("### Classification")
        st.markdown(f"<span class='{color_class}' style='font-size: 1.5rem'>{label}</span>", 
                   unsafe_allow_html=True)
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col2:
        st.markdown("### Stickiness Index")
        st.metric("Score", f"{scores.stickiness_index:.2f}",
                 delta="Anomaly" if scores.stickiness_index > 0.8 else "Normal",
                 delta_color="inverse" if scores.stickiness_index > 0.8 else "off")
        st.caption("Target tracking precision")
    
    with col3:
        st.markdown("### Snap Score")
        st.metric("Score", f"{scores.snap_score:.2f}",
                 delta="Anomaly" if scores.snap_score > 0.5 else "Normal",
                 delta_color="inverse" if scores.snap_score > 0.5 else "off")
        st.caption("Sudden aim changes")
    
    with col4:
        st.markdown("### Smoothness Anomaly")
        st.metric("Score", f"{scores.smoothness_anomaly:.2f}",
                 delta="Anomaly" if scores.smoothness_anomaly > 0.6 else "Normal",
                 delta_color="inverse" if scores.smoothness_anomaly > 0.6 else "off")
        st.caption("Movement naturalness")
    
    st.markdown("---")
    st.markdown("## Visualizations")
    
    visualizer = st.session_state.visualizer
    features = results['features']
    positions = results['positions']
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Trajectory", "Velocity", "Angular Velocity", "Statistics", "Dashboard"
    ])
    
    with tab1:
        per_frame_suspicion = np.zeros(len(positions))
        for i, segment in enumerate(results['segments']):
            if i < len(results['segment_scores']):
                start, end = segment.start_frame, segment.end_frame
                scaled_start = start // max(1, len(positions) // len(per_frame_suspicion))
                scaled_end = end // max(1, len(positions) // len(per_frame_suspicion))
                per_frame_suspicion[scaled_start:scaled_end] = results['segment_scores'][i].overall_suspicion
        
        fig = visualizer.plot_trajectory_2d(positions, results['timestamps'], per_frame_suspicion)
        fig.update_layout(title="Extracted Trajectory — Screen Motion / Legacy Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = visualizer.plot_velocity_profile(features, results['timestamps'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if len(features.angular_velocity) > 0:
            fig = visualizer.plot_angular_velocity(features.angular_velocity)
            st.plotly_chart(fig, use_container_width=True)
            
            snap_indices = np.where(features.angular_velocity > 500)[0]
            if len(snap_indices) > 0:
                st.warning(f"Detected {len(snap_indices)} potential snap events")
        else:
            st.info("Insufficient trajectory data to compute angular velocity")
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            fig = visualizer.plot_speed_histogram(features.speed)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            human_baseline = {
                'Mean Speed': 150,
                'Jitter': 2.5,
                'Smoothness': 0.55,
                'Dir Changes': 10
            }
            sample_features = {
                'Mean Speed': features.mean_speed,
                'Jitter': features.jitter,
                'Smoothness': features.smoothness,
                'Dir Changes': features.direction_changes
            }
            fig = visualizer.plot_feature_comparison(human_baseline, sample_features)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        fig = visualizer.create_analysis_dashboard(positions, features, scores)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("## Detailed Data")
    
    with st.expander("Motion Features"):
        feature_data = {
            'Metric': ['Mean Speed', 'Max Speed', 'Speed Variance', 'Jitter', 'Smoothness', 'Direction Changes'],
            'Value': [
                f"{features.mean_speed:.2f} px/s",
                f"{features.max_speed:.2f} px/s",
                f"{features.speed_variance:.2f}",
                f"{features.jitter:.3f}",
                f"{features.smoothness:.3f}",
                f"{features.direction_changes}"
            ],
            'Human Baseline': [
                "50-300 px/s",
                "200-1500 px/s",
                "1000-50000",
                "0.5-5.0",
                "0.3-0.8",
                "5-30"
            ]
        }
        st.table(pd.DataFrame(feature_data))


def render_train_tab(settings: Dict):
    """渲染模型訓練頁籤。"""
    st.markdown("## Train Model")
    st.markdown("""
    Upload labeled gameplay videos to train the CNN+LSTM temporal model.
    The model learns to distinguish human players from aimbot users.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Add Training Data")
        
        label = st.radio(
            "Video Label",
            options=['human', 'aimbot'],
            format_func=lambda x: 'Human Player' if x == 'human' else 'Aimbot User',
            horizontal=True
        )
        
        train_file = st.file_uploader(
            "Upload Training Video",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            key='train_upload'
        )
        
        if train_file and st.button("Add to Training Set"):
            video_path = _save_uploaded_to_tempfile(train_file, suffix='.mp4')
            
            with st.spinner("Processing video..."):
                positions, timestamps, _ = extract_trajectory_from_video(
                    video_path,
                    frame_interval=settings['frame_interval'],
                    detection_method=settings['detection_method']
                )
                
                analyzer = st.session_state.trajectory_analyzer
                segments = analyzer.segment_trajectory(positions, timestamps, window_size=60, overlap=30)
                
                for segment in segments:
                    if segment.features is not None:
                        feature_vector = analyzer.get_feature_vector(segment.features)
                        st.session_state.training_data['sequences'].append(feature_vector)
                        st.session_state.training_data['labels'].append(0 if label == 'human' else 1)
                
                st.session_state.training_data['video_info'].append({
                    'filename': train_file.name,
                    'label': label,
                    'segments': len(segments),
                    'timestamp': datetime.now().isoformat()
                })
            
            os.unlink(video_path)
            st.success(f"Added {len(segments)} training samples")
            st.rerun()
    
    with col2:
        st.markdown("### Training Set Status")
        
        labels = st.session_state.training_data['labels']
        human_count = labels.count(0)
        aimbot_count = labels.count(1)
        
        st.metric("Human Samples", human_count)
        st.metric("Aimbot Samples", aimbot_count)
        st.metric("Total Samples", len(labels))
        
        if st.session_state.training_data['video_info']:
            with st.expander("Uploaded Videos"):
                for info in st.session_state.training_data['video_info']:
                    icon = '👤' if info['label'] == 'human' else '🤖'
                    st.text(f"{icon} {info['filename']} ({info['segments']} samples)")
    
    st.markdown("---")
    st.markdown("### Start Training")
    
    min_samples = 20
    can_train = len(labels) >= min_samples and human_count > 0 and aimbot_count > 0
    
    if not can_train:
        st.warning(f"Need at least {min_samples} samples (both classes) to start training")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.slider("Epochs", 10, 100, 30)
    with col2:
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
    with col3:
        learning_rate = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005], value=0.001)
    
    if st.button("Start Training", disabled=not can_train, type="primary", use_container_width=True):
        sequences = np.array(st.session_state.training_data['sequences'])
        labels_arr = np.array(st.session_state.training_data['labels'])
        
        config = TrainingConfig(
            sequence_length=1,
            feature_dim=sequences.shape[1],
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        model = SequenceModel(config)
        model.build_model()
        
        from tensorflow.keras.utils import to_categorical
        y_onehot = to_categorical(labels_arr, num_classes=2)
        
        sequences_3d = sequences.reshape(sequences.shape[0], 1, sequences.shape[1])
        
        X_train, X_val, y_train, y_val = train_test_split(
            sequences_3d, y_onehot, test_size=0.2, stratify=labels_arr, random_state=42
        )
        
        with st.spinner("Training..."):
            progress_text = st.empty()

            class StreamlitCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress_text.text(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Loss: {logs['loss']:.4f} - "
                        f"Acc: {logs['accuracy']:.4f} - "
                        f"Val Loss: {logs.get('val_loss', 0):.4f} - "
                        f"Val Acc: {logs.get('val_accuracy', 0):.4f}"
                    )

            try:
                history = model.model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=[StreamlitCallback()],
                    verbose=0
                )
            except tf.errors.ResourceExhaustedError as e:
                st.error("❌ 訓練失敗：GPU/記憶體不足 (OOM)。"
                         "請縮小 Batch Size、減少訓練樣本，或改用較小的序列長度。")
                st.exception(e)
                return
            except (ValueError, tf.errors.InvalidArgumentError) as e:
                st.error(f"❌ 訓練失敗：資料形狀或參數錯誤。請檢查輸入樣本是否足夠、標籤是否平衡。\n\n{e}")
                return
            except Exception as e:
                st.error(f"❌ 訓練失敗（未預期錯誤）：{type(e).__name__}: {e}")
                st.exception(e)
                return

        st.session_state.sequence_model = model
        st.session_state.model_trained = True
        st.session_state.training_history = history.history

        st.success("Training Complete!")

        visualizer = st.session_state.visualizer
        fig = visualizer.plot_training_history(history.history)
        st.plotly_chart(fig, use_container_width=True)

        final_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Accuracy", f"{final_acc:.1%}")
        with col2:
            st.metric("Validation Accuracy", f"{final_val_acc:.1%}")
    
    if st.session_state.training_history:
        st.markdown("### Training History")
        visualizer = st.session_state.visualizer
        fig = visualizer.plot_training_history(st.session_state.training_history)
        st.plotly_chart(fig, use_container_width=True)


def render_lr_tab(settings):
    """渲染邏輯迴歸權重學習頁籤。"""
    st.markdown("## Logistic Regression Weight Learning")
    st.markdown("""
    Train a Logistic Regression model on labeled data to **automatically learn optimal feature weights** 
    with statistical justification. The learned coefficients replace the manually configured weights 
    in the scoring formula, providing scientifically grounded detection.
    """)
    
    st.markdown(r"""
    **Model**: $P(\text{aimbot}|\mathbf{X}) = \sigma(w_1 \cdot \text{stickiness} + w_2 \cdot \text{snap} + w_3 \cdot \text{smoothness} + w_4 \cdot \text{entropy} + b)$
    
    Where $\sigma(z) = \frac{1}{1+e^{-z}}$ is the sigmoid function. After training, 
    the absolute values of $w_1 \sim w_4$ are normalized to sum to 1.0 as the new scoring weights.
    """)
    
    lr_tab_data, lr_tab_train, lr_tab_results = st.tabs([
        "Prepare Data", "Train Model", "Results & Weights"
    ])
    
    # 初始化邏輯迴歸的 session state
    if 'lr_feature_data' not in st.session_state:
        st.session_state.lr_feature_data = []
        st.session_state.lr_labels = []
        st.session_state.lr_results = None
        st.session_state.lr_sample_info = []
    
    with lr_tab_data:
        st.markdown("### Prepare Labeled Feature Data")
        st.markdown("""
        Add labeled feature samples for logistic regression training.
        You can add data from **analyzed videos** or enter **manual feature values**.
        """)
        
        data_method = st.radio(
            "Data Input Method",
            ['From Analyzed Video', 'Manual Input', 'Generate Synthetic Data'],
            horizontal=True,
            key='lr_data_method'
        )
        
        if data_method == 'From Analyzed Video':
            st.markdown("""
            Upload a video, analyze it, and assign a label (Human/Aimbot). 
            The system will extract feature scores from each trajectory segment.
            """)
            
            lr_label = st.radio(
                "Label for this video",
                ['human', 'aimbot'],
                format_func=lambda x: 'Human Player' if x == 'human' else 'Aimbot User',
                horizontal=True,
                key='lr_video_label'
            )
            
            lr_video = st.file_uploader(
                "Upload Video for LR Training",
                type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
                key='lr_video_upload'
            )
            
            if lr_video and st.button("Extract Features & Add", key='lr_extract_btn'):
                video_path = _save_uploaded_to_tempfile(lr_video, suffix='.mp4')
                
                with st.spinner("Extracting trajectory features..."):
                    positions, timestamps, _ = extract_trajectory_from_video(
                        video_path,
                        frame_interval=settings['frame_interval'],
                        detection_method=settings['detection_method']
                    )
                    
                    analyzer = st.session_state.trajectory_analyzer
                    detector = st.session_state.anomaly_detector
                    segments = analyzer.segment_trajectory(positions, timestamps, window_size=60, overlap=30)
                    
                    added = 0
                    for segment in segments:
                        if segment.features is not None:
                            scores = detector.analyze_segment(segment)
                            
                            entropy_anomaly = max(0, 0.6 - scores.direction_entropy) / 0.6 if scores.direction_entropy < 0.6 else 0
                            
                            feature_dict = {
                                'stickiness': scores.stickiness_index,
                                'snap_score': scores.snap_score,
                                'smoothness_anomaly': scores.smoothness_anomaly,
                                'entropy_anomaly': entropy_anomaly
                            }
                            
                            st.session_state.lr_feature_data.append(feature_dict)
                            st.session_state.lr_labels.append(0 if lr_label == 'human' else 1)
                            added += 1
                    
                    st.session_state.lr_sample_info.append({
                        'filename': lr_video.name,
                        'label': lr_label,
                        'samples': added
                    })
                
                os.unlink(video_path)
                st.success(f"Extracted {added} feature samples from video")
                st.rerun()
        
        elif data_method == 'Manual Input':
            st.markdown("Enter feature values manually (useful for testing or adding known reference data).")
            
            ml_col1, ml_col2 = st.columns(2)
            with ml_col1:
                m_stickiness = st.slider("Stickiness Index", 0.0, 1.0, 0.5, 0.01, key='m_stick')
                m_snap = st.slider("Snap Score", 0.0, 1.0, 0.3, 0.01, key='m_snap')
            with ml_col2:
                m_smooth = st.slider("Smoothness Anomaly", 0.0, 1.0, 0.3, 0.01, key='m_smooth')
                m_entropy = st.slider("Entropy Anomaly", 0.0, 1.0, 0.2, 0.01, key='m_entropy')
            
            m_label = st.radio("Label", ['Human (0)', 'Aimbot (1)'], horizontal=True, key='m_label')
            
            if st.button("Add Manual Sample", key='lr_manual_add'):
                feature_dict = {
                    'stickiness': m_stickiness,
                    'snap_score': m_snap,
                    'smoothness_anomaly': m_smooth,
                    'entropy_anomaly': m_entropy
                }
                label_val = 0 if 'Human' in m_label else 1
                st.session_state.lr_feature_data.append(feature_dict)
                st.session_state.lr_labels.append(label_val)
                st.session_state.lr_sample_info.append({
                    'filename': 'manual_input',
                    'label': 'human' if label_val == 0 else 'aimbot',
                    'samples': 1
                })
                st.success("Added 1 manual sample")
                st.rerun()
        
        else:  # Generate Synthetic Data
            st.markdown("""
            Generate synthetic training data based on known human and aimbot behavioral distributions.
            This is useful for **demonstration and testing** when real labeled data is not available.
            
            **Human distribution**: Based on neuromuscular control characteristics  
            **Aimbot distribution**: Based on algorithmic aim assistance patterns
            """)
            
            syn_col1, syn_col2 = st.columns(2)
            with syn_col1:
                n_human = st.number_input("Number of Human Samples", 10, 500, 50, key='syn_human')
            with syn_col2:
                n_aimbot = st.number_input("Number of Aimbot Samples", 10, 500, 50, key='syn_aimbot')
            
            noise_level = st.slider("Noise Level", 0.0, 1.0, 0.3, 0.05, 
                                     help="Higher = more realistic variation", key='syn_noise')
            
            if st.button("Generate Synthetic Data", key='lr_syn_btn', type='primary'):
                np.random.seed(42)
                
                for _ in range(n_human):
                    feature_dict = {
                        'stickiness': np.clip(np.random.normal(0.45, 0.15 * (1 + noise_level)), 0, 1),
                        'snap_score': np.clip(np.random.exponential(0.1) * (1 + noise_level * 0.5), 0, 0.8),
                        'smoothness_anomaly': np.clip(np.random.normal(0.3, 0.15 * (1 + noise_level)), 0, 1),
                        'entropy_anomaly': np.clip(np.random.exponential(0.08) * (1 + noise_level * 0.3), 0, 0.5)
                    }
                    st.session_state.lr_feature_data.append(feature_dict)
                    st.session_state.lr_labels.append(0)
                
                for _ in range(n_aimbot):
                    feature_dict = {
                        'stickiness': np.clip(np.random.normal(0.85, 0.1 * (1 + noise_level)), 0.3, 1),
                        'snap_score': np.clip(np.random.normal(0.7, 0.15 * (1 + noise_level)), 0.2, 1),
                        'smoothness_anomaly': np.clip(np.random.normal(0.75, 0.12 * (1 + noise_level)), 0.2, 1),
                        'entropy_anomaly': np.clip(np.random.normal(0.6, 0.15 * (1 + noise_level)), 0.1, 1)
                    }
                    st.session_state.lr_feature_data.append(feature_dict)
                    st.session_state.lr_labels.append(1)
                
                st.session_state.lr_sample_info.append({
                    'filename': f'synthetic_{n_human}h_{n_aimbot}a',
                    'label': 'mixed',
                    'samples': n_human + n_aimbot
                })
                
                st.success(f"Generated {n_human} human + {n_aimbot} aimbot synthetic samples")
                st.rerun()
        
        # 顯示目前資料集狀態
        st.markdown("### Current Dataset")
        lr_labels = st.session_state.lr_labels
        lr_human = lr_labels.count(0)
        lr_aimbot = lr_labels.count(1)
        
        ds_col1, ds_col2, ds_col3 = st.columns(3)
        with ds_col1:
            st.metric("Human Samples", lr_human)
        with ds_col2:
            st.metric("Aimbot Samples", lr_aimbot)
        with ds_col3:
            st.metric("Total", len(lr_labels))
        
        if st.session_state.lr_feature_data:
            with st.expander("Dataset Preview"):
                df = pd.DataFrame(st.session_state.lr_feature_data)
                df['label'] = ['Human' if l == 0 else 'Aimbot' for l in st.session_state.lr_labels]
                st.dataframe(df, use_container_width=True)
            
            with st.expander("Feature Distribution"):
                df_plot = pd.DataFrame(st.session_state.lr_feature_data)
                df_plot['label'] = ['Human' if l == 0 else 'Aimbot' for l in st.session_state.lr_labels]
                
                fig = make_subplots(rows=2, cols=2, subplot_titles=[
                    'Stickiness', 'Snap Score', 'Smoothness Anomaly', 'Entropy Anomaly'
                ])
                
                for i, feat in enumerate(['stickiness', 'snap_score', 'smoothness_anomaly', 'entropy_anomaly']):
                    row, col = (i // 2) + 1, (i % 2) + 1
                    
                    human_vals = df_plot[df_plot['label'] == 'Human'][feat]
                    aimbot_vals = df_plot[df_plot['label'] == 'Aimbot'][feat]
                    
                    fig.add_trace(go.Histogram(x=human_vals, name='Human', marker_color='#00d4aa', 
                                               opacity=0.7, showlegend=(i==0)), row=row, col=col)
                    fig.add_trace(go.Histogram(x=aimbot_vals, name='Aimbot', marker_color='#ff4444', 
                                               opacity=0.7, showlegend=(i==0)), row=row, col=col)
                
                fig.update_layout(
                    height=500, barmode='overlay',
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    title_text='Feature Distribution by Class'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Clear All LR Data", key='lr_clear'):
            st.session_state.lr_feature_data = []
            st.session_state.lr_labels = []
            st.session_state.lr_results = None
            st.session_state.lr_sample_info = []
            st.rerun()
    
    with lr_tab_train:
        st.markdown("### Train Logistic Regression")
        
        lr_labels = st.session_state.lr_labels
        lr_human = lr_labels.count(0)
        lr_aimbot = lr_labels.count(1)
        can_train_lr = len(lr_labels) >= 10 and lr_human > 0 and lr_aimbot > 0
        
        if not can_train_lr:
            st.warning("Need at least 10 samples with both classes (Human & Aimbot) to train. "
                       "Go to 'Prepare Data' tab to add samples.")
        
        test_ratio = st.slider("Test Set Ratio", 0.1, 0.4, 0.2, 0.05, key='lr_test_ratio',
                                help="Proportion of data reserved for evaluation")
        
        if st.button("Train Logistic Regression", disabled=not can_train_lr, 
                      type="primary", use_container_width=True, key='lr_train_btn'):
            detector = st.session_state.anomaly_detector
            
            with st.spinner("Training Logistic Regression model..."):
                try:
                    results = detector.train_logistic_regression(
                        st.session_state.lr_feature_data,
                        st.session_state.lr_labels,
                        test_size=test_ratio
                    )
                    st.session_state.lr_results = results
                    
                    detector.save_lr_model()
                    
                    st.success(f"Training complete! Accuracy: {results['accuracy']:.1%}")
                    st.info("Go to 'Results & Weights' tab to see detailed results and apply learned weights.")
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
    
    with lr_tab_results:
        st.markdown("### Logistic Regression Results")
        
        if st.session_state.lr_results is None:
            st.info("No training results yet. Train the model first.")
        else:
            results = st.session_state.lr_results
            
            if results.get('split_warning'):
                st.warning(results['split_warning'])
            
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Test Accuracy", f"{results['accuracy']:.1%}")
            with res_col2:
                st.metric("Training Samples", results['model_params']['n_samples_train'])
            with res_col3:
                st.metric("Test Samples", results['model_params']['n_samples_test'])
            
            st.markdown("---")
            
            st.markdown("### Learned Weights vs Default Weights")
            
            default_weights = {
                'stickiness': 0.25,
                'snap_score': 0.40,
                'smoothness_anomaly': 0.20,
                'entropy_anomaly': 0.15
            }
            
            learned_weights = results['weights']
            feature_display_names = {
                'stickiness': 'Stickiness',
                'snap_score': 'Snap Score', 
                'smoothness_anomaly': 'Smoothness',
                'entropy_anomaly': 'Entropy'
            }
            
            weight_df = pd.DataFrame({
                'Feature': [feature_display_names[f] for f in default_weights.keys()],
                'Default Weight': [f"{v:.3f}" for v in default_weights.values()],
                'Learned Weight': [f"{learned_weights[f]:.3f}" for f in default_weights.keys()],
                'Raw Coefficient': [f"{results['raw_coefficients'][f]:.4f}" for f in default_weights.keys()],
                'Change': [f"{(learned_weights[f] - default_weights[f]) / default_weights[f]:+.1%}" 
                          for f in default_weights.keys()]
            })
            st.dataframe(weight_df, use_container_width=True, hide_index=True)
            
            features = list(default_weights.keys())
            display_names = [feature_display_names[f] for f in features]
            
            fig_weights = go.Figure()
            fig_weights.add_trace(go.Bar(
                name='Default (Empirical)',
                x=display_names,
                y=[default_weights[f] for f in features],
                marker_color='#666666',
                text=[f"{default_weights[f]:.3f}" for f in features],
                textposition='auto'
            ))
            fig_weights.add_trace(go.Bar(
                name='Learned (Logistic Regression)',
                x=display_names,
                y=[learned_weights[f] for f in features],
                marker_color='#00d4aa',
                text=[f"{learned_weights[f]:.3f}" for f in features],
                textposition='auto'
            ))
            
            fig_weights.update_layout(
                barmode='group',
                title='Feature Weight Comparison: Default vs Learned',
                yaxis_title='Weight',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig_weights, use_container_width=True)
            
            st.markdown("### Raw Logistic Regression Coefficients")
            st.markdown(f"**Intercept (b)**: {results['intercept']:.4f}")
            
            raw_coefs = results['raw_coefficients']
            fig_coef = go.Figure(go.Bar(
                x=[feature_display_names[f] for f in raw_coefs.keys()],
                y=list(raw_coefs.values()),
                marker_color=['#00d4aa' if v > 0 else '#ff4444' for v in raw_coefs.values()],
                text=[f"{v:.4f}" for v in raw_coefs.values()],
                textposition='auto'
            ))
            fig_coef.update_layout(
                title='Logistic Regression Coefficients (before normalization)',
                yaxis_title='Coefficient Value',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350
            )
            st.plotly_chart(fig_coef, use_container_width=True)
            
            st.markdown("""
            > **Interpretation**: Positive coefficients increase aimbot probability. 
            > The absolute magnitude indicates the feature's contribution to the decision boundary.
            > Coefficients are computed on standardized features (Z-score normalized).
            """)
            
            st.markdown("### Confusion Matrix")
            cm = np.array(results['confusion_matrix'])
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Human', 'Predicted Aimbot'],
                y=['Actual Human', 'Actual Aimbot'],
                text=cm,
                texttemplate='%{text}',
                colorscale='Teal',
                showscale=False
            ))
            fig_cm.update_layout(
                title='Confusion Matrix',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            with st.expander("Classification Report"):
                report = results['classification_report']
                report_df = pd.DataFrame({
                    'Class': ['Human', 'Aimbot', 'Macro Avg', 'Weighted Avg'],
                    'Precision': [
                        f"{report['Human']['precision']:.3f}",
                        f"{report['Aimbot']['precision']:.3f}",
                        f"{report['macro avg']['precision']:.3f}",
                        f"{report['weighted avg']['precision']:.3f}"
                    ],
                    'Recall': [
                        f"{report['Human']['recall']:.3f}",
                        f"{report['Aimbot']['recall']:.3f}",
                        f"{report['macro avg']['recall']:.3f}",
                        f"{report['weighted avg']['recall']:.3f}"
                    ],
                    'F1-Score': [
                        f"{report['Human']['f1-score']:.3f}",
                        f"{report['Aimbot']['f1-score']:.3f}",
                        f"{report['macro avg']['f1-score']:.3f}",
                        f"{report['weighted avg']['f1-score']:.3f}"
                    ]
                })
                st.dataframe(report_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("### Apply Learned Weights")
            
            detector = st.session_state.anomaly_detector
            current_mode = "Learned (LR)" if detector.use_learned_weights else "Default (Empirical)"
            st.info(f"Currently using: **{current_mode}** weights")
            
            apply_col1, apply_col2 = st.columns(2)
            with apply_col1:
                if st.button("Use Learned Weights", type="primary", key='lr_apply', 
                             use_container_width=True):
                    detector.apply_learned_weights(enable=True)
                    st.success("Now using logistic regression learned weights for scoring!")
                    st.rerun()
            with apply_col2:
                if st.button("Use Default Weights", key='lr_default', use_container_width=True):
                    detector.apply_learned_weights(enable=False)
                    st.success("Reverted to default empirical weights.")
                    st.rerun()
            
            if detector.use_learned_weights and detector.lr_weights:
                w = detector.lr_weights
                st.markdown("**Current Scoring Formula:**")
                st.code(
                    f"overall_suspicion = (\n"
                    f"    stickiness × {w.get('stickiness', 0):.3f} +\n"
                    f"    snap_score × {w.get('snap_score', 0):.3f} +\n"
                    f"    smoothness × {w.get('smoothness_anomaly', 0):.3f} +\n"
                    f"    entropy   × {w.get('entropy_anomaly', 0):.3f}\n"
                    f") × sensitivity_modifier",
                    language='python'
                )
            else:
                st.markdown("**Current Scoring Formula (Default):**")
                st.code(
                    "overall_suspicion = (\n"
                    "    stickiness × 0.250 +\n"
                    "    snap_score × 0.400 +\n"
                    "    smoothness × 0.200 +\n"
                    "    entropy   × 0.150\n"
                    ") × sensitivity_modifier",
                    language='python'
                )


def render_export_tab():
    """渲染資料匯出頁籤。"""
    st.markdown("## Export Data")
    
    if not st.session_state.analysis_results:
        st.info("Please complete video analysis in the 'Analyze Video' tab first")
        return
    
    results = st.session_state.analysis_results
    
    st.markdown("### Available Export Formats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### CSV")
        st.caption("For Excel and statistics software")
        
        df = pd.DataFrame({
            'timestamp': results['timestamps'],
            'x': results['positions'][:, 0],
            'y': results['positions'][:, 1]
        })
        
        if len(results['features'].speed) == len(df) - 1:
            df['speed'] = np.append(results['features'].speed, np.nan)
        
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "trajectory_data.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### JSON")
        st.caption("Complete structured data")
        
        export_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'classification': results['classification'],
            'confidence': results['confidence'],
            'scores': {
                'stickiness_index': results['overall_scores'].stickiness_index,
                'snap_score': results['overall_scores'].snap_score,
                'smoothness_anomaly': results['overall_scores'].smoothness_anomaly,
                'direction_entropy': results['overall_scores'].direction_entropy,
                'human_likelihood': results['overall_scores'].human_likelihood,
                'overall_suspicion': results['overall_scores'].overall_suspicion
            },
            'features': {
                'mean_speed': results['features'].mean_speed,
                'max_speed': results['features'].max_speed,
                'speed_variance': results['features'].speed_variance,
                'jitter': results['features'].jitter,
                'smoothness': results['features'].smoothness,
                'direction_changes': results['features'].direction_changes
            }
        }
        
        st.download_button(
            "Download JSON",
            json.dumps(export_data, indent=2),
            "analysis_results.json",
            "application/json",
            use_container_width=True
        )
    
    with col3:
        st.markdown("#### Report")
        st.caption("Human-readable format")
        
        report = f"""
# Aim Behavior Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Classification Result
- Classification: {results['classification']}
- Confidence: {results['confidence']:.1%}

## Anomaly Scores
- Stickiness Index: {results['overall_scores'].stickiness_index:.3f}
- Snap Score: {results['overall_scores'].snap_score:.3f}
- Smoothness Anomaly: {results['overall_scores'].smoothness_anomaly:.3f}
- Direction Entropy: {results['overall_scores'].direction_entropy:.3f}
- Human Likelihood: {results['overall_scores'].human_likelihood:.3f}
- Overall Suspicion: {results['overall_scores'].overall_suspicion:.3f}

## Motion Features
- Mean Speed: {results['features'].mean_speed:.2f} px/s
- Max Speed: {results['features'].max_speed:.2f} px/s
- Speed Variance: {results['features'].speed_variance:.2f}
- Jitter: {results['features'].jitter:.3f}
- Smoothness: {results['features'].smoothness:.3f}
- Direction Changes: {results['features'].direction_changes}
        """
        
        st.download_button(
            "Download Report",
            report,
            "analysis_report.md",
            "text/markdown",
            use_container_width=True
        )
    
    with col4:
        st.markdown("#### Complete Package")
        st.caption("All data in one file")
        
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('trajectory.csv', df.to_csv(index=False))
            zf.writestr('results.json', json.dumps(export_data, indent=2))
            zf.writestr('report.md', report)
        
        st.download_button(
            "Download ZIP",
            buffer.getvalue(),
            "complete_analysis.zip",
            "application/zip",
            use_container_width=True
        )


# ==============================================================================
# 主程式進入點
# ==============================================================================

def render_appearance_tab():
    """Appearance / UI 自訂頁籤。"""
    st.markdown("## Appearance")
    st.caption("自訂介面外觀：格網、掃描線、霓虹主色、手槍游標（背景圖在左側 Sidebar）")

    # ---- 疊層效果 ----
    st.markdown("### 疊層效果")
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        st.toggle("顯示格網（hacker grid）",
                  value=st.session_state.get('ui_show_grid', True),
                  key='ui_show_grid')
    with col_o2:
        st.toggle("顯示掃描線（scanline）",
                  value=st.session_state.get('ui_show_scanline', True),
                  key='ui_show_scanline')

    st.divider()

    # ---- 主色 ----
    st.markdown("### 霓虹主色")
    accent = st.color_picker(
        "選擇強調色（影響標題光暈、格網、按鈕微光）",
        value=st.session_state.get('ui_accent_hex', '#38bdf8'),
        key='ui_accent_hex'
    )

    st.divider()

    # ---- 手槍游標 ----
    st.markdown("### 手槍游標")
    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        st.toggle("顯示右上角手槍",
                  value=st.session_state.get('ui_show_pistol', True),
                  key='ui_show_pistol')
    with col_p2:
        st.slider("手槍大小 (px)", 32, 120,
                  st.session_state.get('ui_pistol_size', 56), 4,
                  key='ui_pistol_size')

    st.divider()

    # ---- 重置 ----
    if st.button("重置所有外觀設定", type="secondary"):
        for k in ['bg_image_b64', 'bg_image_mime', 'bg_dim_level',
                  'ui_show_grid', 'ui_show_scanline', 'ui_accent_hex',
                  'ui_show_pistol', 'ui_pistol_size']:
            st.session_state.pop(k, None)
        st.rerun()


def _inject_appearance_overrides():
    """根據 session_state 生成動態 CSS 覆寫 + 顏色強調。"""
    import re as _re
    accent = st.session_state.get('ui_accent_hex', '#38bdf8')
    # hex → rgb
    m = _re.match(r'#([0-9a-fA-F]{6})', accent)
    if m:
        r = int(accent[1:3], 16); g = int(accent[3:5], 16); b = int(accent[5:7], 16)
    else:
        r, g, b = 56, 189, 248
    show_grid = st.session_state.get('ui_show_grid', True)
    show_scanline = st.session_state.get('ui_show_scanline', True)

    parts = ["<style>"]
    # 主色覆寫：標題光暈、指示色、主要邊框
    parts.append(f"""
        h1, h2, h3 {{
            text-shadow:
                0 0 8px rgba({r},{g},{b},0.35),
                0 0 22px rgba({r},{g},{b},0.18) !important;
        }}
        .human-indicator {{ color: {accent} !important;
            text-shadow: 0 0 8px rgba({r},{g},{b},0.5) !important; }}
        .main-header {{ border-color: rgba({r},{g},{b},0.35) !important; }}
    """)
    # 格網
    if not show_grid:
        parts.append(".stApp::before { display: none !important; }")
    else:
        parts.append(f"""
        .stApp::before {{
            background-image:
                linear-gradient(rgba({r},{g},{b},0.035) 1px, transparent 1px),
                linear-gradient(90deg, rgba({r},{g},{b},0.035) 1px, transparent 1px) !important;
        }}""")
    # 掃描線
    if not show_scanline:
        parts.append(".stApp::after { display: none !important; }")
    parts.append("</style>")
    st.markdown("\n".join(parts), unsafe_allow_html=True)


def main():
    """主程式進入點。"""
    init_session_state()
    
    # 標題
    st.markdown("""
    <div class="main-header">
        <h1>Aim Behavior Analyzer</h1>
        <p>Trajectory-based aimbot detection using motion analysis and temporal modeling</p>
    </div>
    """, unsafe_allow_html=True)

    # 動態外觀覆寫（主色、格網、掃描線）
    _inject_appearance_overrides()

    # 自訂背景圖（覆蓋格網）
    if st.session_state.get('bg_image_b64'):
        _mime = st.session_state.get('bg_image_mime', 'image/png')
        _b64 = st.session_state['bg_image_b64']
        _dim = st.session_state.get('bg_dim_level', 0.55)
        st.markdown(f"""
        <style>
            .stApp::before {{ display: none !important; }}
            .stApp {{
                background:
                    linear-gradient(rgba(0,0,0,{_dim}), rgba(0,0,0,{_dim})),
                    url("data:{_mime};base64,{_b64}") center/cover no-repeat fixed,
                    #000000 !important;
            }}
        </style>
        """, unsafe_allow_html=True)

    # 跟隨滑鼠的手槍游標（注入到父文檔）
    import streamlit.components.v1 as _components
    _show_pistol = st.session_state.get('ui_show_pistol', True)
    _pistol_size = int(st.session_state.get('ui_pistol_size', 56))
    _pistol_cfg = f"<script>window.__PISTOL_CFG__ = {{show: {str(_show_pistol).lower()}, size: {_pistol_size}}};</script>"
    _components.html(_pistol_cfg + """
    <script>
    (function() {
        try {
        const doc = window.parent.document;
        const cfg = window.__PISTOL_CFG__ || { show: true, size: 56 };
        const SHOW = cfg.show;
        const SIZE_PX = cfg.size;
        // 清理舊元素（以便切換大小 / 關閉時生效）
        const old = doc.getElementById('mouse-pistol-root');
        if (old) old.remove();
        const oldTrail = doc.getElementById('mouse-pistol-trail');
        if (oldTrail) oldTrail.remove();
        if (!SHOW) return;

        // 建立容器
        const root = doc.createElement('div');
        root.id = 'mouse-pistol-root';
        root.style.cssText = `
            position: fixed; top: 0; left: 0;
            width: 0; height: 0;
            pointer-events: none; z-index: 99999;
        `;

        // 軌跡點容器
        const trail = doc.createElement('div');
        trail.id = 'mouse-pistol-trail';
        trail.style.cssText = 'position:fixed; inset:0; pointer-events:none; z-index:99998;';
        doc.body.appendChild(trail);

        // 手槍本體（SVG，右手持槍，槍口朝右）
        const pistol = doc.createElement('div');
        pistol.id = 'mouse-pistol';
        pistol.style.cssText = `
            position: fixed;
            right: 32px; top: 72px;
            width: ${SIZE_PX}px; height: ${SIZE_PX}px;
            transform-origin: 25% 50%;
            transition: transform 0.18s cubic-bezier(.22,.9,.3,1.1);
            filter: drop-shadow(0 0 10px rgba(56,189,248,0.5)) drop-shadow(0 4px 10px rgba(0,0,0,0.7));
            pointer-events: none;
        `;
        pistol.innerHTML = `
        <svg viewBox="0 0 64 64" width="${SIZE_PX}" height="${SIZE_PX}" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="mg" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0" stop-color="#2a2f3a"/>
                    <stop offset="0.5" stop-color="#12141a"/>
                    <stop offset="1" stop-color="#05060a"/>
                </linearGradient>
                <linearGradient id="hl" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0" stop-color="#7dd3fc"/>
                    <stop offset="1" stop-color="#1e40af"/>
                </linearGradient>
            </defs>
            <!-- 槍身主體 -->
            <path d="M6 28 L48 28 L56 32 L56 38 L50 38 L48 34 L22 34 L22 42 L14 52 L8 52 L12 42 L6 36 Z"
                  fill="url(#mg)" stroke="#38bdf8" stroke-width="0.8" stroke-opacity="0.6"/>
            <!-- 槍管高光 -->
            <rect x="24" y="29" width="24" height="1.5" fill="url(#hl)" opacity="0.85"/>
            <!-- 準心 -->
            <circle cx="52" cy="31" r="1.4" fill="#38bdf8"/>
            <!-- 扳機護環 -->
            <path d="M20 34 Q20 42 28 42" stroke="#38bdf8" stroke-width="1.2" fill="none" opacity="0.7"/>
        </svg>`;

        root.appendChild(pistol);
        doc.body.appendChild(root);

        // 手槍固定在右上角；以轉軸(pivot)為旋轉中心指向滑鼠
        const RIGHT_OFFSET = 32;
        const TOP_OFFSET = 72;
        function getPivot() {
            const vw = window.innerWidth;
            return {
                x: vw - RIGHT_OFFSET - SIZE_PX + SIZE_PX * 0.25,
                y: TOP_OFFSET + SIZE_PX * 0.5
            };
        }
        // 槍口在 SVG (viewBox 0..64) 上約 (52, 31)
        // 對 pivot (16, 32 in viewBox) 偏移 ≈ (36, -1) → 距離 ~36/64 * SIZE
        const MUZZLE_OFFSET = SIZE_PX * (36/64);

        let mx = window.innerWidth * 0.3, my = window.innerHeight * 0.5;
        let curAngle = 0;

        doc.addEventListener('mousemove', (e) => {
            mx = e.clientX;
            my = e.clientY;
        });

        function tick() {
            const pv = getPivot();
            const dx = mx - pv.x;
            const dy = my - pv.y;
            const targetAngle = Math.atan2(dy, dx) * 180 / Math.PI;
            // 平滑旋轉
            let diff = ((targetAngle - curAngle + 540) % 360) - 180;
            curAngle += diff * 0.18;
            // 當槍口朝左時垂直翻轉，避免握把朝上（上下顛倒）
            const norm = ((curAngle % 360) + 540) % 360 - 180;  // (-180, 180]
            const flipY = (Math.abs(norm) > 90) ? -1 : 1;
            pistol.style.transform = `rotate(${curAngle}deg) scaleY(${flipY})`;
            requestAnimationFrame(tick);
        }
        tick();

        // (已移除射擊火光特效)
        } catch (e) {
            // 部署到嵌套 iframe 時可能因跨來源政策被擋；靜默忽略，避免整個元件掛掉
            console.warn('Pistol cursor init failed:', e);
        }
    })();
    </script>
    """, height=0)
    
    # 側邊欄設定
    settings = render_sidebar()
    
    # 主要頁籤
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Analyze Video", "Train Model", "Weight Learning", "Export Data", "Appearance"
    ])
    
    with tab1:
        render_analyze_tab(settings)
    
    with tab2:
        render_train_tab(settings)
    
    with tab3:
        render_lr_tab(settings)
    
    with tab4:
        render_export_tab()

    with tab5:
        render_appearance_tab()


if __name__ == "__main__":
    main()