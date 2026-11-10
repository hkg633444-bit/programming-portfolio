"""Training UI for the versioned temporal pipeline."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from training_pipeline import (
    load_dataset, save_dataset, append_video, video_id, temporal_windows,
    grouped_split, split_manifest, standardize,
)


def render(extract, save_upload, model_class, config_class):
    st.markdown("## Train Temporal Model")
    st.info("固定 2 秒視窗、1 秒步長（50% 重疊）；統一 30 Hz，每序列 60 × 6。"
            "只使用畫面位移估計，強制逐幀提取，不受側邊欄舊中心法設定影響。")
    st.caption("先按來源分配集合，再以該來源的所有視窗訓練或評估。"
               "來源比例目標為 60/20/20，依各類影片數取整；少量資料比例會不同。"
               "至少每類 3 個獨立來源。相同玩家／場次的剪輯請填相同來源代號。")
    try:
        data = load_dataset()
    except Exception as exc:
        st.error(f"無法載入訓練資料：{exc}")
        return
    label = st.radio("Video Label", ["Human", "Aimbot"], horizontal=True, key="temporal_label")
    group_name = st.text_input("來源代號（同一場錄影的所有剪輯填相同值；留空以影片內容分組）",
                               key="temporal_group")
    upload = st.file_uploader("Upload Training Video", type=["mp4", "avi", "mov", "mkv", "webm"],
                              key="temporal_upload")
    if st.button("Add to Training Set", disabled=upload is None, key="temporal_add"):
        path = None
        try:
            content_id = video_id(upload.getvalue())
            if any(v["id"] == content_id for v in data["videos"]):
                raise ValueError("這部影片已加入，不接受重複內容")
            path = save_upload(upload, suffix=".mp4")
            with st.spinner("Extracting full-rate screen motion..."):
                positions, times, _ = extract(path, frame_interval=1, detection_method="phase")
                sequences, starts = temporal_windows(positions, times)
                group = "source:" + video_id(group_name.strip().encode()) if group_name.strip() else content_id
                updated = append_video(data, content_id, group, int(label == "Aimbot"),
                                       upload.name, sequences, starts)
                save_dataset(updated, expected_ids=[v["id"] for v in data["videos"]])
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
        finally:
            if path and os.path.exists(path):
                os.unlink(path)
    st.metric("Total Training Sequences", len(data["y"]))
    st.caption("此頁訓練並保存模型；Analyze Video 目前仍使用規則評分，不會自動使用此模型。")
    st.caption(f"已保存 {len(data['videos'])} 部影片，{len(set(data['groups']))} 個來源；重啟後仍可載入。")
    if data["videos"]:
        st.dataframe([{k: v[k] for k in ("filename", "label", "windows", "group")}
                      for v in data["videos"]], use_container_width=True)
    parts = None
    try:
        parts = grouped_split(data["y"], data["groups"])
        report = split_manifest(data, parts)
        st.dataframe([{"split": key, **{k: value[k] for k in
                       ("video_count", "window_count", "human_windows", "aimbot_windows")}}
                      for key, value in report["splits"].items()], use_container_width=True)
        st.success("來源交集檢查通過：訓練／驗證／測試兩兩交集皆為 0。")
        st.download_button("Download Split Manifest", json.dumps(report, indent=2),
                           "split-manifest.json", "application/json")
    except ValueError as exc:
        st.warning(str(exc))
    st.warning("測試集只在訓練完成後評估；不要根據測試成績反覆調參。"
               "同內容雜湊可防重新命名的重複檔，但不能自動識別重編碼／剪輯；請用來源代號合併。")
    epochs = st.slider("Epochs", 1, 100, 30, key="temporal_epochs")
    batch = st.selectbox("Batch Size", [16, 32, 64], index=1, key="temporal_batch")
    if st.button("Start Training", disabled=parts is None, type="primary", key="temporal_train"):
        import tensorflow as tf
        try:
            tf.keras.utils.set_random_seed(42)
            x, mean, scale = standardize(data["x"], parts["train"])
            y = tf.keras.utils.to_categorical(data["y"], 2)
            model = model_class(config_class(sequence_length=60, feature_dim=6,
                                             epochs=epochs, batch_size=batch))
            model.build_model()
            progress = st.empty()

            class Progress(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress.text(f"Epoch {epoch + 1}: validation loss={logs['val_loss']:.4f}")

            with st.spinner("Training on source-disjoint sequences..."):
                history = model.model.fit(
                    x[parts["train"]], y[parts["train"]],
                    validation_data=(x[parts["validation"]], y[parts["validation"]]),
                    epochs=epochs, batch_size=batch, verbose=0,
                    callbacks=[Progress(), tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=5, restore_best_weights=True)])
                probabilities = model.model.predict(x[parts["test"]], verbose=0)
                pred = probabilities.argmax(axis=1)
            report["test_metrics"] = classification_report(
                data["y"][parts["test"]], pred, labels=[0, 1],
                target_names=["Human", "Aimbot"], output_dict=True, zero_division=0)
            report["test_confusion_matrix"] = confusion_matrix(
                data["y"][parts["test"]], pred, labels=[0, 1]).tolist()
            source_predictions, source_labels = [], []
            test_groups = data["groups"][parts["test"]]
            for source in np.unique(test_groups):
                mask = test_groups == source
                source_predictions.append(int(probabilities[mask].mean(axis=0).argmax()))
                source_labels.append(int(data["y"][parts["test"]][mask][0]))
            report["source_level_test_metrics"] = classification_report(
                source_labels, source_predictions, labels=[0, 1],
                target_names=["Human", "Aimbot"], output_dict=True, zero_division=0)
            report["history"] = {k: [float(v) for v in values] for k, values in history.history.items()}
            report["metric_unit"] = "window; overlapping windows are not independent observations"
            run = Path("saved_models") / ("temporal_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
            run.mkdir(parents=True)
            model.model.save(run / "model.keras")
            np.savez(run / "scaler.npz", mean=mean, scale=scale)
            (run / "report.json").write_text(json.dumps(report, indent=2))
            st.session_state["temporal_last_report"] = report
            st.success(f"訓練完成，模型、訓練集標準化參數與完整報告已保存：{run}")
        except Exception as exc:
            st.error(f"訓練未完成：{exc}")
    if st.session_state.get("temporal_last_report"):
        result = st.session_state["temporal_last_report"]
        st.markdown("### Held-out Test Results")
        st.json(result["test_metrics"])
        st.caption("以下以來源為單位，平均該來源所有視窗的預測機率後分類，避免長影片主導計數。")
        st.json(result["source_level_test_metrics"])
        st.write("Confusion matrix (rows=true, columns=predicted; Human, Aimbot)",
                 result["test_confusion_matrix"])
        st.download_button("Download Training Report", json.dumps(result, indent=2),
                           "training-report.json", "application/json")
