"""
SAM 3.1 helpers for labelimg-plus Annotation Helper (runs in `sam3` conda env).

1) Sam3TextModel: text-prompt detection on a single image (SAM3 image model,
   16-bit) with an ultralytics-like interface so PredictThread can use it.
2) Sam3PropagateThread: SINGLE-box forward tracking across consecutive
   images (frames) using the SAM3.1 multiplex video predictor (16-bit).
   Stages the image range into a temp mp4, prompts frame-0 with the selected
   box, propagates forward, writes one tracked box per frame.

VRAM safety (RTX 3050 6GB, measured on this machine):
  - 16-bit weights (bf16 default), complex RoPE freqs_cis kept + real/imag copies
  - use_fa3=False (FA3 needs Hopper), SDPA flash->fallback patch lives in sam3 repo
  - batched_grounding_batch_size=2, postprocess_batch_size=4, max_num_objects=8
  - frames offloaded to CPU, fresh session per run

Checkpoints (.pt) are NEVER committed: stored under annotation_helper/models/
which is git-ignored. They are auto-created on first use (~10s conversion).
"""

import gc
import os
import shutil
import tempfile
import time
import traceback

import cv2
import numpy as np
import torch

from sam3.model_builder import (
    build_sam3_image_model,
    build_sam3_multiplex_video_predictor,
    download_ckpt_from_hf,
)
from sam3.model.sam3_image_processor import Sam3Processor

# ---------------- Config ----------------
SAM3_DTYPE = torch.bfloat16
SAM3_DTYPE_NAME = "bf16"
SAM3_SCORE_THRES = 0.35
SAM3_NMS_IOU = 0.45
GROUND_BATCH = 2
POST_BATCH = 4
MAX_OBJECTS = 8

_HERE = os.path.dirname(os.path.abspath(__file__))
SAM3_MODELS_DIR = os.path.join(_HERE, "models")
os.makedirs(SAM3_MODELS_DIR, exist_ok=True)


# ---------------- 16-bit checkpoint ----------------
def ensure_half_ckpt(dtype=SAM3_DTYPE, version="sam3.1"):
    """CPU one-time conversion fp32 HF ckpt -> 16-bit file (~10s, ~1.75GB).

    Complex RoPE `freqs_cis` kept as-is AND split into _real/_imag copies:
    multiplex video model needs all three, else missing-keys + wrong RoPE.
    """
    name = "bf16" if dtype == torch.bfloat16 else "fp16"
    fname = f"sam3.{name}.pt" if version == "sam3" else f"sam3.1_multiplex.{name}.pt"
    out = os.path.join(SAM3_MODELS_DIR, fname)

    def _is_fixed(path):
        try:
            ck = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
                ck = ck["model"]
            keys = set(ck.keys())
            has_complex = any(k.endswith("freqs_cis") and not k.endswith(("_real", "_imag")) for k in keys)
            has_split = any(k.endswith("freqs_cis_real") for k in keys)
            del ck
            return has_complex and has_split
        except Exception:
            return False

    if os.path.exists(out) and _is_fixed(out):
        return out
    if os.path.exists(out):
        os.remove(out)
    src = download_ckpt_from_hf(version=version)
    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    half = {}
    for k, v in ckpt.items():
        if v.is_complex() and k.endswith("freqs_cis"):
            half[k] = v  # ViT complex path
            half[k + "_real"] = v.real.to(dtype)
            half[k + "_imag"] = v.imag.to(dtype)
        elif v.is_floating_point():
            half[k] = v.to(dtype)
        else:
            half[k] = v
    del ckpt
    torch.save(half, out)
    del half
    return out


def cast_module_16bit(module, dtype=SAM3_DTYPE):
    n = 0
    with torch.no_grad():
        for p in module.parameters():
            if p.is_floating_point() and not p.is_complex() and p.dtype != dtype:
                p.data = p.data.to(dtype)
                n += 1
        for b in module.buffers():
            if b.is_floating_point() and not b.is_complex() and b.dtype != dtype:
                b.data = b.data.to(dtype)
                n += 1
    return n


def _apply_vram_safe(predictor):
    for k, v in {"batched_grounding_batch_size": GROUND_BATCH,
                 "postprocess_batch_size": POST_BATCH,
                 "max_num_objects": MAX_OBJECTS}.items():
        if hasattr(predictor.model, k):
            setattr(predictor.model, k, v)


# ---------------- Single-image text detection (Predict panel) ----------------
class _Boxes:
    def __init__(self, items):
        self._items = items  # list of (cls_idx, x1, y1, x2, y2)

    def __iter__(self):
        for cls_idx, x1, y1, x2, y2 in self._items:
            yield _Box(cls_idx, x1, y1, x2, y2)


class _Box:
    def __init__(self, cls_idx, x1, y1, x2, y2):
        self.cls = np.array([cls_idx])
        self.xyxy = np.array([[x1, y1, x2, y2]])


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class Sam3TextModel:
    """Ultralytics-like wrapper: model(bgr_image) -> [result with .boxes]."""

    def __init__(self, prompt, class_idx, class_name,
                 score_thres=SAM3_SCORE_THRES, dtype=SAM3_DTYPE):
        self.prompt = prompt
        self.class_idx = int(class_idx)
        self.names = {0: class_name}  # remap target (PredictThread uses this)
        self.score_thres = score_thres
        half_ckpt = ensure_half_ckpt(dtype, version="sam3")
        model = build_sam3_image_model(
            device="cuda" if torch.cuda.is_available() else "cpu",
            eval_mode=True, checkpoint_path=half_ckpt, load_from_HF=False,
        )
        model.eval()
        cast_module_16bit(model, dtype)
        self.processor = Sam3Processor(model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ctx(self):
        if self.device == "cuda":
            return torch.autocast(device_type="cuda", dtype=SAM3_DTYPE)
        return torch.no_grad()

    def __call__(self, img_bgr):
        from PIL import Image
        h, w = img_bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        with torch.inference_mode():
            with self._ctx():
                state = self.processor.set_image(pil)
            with torch.inference_mode():
                with self._ctx():
                    result = self.processor.set_text_prompt(state=state, prompt=self.prompt)
        scores = result.get("scores", result.get("out_probs"))
        boxes = result.get("boxes", result.get("out_boxes_xyxy"))
        if scores is None:
            return [_Result(_Boxes([]))]
        scores = np.asarray(torch.as_tensor(scores).detach().float().cpu().numpy()).reshape(-1)
        if boxes is not None:
            boxes = np.asarray(torch.as_tensor(boxes).detach().float().cpu().numpy()).reshape(-1, 4)
        else:
            boxes = np.zeros((0, 4))
        items = []
        for i, s in enumerate(scores):
            if float(s) < self.score_thres or i >= len(boxes):
                continue
            x1, y1, x2, y2 = [float(v) for v in boxes[i]]
            x1 = max(0, min(w - 1, x1)); y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2)); y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            items.append((0, x1, y1, x2, y2))
        # same-class NMS (keep it dependency-free)
        items = _nms(items)
        return [_Result(_Boxes(items))]


def _nms(items, iou_thres=SAM3_NMS_IOU):
    if len(items) <= 1:
        return items
    order = sorted(range(len(items)), key=lambda i: 1.0, reverse=False)
    kept = []
    used = [False] * len(items)
    for i in range(len(items)):
        if used[i]:
            continue
        kept.append(items[i])
        ax1, ay1, ax2, ay2 = items[i][1:]
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            bx1, by1, bx2, by2 = items[j][1:]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            aa = (ax2 - ax1) * (ay2 - ay1)
            bb = (bx2 - bx1) * (by2 - by1)
            if inter / (aa + bb - inter + 1e-6) > iou_thres:
                used[j] = True
    return kept


# ---------------- Single-box forward tracking (QThread) ----------------
from PyQt5.QtCore import QThread, pyqtSignal  # noqa: E402


def _iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-6)


class Sam3PropagateThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(int, str)
    error = pyqtSignal(str)
    snapshot = pyqtSignal(dict)

    def __init__(self, images, images_dir, labels_dir, start_idx, end_idx,
                 box_parts, score_thres=0.15):
        """box_parts: [cls, xc, yc, w, h] strings (normalized, from start image).

        score_thres intentionally low (0.15): tracker confidence fluctuates
        frame to frame; gaps are filled by interpolation below instead of
        dropping boxes (annotator deletes false positives faster than
        drawing missing boxes).
        """
        super().__init__()
        self.images = images
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.box_parts = box_parts
        self.score_thres = score_thres

    def _read_labels(self, name):
        path = os.path.join(self.labels_dir, os.path.splitext(name)[0] + ".txt")
        lines = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        return path, lines

    def run(self):
        import uuid
        updated = 0
        files_snapshot = {}
        tmpdir = tempfile.mkdtemp(prefix="sam3track_")
        predictor = None
        try:
            names = self.images[self.start_idx - 1:self.end_idx]
            if not names:
                self.finished.emit(0, "No target images.")
                return
            # stage frames: uniform size mp4 (first frame's size)
            first = cv2.imread(os.path.join(self.images_dir, names[0]))
            if first is None:
                raise RuntimeError(f"Cannot read {names[0]}")
            H0, W0 = first.shape[:2]
            clip = os.path.join(tmpdir, "clip.mp4")
            vw = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (W0, H0))
            for nm in names:
                fr = cv2.imread(os.path.join(self.images_dir, nm))
                if fr is None:
                    raise RuntimeError(f"Cannot read {nm}")
                if fr.shape[1] != W0 or fr.shape[0] != H0:
                    fr = cv2.resize(fr, (W0, H0), interpolation=cv2.INTER_AREA)
                vw.write(fr)
            vw.release()

            cls_id = int(float(self.box_parts[0]))
            xc, yc, bw, bh = map(float, self.box_parts[1:5])
            box_xywh_rel = [[xc - bw / 2, yc - bh / 2, bw, bh]]  # x,y,w,h rel

            half_ckpt = ensure_half_ckpt(SAM3_DTYPE, version="sam3.1")
            predictor = build_sam3_multiplex_video_predictor(
                checkpoint_path=half_ckpt, use_fa3=False, use_rope_real=True,
                async_loading_frames=False,
            )
            cast_module_16bit(predictor.model, SAM3_DTYPE)
            _apply_vram_safe(predictor)

            inference_state = predictor.model.init_state(
                resource_path=clip, offload_video_to_cpu=True,
                async_loading_frames=False)
            session_id = str(uuid.uuid4())
            predictor._all_inference_states[session_id] = {
                "state": inference_state, "session_id": session_id,
                "start_time": time.time(), "last_use_time": time.time()}

            predictor.handle_request(dict(type="add_prompt", session_id=session_id,
                                          frame_index=0,
                                          bounding_boxes=box_xywh_rel,
                                          bounding_box_labels=[1]))
            total = len(names)
            stream = predictor.handle_stream_request(dict(
                type="propagate_in_video", session_id=session_id,
                propagation_direction="forward",
                max_frame_num_to_track=total))
            # Phase 1: collect best tracked box per frame (single object).
            tracked = {}  # fi -> (cx, cy, w, h)
            done = 0
            for resp in stream:
                fi = resp["frame_index"]
                if fi < 0 or fi >= total:
                    continue
                out = resp["outputs"]
                ids = np.asarray(out.get("out_obj_ids", []))
                probs = np.asarray(out.get("out_probs", np.zeros(len(ids))))
                rboxes = np.asarray(out.get("out_boxes_xywh", np.zeros((len(ids), 4))))
                best, best_p = None, -1.0
                for j in range(len(ids)):
                    p = float(probs[j]) if j < len(probs) else 0.0
                    if j >= len(rboxes) or p < self.score_thres:
                        continue
                    x, y, w_, h_ = [float(v) for v in rboxes[j].tolist()]
                    if w_ <= 0 or h_ <= 0:
                        continue
                    if p > best_p:
                        best, best_p = (x + w_ / 2, y + h_ / 2, w_, h_), p
                if best is not None:
                    tracked[fi] = best
                done += 1
                self.progress.emit(int(done / total * 100))
                if done % 10 == 0:
                    torch.cuda.empty_cache()

            # Phase 2: interpolate interior gaps (CVAT-style) so every frame
            # between first and last tracked frame gets a box.
            interp = 0
            if tracked:
                lo, hi = min(tracked), max(tracked)
                fis = sorted(tracked)
                for a, b in zip(fis, fis[1:]):
                    if b - a <= 1:
                        continue
                    ba = tracked[a]
                    bb = tracked[b]
                    for fi in range(a + 1, b):
                        t = (fi - a) / (b - a)
                        tracked[fi] = tuple(ba[k] + (bb[k] - ba[k]) * t for k in range(4))
                        interp += 1

            # Phase 3: write one tracked box per frame (dedup same-class overlaps).
            for fi in sorted(tracked):
                name = names[fi]
                label_path, original_lines = self._read_labels(name)
                exist = [ln.strip().split() for ln in original_lines if ln.strip()]
                cx, cy, w_, h_ = tracked[fi]
                x1, y1, x2, y2 = cx - w_ / 2, cy - h_ / 2, cx + w_ / 2, cy + h_ / 2
                kept = []
                for p in exist:
                    try:
                        if int(float(p[0])) == cls_id:
                            px, py, pw, ph = map(float, p[1:5])
                            pa = [px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2]
                            if _iou_xyxy([x1, y1, x2, y2], pa) > 0.3:
                                continue
                    except Exception:
                        pass
                    kept.append(p)
                kept.append([str(cls_id), f"{cx:.6f}", f"{cy:.6f}",
                             f"{w_:.6f}", f"{h_:.6f}"])
                new_lines = [" ".join(x) + "\n" for x in kept]
                if new_lines != original_lines:
                    files_snapshot[label_path] = original_lines
                    with open(label_path, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                    updated += 1

            self.snapshot.emit(files_snapshot)
            self.finished.emit(updated, f"SAM3 tracked 1 object across {updated} files "
                                       f"({interp} interpolated).")
        except Exception as e:
            self.error.emit(f"SAM3 track error: {e}\n{traceback.format_exc()}")
        finally:
            try:
                if predictor is not None:
                    for sid in list(predictor._all_inference_states.keys()):
                        try:
                            predictor.handle_request(
                                dict(type="close_session", session_id=sid))
                        except Exception:
                            pass
                    del predictor
            except Exception:
                pass
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            shutil.rmtree(tmpdir, ignore_errors=True)
