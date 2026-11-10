# SAM 2 tracking in Annotation Helper

Added to `annotation_helper_current.py` as a **Dynamic SAM 2** mode inside the
existing **Propagate** action.

## Workflow

1. Open a batch so `images/` and `labels/` are loaded.
2. Go to the frame you want to use as the prompt frame.
3. In the image, select one or more existing YOLO boxes
   (`Ctrl`/`Shift`+click for multiple boxes). Every selected box is tracked as
   a separate SAM 2 object and keeps its own class index.
4. In **Actions**, select **Propagate**.
5. Choose the propagation type:
   - **Static Propagation** — the old copy-selected-boxes-to-frames behavior
   - **Dynamic SAM 2** — SAM 2 tracking
6. For Dynamic SAM 2, choose the direction:
   - **Backward** — current frame to the first frame
   - **Forward** — current frame to the last frame
   - **Entire video** — both directions, all frames
7. Set **Min confidence** (default **50%**, `0%` disables filtering).
   SAM 2's per-frame object-presence score is used; detections below the
   threshold are not written to YOLO labels.
8. Click **Start SAM 2 Tracking** (the same Apply Action button).
9. The existing model progress bar under **Model & Predict** updates while
   frames are tracked. Labels are written as each frame is processed.
10. When finished, the current frame reloads with the new boxes.
    `Undo` (or `Ctrl+Z`) restores every label file changed by the run.

Class dropdowns are not shown for Propagate; each selected box already carries
its YOLO class. They remain available for Delete and Change Class.

## Confidence threshold

SAM 2 is class-agnostic, so it does not produce a YOLO-style class confidence
per box. Instead, this integration uses SAM 2's **object-presence score**
(`object_score_logits.sigmoid()`), which indicates how strongly the tracker
believes the object is present in a frame.

- **Min confidence** is in the Dynamic SAM 2 panel, 0–100%.
- Default: **50%**
- `0%` = write every tracked detection.
- Detections below the threshold are skipped; other labels in that frame are
  preserved.
- The final status message reports how many low-confidence detections were
  skipped.

This filters object disappearance/poor-presence frames, but it is not a
guarantee of mask accuracy. For extra safety, review the run and use Undo if
the tracker drifts.

## YOLO merge policy

For every tracked frame:

- Existing labels of **other classes** are preserved.
- Existing labels of the **same class** that overlap a tracked box with
  **IoU > 0.5** are replaced by the tracked box.
- Existing same-class boxes that do not overlap are preserved.
- Frames where SAM 2 finds no mask are left unchanged.

Output lines use the standard YOLO format:

```text
class_id x_center y_center width height
```

Normalized to the original frame size (6 decimal places).

## Runtime / files

- The tool runs in the `lbl_env` conda environment.
- SAM 2 package: installed editable from
  `C:\Users\kesha\Downloads\Sam2\sam2`
- Dependencies added: `hydra-core`, `iopath`
- Checkpoint:
  `annotation_helper/sam2_models/sam2.1_hiera_small.pt`
  (~176 MB, official Meta release:
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- **First use: the checkpoint downloads automatically** (one-time, ~176 MB)
  when tracking starts, or via the **Download model** button in the
  Dynamic SAM 2 panel. It is git-ignored, so end users never need to
  copy any `.pt` file manually.
- Model config:
  `configs/sam2.1/sam2.1_hiera_s.yaml`
- Frames are mirrored to a temporary numeric-name folder while tracking
  because SAM 2's frame-folder loader requires numeric filenames; the folder
  is deleted automatically.

## Reinstall (if the environment is recreated)

```powershell
conda activate lbl_env
pip install hydra-core iopath
$env:SAM2_BUILD_CUDA="0"
pip install -e C:\Users\kesha\Downloads\Sam2\sam2 --no-deps
```

Then make sure the `sam2` package is installed (see above). The checkpoint
itself downloads automatically on first use — no manual step needed.

## Notes

- Only **selected YOLO boxes** are used as prompts; draw the box in LabelImg
  first if the object is not labeled yet.
- Tracking is limited to the current batch folder (`images/` + `labels/`).
- One SAM 2 run is processed at a time; YOLO prediction and SAM 2 tracking
  cannot run simultaneously.
- The optional SAM 2 CUDA post-processing extension is not built; the
  `cannot import name '_C'` warning is harmless.

## CPU-only systems

Yes, it works without a GPU. The worker automatically uses:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Tested on an Intel i5-13420H with the SAM 2.1 small checkpoint at 1920×1080:

- model load: ~1 s
- frame loading / init: ~2 s for 3 frames
- tracking: ~1.3 s per frame on CPU

So a 75-frame clip is roughly 1.5–2 minutes and a 150-frame clip roughly
3–4 minutes (plus frame preparation and label writes). GPU is faster, but CPU
is usable. For CPU-only deployment, 16 GB RAM is recommended; install a CPU
PyTorch wheel if the machine has no CUDA, then install SAM 2 with
`SAM2_BUILD_CUDA=0`.

