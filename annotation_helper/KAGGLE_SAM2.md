# Kaggle T4 x2 offload for YOLO predict + SAM 2 tracking

This is the optional cloud path for the heavy Annotation Helper tasks.  The
local laptop remains the source of truth; Kaggle only performs the expensive
inference/tracking and returns YOLO labels.

```
Annotation Helper (Python 3.10, local)
        │
        │  tick "Run heavy work on Kaggle (T4 x2)"
        ▼
KaggleJobThread ──► KaggleJobRunner
        │
        ├─ 1. reuse model/checkpoint from the private weights dataset
        │     (models are uploaded only ONCE)
        ├─ 2. upload the current images folder ONCE per app session
        │     (private session dataset; reused by every click)
        ├─ 3. per click: embed the tiny command (prompts/labels/params)
        │     into a new version of ONE fixed kernel (no command dataset)
        ├─ 4. the kernel mounts session + weights, runs on NvidiaTeslaT4
        ├─ 5. download coordinates.json + result.json (no images back)
        └─ 6. merge coordinates into local labels/ + Undo snapshot,
              then jump to the first changed frame so boxes are visible
```

When LabelImg closes, the session image dataset is deleted automatically;
the model weights dataset and the reusable kernel stay.

### Models once, images later

The first time a model is used, it is uploaded once to a private dataset
(default `<username>/labelimg-plus-weights`).  Every later job sends only the
selected images, their current labels and a tiny manifest.  The kernel mounts
the weights dataset alongside the job dataset, reads the model/checkpoint from
`/kaggle/input/...`, runs inference, and returns **only YOLO coordinates**.

Example from the live validation:

* models inside every job: ~153 MB per click
* model reused from weights dataset: **0.14 MB per click**
* session mode (images uploaded once, command embedded): **578 bytes per click**
* SAM 2 with local wheel + reused checkpoint: 5/5 frames in ~121 s

Run the one-time sync explicitly if you want all local models uploaded now:

```bash
python -m kaggle_bridge sync-weights
```

The first real job also uploads its selected model automatically, so this is
optional.  Updated model files are detected by size/mtime and a new weights
dataset version is created.

### Fast session mode

Enabled by default (`KAGGLE_USE_SESSION_DATASET=true`):

1. Dataset load / Kaggle checkbox -> the **whole current images folder** is
   packed and uploaded once as a private session dataset.  This runs in the
   background; the UI shows `Kaggle: uploading session images...`.
2. Every Predict/SAM 2 click writes only a tiny command (selected boxes,
   range, classes, current labels) into a new version of the fixed kernel.
   No images and no separate command dataset are uploaded.
3. **Cancel** button in the Compute card stops a queued/running kernel
   immediately (deletes that kernel; the next click recreates it).
4. **GPU-only fallback:** T4 x2 → L4 x1 → H100 → A100.  Kaggle has no
   separate single-T4 option: `NvidiaTeslaT4` is always T4 x2.  Each
   accelerator gets **`KAGGLE_QUEUE_TIMEOUT_MINUTES` = 1** by default
   (fast mode); if it is still queued, the bridge cancels it and tries the
   next.  If none starts, the job **stops** with a clear LabelImg error.
   CPU is never used.  The **Check GPUs** button runs a **20-second** probe
   per accelerator and shows the actual GPU names (or CPU/queued).
   Kaggle has no public "is a GPU free right now?" API: the only way to know
   is to request a kernel, which is exactly what the probe/job does.
5. The fixed kernel `labelimg-plus-runner` is reused; each click creates a
   new version (visible in Kaggle, no kernel spam).
6. On app close, the session dataset is deleted from Kaggle.  The weights
   dataset and the fixed kernel remain.  If a job is still running, the app
   waits up to 10 minutes before closing/cleaning.

SAM 2 is installed from a prebuilt local wheel shipped inside the session
dataset, so the old 4-5 minute git build is gone (~15 s instead).

The local Predict button and local Dynamic SAM 2 are unchanged.  Cloud mode is
opt-in and can be turned off at any time.

---

## 1. What is needed

| Requirement | Notes |
|---|---|
| Kaggle account | Free |
| Phone verification | Required by Kaggle for GPU / internet kernels. If the `quota` command shows GPU hours, it is already done. |
| Kaggle API token | `kaggle.json` or a new `KAGGLE_API_TOKEN`. Keep it secret. |
| Internet-enabled kernel | Needed for the one-time `pip install` of Ultralytics/SAM 2 and the SAM 2 checkpoint download. |
| Python 3.11+ | Only for the Kaggle CLI venv. The helper itself can stay on Python 3.10. |
| Disk space | The job bundle is a copy of the frames + selected model. Keep at least a few GB free. |
| Weekly GPU quota | Free Kaggle accounts get 30 GPU hours/week. T4 x2 counts as one wall-clock hour per hour, even though two GPUs are attached. |

Check everything with:

```bash
python -m kaggle_bridge.doctor
```

The doctor prints the Kaggle username, CLI version, GPU quota, credentials
source and any missing setup.  Secrets are never printed.

Current token locations supported (first match wins):

1. `KAGGLE_API_TOKEN` in `.env`
2. `KAGGLE_USERNAME` + `KAGGLE_KEY` in `.env`
3. `%USERPROFILE%\.kaggle\kaggle.json` (Windows)
4. `~/.kaggle/access_token`

`.env` is git-ignored.  Never commit it.

---

## 2. One-time setup

### 2.1 Create the Kaggle CLI venv

The Kaggle CLI 2.2+ requires Python 3.11+.  The helper environment may be
Python 3.10, so the CLI is installed into a separate project venv:

```bash
python -m kaggle_bridge.setup --create-venv
```

This creates `.kaggle-venv/` and installs `kaggle>=2.2,<3` plus `six` (an
undeclared Kaggle dependency).  On the developer machine this used the base
Python 3.13 while `lbl_env` stayed on Python 3.10.

If the venv is broken:

```bash
python -m kaggle_bridge.setup --create-venv --force
```

### 2.2 Create `.env`

```bash
python -m kaggle_bridge.setup --write-env
```

Then edit `.env` and fill one credential method.  Example:

```dotenv
KAGGLE_API_TOKEN=your-token-here
```

If you downloaded `kaggle.json`, put it at `%USERPROFILE%\.kaggle\kaggle.json`
and leave the token fields blank.

### 2.3 Build the SAM 2 wheel + upload models once

```bash
python -m kaggle_bridge.setup --build-sam2-wheel
python -m kaggle_bridge sync-weights
```

The wheel is built once into `kaggle_weights/wheels/` and is shipped inside
every session dataset.  `sync-weights` finds `annotation_helper/models/*.pt`
(YOLO) and `annotation_helper/sam2_models/*.pt` and uploads them to
`<username>/labelimg-plus-weights`.  Both commands are safe to run again;
unchanged files are skipped.

### 2.4 Verify

```bash
python -m kaggle_bridge.doctor
```

Expected:

```
Authenticated: yes (your-user)
GPU quota:    30.00h remaining of 30.00h
Everything looks ready.
```

### 2.5 Use it from the main window

1. Open a batch (`images/` + `labels/`) in the normal LabelImg window.
2. Press `Ctrl+Shift+K` for **Kaggle controls** (settings, jobs, session
   upload, connection and GPU checks).
3. For predict: press `Ctrl+Shift+Y`, choose a YOLO model, tick **Run on
   Kaggle (T4 x2)**, then **Run predict**.
4. For tracking: select one or more boxes, press `Ctrl+Shift+T`, choose the
   SAM 2 model/direction, tick **Run on Kaggle**, then **Start tracking**.
5. The job uploads, runs, downloads and merges automatically.  `Ctrl+Alt+Z`
   restores every label file changed by the import.

The **Kaggle settings** button opens the token / CLI / connection-test dialog.
The **Job history** button opens local job history and a **Resume selected**
button for interrupted jobs.

---

## 3. What gets uploaded

In fast session mode the uploads are:

```
<username>/labelimg-plus-session-...  images + runner_lib.py + SAM2 wheel
                                      (one per app session, deleted on close)
<username>/labelimg-plus-weights      YOLO models + SAM 2 checkpoints
                                      (persistent, uploaded once)
<username>/labelimg-plus-runner       ONE fixed kernel; new version per click
```

The per-click command is embedded directly in `runner.py`:

```
manifest session.json   all image names/order/dimensions
command (embedded)      selected boxes, range, classes, current labels, model
images/000000.jpg       only in the session dataset, not per click
```

The legacy per-job bundle (`KAGGLE_USE_SESSION_DATASET=false`) still works and
uses a private job dataset that is deleted after success.

If no SAM 2 checkpoint is available locally, the kernel downloads the official
Meta checkpoint once per run (~176 MB).  To avoid that, put the checkpoint in
`annotation_helper/sam2_models/` and run `sync-weights`.  No Kaggle credential
is ever put in the bundle.

Kaggle auto-extracts the zip inside the private dataset.  The runner supports
both layouts: an extracted dataset root (`manifest.json` + `images/`) and a
real `job.zip` for local tests.

### Kernel output

```
output/coordinates.json  {original_image_name: [[cls, xc, yc, w, h], ...]}
output/labels.json       same coordinates (compatibility alias)
output/result.json       status, counts, kernel stats
output/status.json       last progress marker
```

No images, masks or model files are written to `/kaggle/working`; the input is
extracted under `/tmp` and discarded.  Only changed frames are returned.  The
helper reloads the batch and jumps to the first changed frame, so the returned
boxes are immediately visible in LabelImg.  The local importer validates:
- the filename exists in this batch,
- the path cannot escape `labels/`,
- the class index is inside `classes.txt`,
- coordinates are valid normalized YOLO values.

The dataset is deleted after a successful import.  The kernel is kept by
default for logs/audit; set `KAGGLE_DELETE_KERNEL_AFTER=true` to delete it too.

---

## 4. Job states and recovery

States: `created -> bundling -> uploading_dataset -> waiting_dataset ->
pushing_kernel -> queued/running -> downloading -> importing -> done`.

On failure the job is marked `failed`, the kernel log is fetched and the local
job folder is kept.  Use:

```bash
python -m kaggle_bridge jobs
python -m kaggle_bridge status --job <job-id> --log
python -m kaggle_bridge resume --job <job-id>
```

`resume` checks the remote kernel first:
- queued/running: continue polling,
- complete: download + import,
- error: show the log tail,
- not found: prepare a fresh kernel.

The helper's **Jobs... -> Resume selected** does the same from the UI.

---

## 5. Configuration (`.env`)

| Key | Default | Meaning |
|---|---|---|
| `KAGGLE_API_TOKEN` | empty | New Kaggle access token (recommended). |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | empty | Legacy API key. |
| `KAGGLE_CONFIG_DIR` | `~/.kaggle` | Custom Kaggle config folder. |
| `KAGGLE_CLI` | auto | Explicit Kaggle CLI 2.x executable. |
| `KAGGLE_PYTHON` | auto | Python 3.11+ used to build `.kaggle-venv`. |
| `KAGGLE_DATASET_PREFIX` | `labelimg-plus-job` | Private job dataset slug. |
| `KAGGLE_JOB_ROOT` | `kaggle_jobs` | Local job state/log/output folder. |
| `KAGGLE_USE_WEIGHTS_DATASET` | `true` | Reuse models from the weights dataset instead of uploading them per job. |
| `KAGGLE_WEIGHTS_DATASET` | `labelimg-plus-weights` | Persistent private weights dataset slug. |
| `KAGGLE_WEIGHTS_ROOT` | `kaggle_weights` | Local staging folder for weight uploads. |
| `KAGGLE_USE_SESSION_DATASET` | `true` | Upload the current images folder once per app session. |
| `KAGGLE_SESSION_DATASET_PREFIX` | `labelimg-plus-session` | Session dataset slug prefix. |
| `KAGGLE_FIXED_KERNEL` | `true` | Reuse one kernel slug; new version per click. |
| `KAGGLE_KERNEL_SLUG` | `labelimg-plus-runner` | Fixed kernel slug. |
| `KAGGLE_MAX_SESSION_GB` | `20` | Refuse a session images upload larger than this. |
| `KAGGLE_SESSION_KEEP_LOCAL` | `false` | Keep the local session zip after upload. |
| `KAGGLE_ACCELERATORS` | `NvidiaTeslaT4,NvidiaL4X1,NvidiaH100,NvidiaTeslaA100` | GPU fallback order. |
| `KAGGLE_QUEUE_TIMEOUT_MINUTES` | `1` | Fixed per-accelerator queue wait (0.5-5 recommended). |
| `KAGGLE_REQUIRE_GPU` | `true` | If true, never import CPU results and never add CPU. |
| `KAGGLE_CPU_FALLBACK` | `false` | Only used when `KAGGLE_REQUIRE_GPU=false`. |
| `KAGGLE_POLL_SECONDS` | `30` | Kernel status poll interval. |
| `KAGGLE_TIMEOUT_HOURS` | `6` | Give up waiting after this long. |
| `KAGGLE_ACCELERATOR` | `NvidiaTeslaT4` | Kaggle GPU T4 x2. |
| `KAGGLE_DELETE_DATASET_AFTER` | `true` | Delete the remote dataset after success. |
| `KAGGLE_DELETE_KERNEL_AFTER` | `false` | Also delete the kernel/output. |
| `KAGGLE_INCLUDE_SAM2_CHECKPOINT` | `true` | Include a local SAM 2 checkpoint if present, else download in-kernel. |
| `KAGGLE_KEEP_BUNDLE` | `false` | Keep the local `job.zip` after success. |
| `KAGGLE_MAX_BUNDLE_MB` | `4096` | Refuse oversized uploads. |

---

## 6. Quotas and limits

- **30 GPU hours/week** on the free tier; refresh is shown by
  `kaggle quota`.  A T4 x2 run consumes quota by wall-clock time, not 2x.
- **12-hour** maximum kernel session.
- One remote job is allowed at a time locally.  This avoids dataset/kernel
  name collisions and keeps the GPU quota predictable.
- Kaggle's kernel output and dataset limits apply; keep jobs to a reasonable
  number of frames.  The bundle is a copy of the frames, so a 5,000-frame
  1080p clip can be several GB.
- SAM 2 uses one T4 for the predictor.  The second T4 is attached and visible
  (`gpu_count=2` in `result.json`) but the current video predictor is
  single-GPU.  It can be used for future batched/parallel jobs.
- The first session upload can take minutes for a large folder (e.g. 4000
  frames).  It is done once per app session; every click afterwards sends only
  a ~0.5-1 KB command.
- YOLO click: about 20-40 seconds after the kernel starts.
- SAM 2 click: about 2 minutes, mostly model loading + tracking.  The SAM 2
  package comes from the local wheel (~15 s instead of a 4-5 minute git build)
  and the checkpoint is reused from the weights dataset.
- If T4 x2 stays queued for 1 minute (fast default), the bridge automatically
  tries L4 x1/H100/A100 (if your account is eligible).  If none is allocated,
  the job stops; CPU is not used (`KAGGLE_REQUIRE_GPU=true`).  Set
  `KAGGLE_QUEUE_TIMEOUT_MINUTES=5` for patient mode.
- The fixed kernel keeps one version per click.  Set
  `KAGGLE_DELETE_KERNEL_AFTER=true` or delete it manually if you do not want
  version history.

---

## 7. Verified live results

Validated on 2026-11-13 with Kaggle user `dukeshavrao`:

| Test | Result |
|---|---|
| One-time weights upload | `yolov11l_jul10.pt` (146 MB) + `sam2.1_hiera_small.pt` (176 MB) in `dukeshavrao/labelimg-plus-weights` |
| YOLO `yolov8n` on 1 bus image | `status=ok`, 5 boxes, 1 label file merged, `gpu_count=2`, 21.1 s kernel |
| YOLO `yolov11l_jul10.pt` with weights reuse | job bundle **0.14 MB** (was 153 MB), 7 boxes returned, 22.9 s kernel, `gpu_count=2` |
| SAM 2.1 Small, no checkpoint in bundle | job bundle **0.04 MB**, checkpoint reused from weights dataset, 5/5 frames tracked, 318 s (SAM 2 package install time), `gpu_count=2` |
| SAM 2.1 Small, first run with download | 1 object, 5/5 frames tracked, 5 label files merged, `device=cuda`, `gpu_count=2`, 203.5 s including install/checkpoint |
| Session mode, first YOLO click | 5 images uploaded once (285 KB), fixed kernel v4, 7 boxes, **578-byte command** |
| Session mode, second YOLO click | session dataset reused, **no image upload**, fixed kernel v5, 7 boxes |
| Session mode + SAM2 local wheel | 15.5 s wheel install, 5/5 frames tracked, **121 s total**, `gpu_count=2` |
| Session cleanup | session dataset deleted from Kaggle; weights + fixed kernel remain |
| Authentication | `dukeshavrao`, GPU quota 30h remaining |
| Unit tests | 24 tests pass on Python 3.10 and 3.13 |

---

## 8. Troubleshooting

### `409 Conflict` on `SaveKernel`

Kaggle does not allow a dataset and a kernel to share the same slug.  The
bridge uses `labelimg-plus-job-...` for datasets and
`labelimg-plus-run-...` for kernels.  Do not reuse the same prefix for both.

### `job.zip not found under /kaggle/input`

Kaggle auto-extracts uploaded zip files.  The runner supports both an
extracted dataset root and a real zip.  If you see this on an old build,
update `kaggle_bridge/kernel_runner.py`.

### Job is QUEUED for a long time

The Compute card now has **Cancel Kaggle job** (and in the jobs dialog the
same remote kernel can be stopped).  The bridge gives each accelerator
`KAGGLE_QUEUE_TIMEOUT_MINUTES` (1 minute fast default) and then automatically
moves to the next one.  If none starts, the job stops and LabelImg shows
`No usable GPU was allocated`; CPU is not used.

Kaggle does not expose a separate single-T4 accelerator: `NvidiaTeslaT4` is
always T4 x2.  The kernel can use one T4, but the queue/allocation is for the
x2 option.

### Check GPUs first

**Kaggle setup → Check GPUs** runs a 20-second tiny private probe for each
configured accelerator and prints the actual result, e.g.:

```
NvidiaTeslaT4: 2 GPU: Tesla T4, Tesla T4
NvidiaL4X1:    2 GPU: Tesla T4, Tesla T4
NvidiaH100:    2 GPU: Tesla T4, Tesla T4
NvidiaTeslaA100: QUEUED after 45s (no allocation)
```

On this account the L4/H100 requests currently resolve to 2x T4; A100 may
queue.  There is no usable single-T4 option.  TPU is not used because this
pipeline needs PyTorch/SAM 2 (`torch_xla` is absent and TF sees CPU only).

### Session images upload is slow / too large

The whole current `images/` folder is uploaded once.  For 4000+ frames this
can be several GB.  The bridge refuses sessions above `KAGGLE_MAX_SESSION_GB`
(default 20).  Reduce the batch or raise the limit.  The next click reuses the
session dataset, so the wait is one-time per app session.

### App close waits for a running job

If a Kaggle job is running when LabelImg closes, the app waits up to 10
minutes for it.  If it is still queued/running after that, the session
dataset is left for the next launch and a stale-cleanup runs then.  This
prevents deleting images under a live kernel.

### Session cleanup

```bash
python -m kaggle_bridge session-cleanup
python -m kaggle_bridge session-cleanup --stale-hours 24
```

Only the session image dataset (and any tracked legacy command datasets) is
deleted.  `<username>/labelimg-plus-weights` and the fixed kernel stay.

### `NameError: null` in the kernel

Older builds embedded JSON (`null`) into Python.  Current builds embed a
Python literal (`None`); update `kaggle_bridge/session.py`.

### `ModuleNotFoundError: runner_lib`

Kaggle script kernels only upload a single `code_file`.  `runner_lib.py` is
inside `job.zip`; the runner imports it after extraction.  Do not move it back
to the kernel folder.

### `403 Forbidden` while checking a new dataset

Kaggle returns 403 from `GetDatasetStatus` while a freshly created private
dataset is still being processed (observed on a ~100 MB upload).  This is no
longer fatal: the bridge treats 403 as `processing`, keeps polling and
continues when the dataset becomes `ready`.  A 403 line can still appear in
the log; that is expected.

If a dataset stays unqueryable for 30 minutes, the job fails with a clear
message instead of hanging forever.

### Kernel ERROR

Fetch the log:

```bash
python -m kaggle_bridge status --job <job-id> --log
```

The worker also saves `kaggle_jobs/<job-id>/logs/kernel.log`.

Common causes:
- Kaggle account not phone-verified -> GPU unavailable.
- Internet disabled -> SAM 2 pip install/checkpoint fails.
- No GPU quota left -> kernel queued or downgraded.
- SAM 2 config/package mismatch -> see the kernel log; re-run after updating
  the SAM 2 repository.
- Model/checkpoint too large or bundle over the configured limit.

### `Authentication required`

Create a fresh token at <https://www.kaggle.com/settings/api> and either
replace `%USERPROFILE%\.kaggle\kaggle.json` or update `KAGGLE_API_TOKEN` in
`.env`.  Then run `python -m kaggle_bridge.doctor`.

### Kaggle CLI missing or too old

```bash
python -m kaggle_bridge.setup --create-venv --force
```

The CLI must be 2.2+ for `--accelerator NvidiaTeslaT4`.

### One GPU / CPU fallback

Check `result.json` -> `stats.gpu_count`.  A value of 0 means the kernel ran on
CPU; check the Kaggle account verification and the accelerator setting.  The
worker requests `NvidiaTeslaT4` both in metadata and on the CLI push.

---

## 9. CLI reference

```bash
python -m kaggle_bridge.doctor
python -m kaggle_bridge.setup --create-venv
python -m kaggle_bridge.setup --write-env
python -m kaggle_bridge.setup --build-sam2-wheel
python -m kaggle_bridge sync-weights
python -m kaggle_bridge jobs
python -m kaggle_bridge cancel --job <job-id>
python -m kaggle_bridge session-cleanup
python -m kaggle_bridge status --job <job-id> --log
python -m kaggle_bridge resume --job <job-id>
python -m kaggle_bridge import --job <job-id>
python -m kaggle_bridge cleanup --job <job-id> --yes
python -m kaggle_bridge dry-run --spec job-spec.json
python -m kaggle_bridge run --spec job-spec.json
```

`dry-run` builds the bundle without contacting Kaggle; useful for checking
paths, sizes and model/checkpoint availability.

---

## 10. Tests

```bash
# helper environment (Python 3.10 is fine)
python -m unittest discover -s tests -v
```

The suite covers metadata/T4 x2, bundle contents, label merge policy, path
traversal/class validation, job store locking and kernel extraction safety.
Live Kaggle tests were run separately against `dukeshavrao` and are recorded
in section 7.

---

## 11. Known limitations

- Local jobs are serialized; only one remote job at a time.
- Progress is polling-based; there is no live per-frame stream from Kaggle.
- There is no remote-cancel button yet; deleting the kernel on Kaggle stops it.
- Long videos are not automatically chunked.  Split very long clips or use a
  frame range; SAM 2 memory resets per kernel job.
- The first session upload copies/compresses the whole current images folder.
  Large batches take time and local disk; this is one-time per app session.
- The fixed kernel keeps one version per click; old versions remain in Kaggle
  history.  Delete the kernel manually to reset the history.
- If LabelImg is force-killed, the detached `session-cleanup` may not run.
  Reopening the same batch reuses the session (and deletes it on a normal
  close), or run `python -m kaggle_bridge session-cleanup --stale-hours 24`
  manually.
- When a new model/checkpoint is added, the weights dataset version re-uploads
  the staged weight files (Kaggle datasets are snapshots).  Updates are
  detected by size/mtime.
- SAM 3 is not part of this bridge; the same job protocol can add it later.
