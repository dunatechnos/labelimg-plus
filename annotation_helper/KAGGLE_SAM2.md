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
        ├─ 1. pack images + current labels + model into job.zip
        ├─ 2. create a private Kaggle dataset (Kaggle auto-extracts the zip)
        ├─ 3. push a private script kernel on NvidiaTeslaT4 (2x T4)
        ├─ 4. poll status every 30 s
        ├─ 5. download labels.json + result.json
        └─ 6. merge into local labels/ and push an Undo snapshot
```

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

### 2.3 Verify

```bash
python -m kaggle_bridge.doctor
```

Expected:

```
Authenticated: yes (your-user)
GPU quota:    30.00h remaining of 30.00h
Everything looks ready.
```

### 2.4 Use it from the helper

1. Open Annotation Helper on a batch (`images/` + `labels/`).
2. Tick **Run heavy work on Kaggle (T4 x2)** in the **Compute** card.
3. For predict: choose a YOLO model, then **Predict on Kaggle**.
4. For tracking: select one or more boxes, choose **Propagate > Dynamic
   SAM 2**, then **Start SAM 2 on Kaggle**.
5. The job uploads, runs, downloads and merges automatically.  `Ctrl+Z`
   restores every label file changed by the import.

The **Kaggle setup** button opens the token / CLI / connection-test dialog.
The **Jobs...** button opens local job history and a **Resume selected**
button for interrupted jobs.

---

## 3. What gets uploaded

`job.zip` contains:

```
manifest.json          job id, classes, frame mapping, existing labels
runner_lib.py          label merge helpers (shared with the kernel)
images/000000.jpg      sequential JPEG frames
images/...
model/sam2.1_hiera_small.pt   optional (SAM 2)
model/your_yolo.pt            YOLO predict
```

Ultralytics/SAM 2 packages and the SAM 2 checkpoint are only uploaded when
needed.  If a local SAM 2 checkpoint exists, it is included; otherwise the
kernel downloads the official Meta checkpoint (about 176 MB).  No Kaggle
credential is ever put in the bundle.

Kaggle auto-extracts the zip inside the private dataset.  The runner supports
both layouts: an extracted dataset root (`manifest.json` + `images/`) and a
real `job.zip` for local tests.

### Kernel output

```
output/labels.json     {original_image_name: [[cls, xc, yc, w, h], ...]}
output/result.json     status, counts, kernel stats
output/status.json     last progress marker
```

Only changed label files are returned.  The local importer validates:
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
- First SAM 2 run is slower because the kernel installs SAM 2 and downloads
  the checkpoint (about 3-4 minutes on a warm Kaggle image).  YOLO predict is
  about 20-30 seconds after the kernel starts.

---

## 7. Verified live results

Validated on 2026-11-13 with Kaggle user `dukeshavrao`:

| Test | Result |
|---|---|
| YOLO `yolov8n` on 1 bus image | `status=ok`, 5 boxes, 1 label file merged, `gpu_count=2`, 21.1 s kernel |
| SAM 2.1 Small on 5 synthetic frames | 1 object, 5/5 frames tracked, 5 label files merged, `device=cuda`, `gpu_count=2`, 203.5 s including install/checkpoint |
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

### `ModuleNotFoundError: runner_lib`

Kaggle script kernels only upload a single `code_file`.  `runner_lib.py` is
inside `job.zip`; the runner imports it after extraction.  Do not move it back
to the kernel folder.

### `403 Forbidden` while checking a new dataset

`datasets status` returns 403 for a dataset that does not exist yet.  The
worker treats that as "not found" and proceeds to create it.  This is expected
and only appears in the log.

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
python -m kaggle_bridge jobs
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
- The first SAM 2 run pays the install/checkpoint time.  A future optimization
  is a shared private weights/source dataset.
- SAM 3 is not part of this bridge; the same job protocol can add it later.
