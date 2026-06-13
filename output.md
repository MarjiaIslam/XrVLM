# xrVLM Project Output Summary

## Overview

xrVLM is a two-stage chest X-ray analysis pipeline. The main idea is to use a fast CNN model first, and only call a heavier medical vision-language model when the CNN is uncertain.

The pipeline works like this:

1. A chest X-ray image is loaded and preprocessed.
2. Stage 1 uses EfficientNet-B4 to predict probabilities for 14 chest X-ray findings.
3. The probabilities are calibrated using temperature scaling.
4. If Stage 1 is confident enough, the system accepts its result directly.
5. If Stage 1 is uncertain, the image is routed to Stage 2.
6. Stage 2 tries to use CheXagent as a medical VLM. If the VLM cannot be loaded, the code uses a rule-based fallback.
7. The reconciliation layer combines or compares the two stages.
8. Conflicting cases are flagged for radiologist review.
9. Radiologist corrections can be stored for later calibration updates.

This project is for research and educational use only. It is not a medical device.

---

## Actual Project Structure

### `config.py`

Contains the main configuration for the whole project.

Important settings:

| Setting | Value in code | Meaning |
|---|---:|---|
| `NUM_CLASSES` | 14 | Number of disease labels |
| `IMAGE_SIZE` | 380 | Input image size for EfficientNet-B4 |
| `STAGE1_BACKBONE` | `efficientnet_b4` | CNN backbone |
| `STAGE1_DROPOUT` | 0.4 | Dropout before classifier |
| `BATCH_SIZE` | 16 | Default batch size |
| `ROUTING_THRESHOLD` | 0.95 | Confidence threshold for skipping Stage 2 |
| `VLM_MODEL_ID` | `StanfordAIMI/CheXagent-2-3b` | Default VLM model |
| `RECON_WEIGHT_STAGE1` | 0.4 | Stage 1 fusion weight |
| `RECON_WEIGHT_STAGE2` | 0.6 | Stage 2 fusion weight |
| `CONFLICT_THRESHOLD` | 0.3 | Stored conflict threshold value |

The project predicts 14 NIH ChestXray14 disease labels, not a separate `"No Finding"` class.

---

## Disease Labels

The code uses these 14 labels:

```text
Atelectasis
Cardiomegaly
Effusion
Infiltration
Mass
Nodule
Pneumonia
Pneumothorax
Consolidation
Edema
Emphysema
Fibrosis
Pleural_Thickening
Hernia
```

Each image can have multiple labels, so this is a multi-label classification problem.

---

## Data Loading: `dataset.py`

`dataset.py` prepares the chest X-ray dataset for training and testing.

It does the following:

- Reads the NIH-style metadata CSV, usually `Data_Entry_2017.csv`.
- Converts the `Finding Labels` column into 14 binary disease columns.
- Finds image files from NIH-style folders or a flat image directory.
- Uses official train/test split files if available.
- Otherwise, creates a random 80/20 split.
- Splits validation data at patient level when `Patient ID` is available.
- Computes class imbalance weights for training.

Image transforms:

- Training uses resize, random resized crop, small rotation, and brightness/contrast jitter.
- Validation and test use resize and center crop.
- The code intentionally avoids horizontal flipping because left/right anatomy matters in chest X-rays.

---

## Stage 1 Model: `stage1_model.py`

Stage 1 is the fast CNN classifier.

The model uses:

```text
EfficientNet-B4 backbone
Dropout(0.4)
Linear(1792 -> 14)
TemperatureScaler
```

The model outputs 14 logits, one for each disease label. These logits are passed through sigmoid to get probabilities.

Temperature scaling is included to make the confidence scores more reliable. It learns one scalar value `T` and divides logits by `T` before sigmoid.

---

## Training: `train.py`

`train.py` trains the Stage 1 CNN.

Training happens in two phases:

1. **Head warm-up**
   - Backbone is frozen.
   - Only the classifier head and temperature parameter are trained.
   - Default: 3 epochs.

2. **Full fine-tuning**
   - Backbone is unfrozen.
   - The whole model is trained.
   - Default: 12 epochs.

The training loss is:

```text
BCEWithLogitsLoss(pos_weight=...)
```

This is suitable for multi-label classification and helps with class imbalance.

After training, temperature scaling is tuned on the validation set using L-BFGS. The best calibrated model is saved under:

```text
outputs/checkpoints/best_model_calibrated.pth
```

Training logs are saved to:

```text
outputs/loss_log.csv
```

---

## Routing Logic: `pipeline.py`

The main inference class is:

```python
DiagnosticPipeline
```

The main method is:

```python
analyze(...)
```

The routing rule is based on the maximum calibrated probability:

```text
if max(probabilities) >= ROUTING_THRESHOLD:
    accept Stage 1 result
else:
    send to Stage 2
```

By default:

```text
ROUTING_THRESHOLD = 0.95
```

So if the CNN is very confident, Stage 2 is skipped. If the CNN is uncertain, Stage 2 is used.

---

## Stage 2 VLM: `stage2_vlm.py`

Stage 2 tries to use CheXagent:

```text
StanfordAIMI/CheXagent-2-3b
```

It uses two prompt templates and asks for JSON output.

The expected Stage 2 output schema is:

```json
{
  "finding": "<primary finding or Normal>",
  "severity": "<normal|mild|moderate|severe>",
  "confidence": 0.0,
  "rationale": "<brief clinical reasoning>",
  "secondary_findings": []
}
```

If CheXagent cannot be loaded, the code uses a rule-based fallback. The fallback uses Stage 1 probabilities to create a structured response.

---

## Reconciliation: `reconciliation.py`

The reconciliation layer creates the final result.

There are three main cases:

### 1. Stage 1 only

If Stage 1 is confident, the final result is simply the top Stage 1 finding.

### 2. Stage 1 and Stage 2 agree

If both stages return the same finding, the final confidence is a weighted combination:

```text
final_confidence = 0.4 * Stage1_confidence + 0.6 * Stage2_confidence
```

### 3. Stage 1 and Stage 2 disagree

If the stages disagree, the result is marked as a conflict and flagged for radiologist review.

The output includes:

- final finding
- confidence
- severity
- explanation
- secondary findings
- conflict flag
- radiologist review flag

---

## Feedback Store: `feedback_store.py`

The feedback store saves radiologist corrections in JSON-lines format.

The default path is:

```text
outputs/feedback/corrections.json
```

Each correction can store:

- image ID
- pipeline prediction
- pipeline confidence
- radiologist finding
- radiologist severity
- whether the case was routed to Stage 2
- whether a conflict was detected
- optional Stage 1 logits
- optional true labels

After enough corrections are collected, the pipeline can update the temperature value using stored feedback data. This recalibrates confidence without retraining the full CNN.

---

## Demo And Data Scripts

### `download_sample.py`

Streams a small number of NIH ChestXray14 samples from Hugging Face and saves them locally.

It creates:

```text
data/raw/
Data_Entry_2017.csv
train_val_list.txt
test_list.txt
```

### `run_sample.py`

Runs an end-to-end demo:

1. Download or verify data.
2. Train Stage 1.
3. Calibrate temperature.
4. Run the full pipeline on test images.
5. Simulate radiologist feedback.
6. Save results to:

```text
outputs/sample_results.json
```

Note: Some comments in `run_sample.py` mention Kaggle, but the current `download_sample.py` is based on Hugging Face streaming.

---

## Example Usage

### Train Stage 1

```bash
python train.py --sample 100
```

### Run Pipeline On One Image

```bash
python pipeline.py path/to/xray.png
```

### Run Pipeline Without VLM

```bash
python pipeline.py path/to/xray.png --no-vlm
```

This uses the rule-based fallback instead of trying to load CheXagent.

---

## Important Limitations

- The project is a research prototype, not a clinical tool.
- The model predicts 14 NIH disease labels only.
- It does not currently model `"No Finding"` as a separate output class.
- CheXagent may require a large download and GPU memory.
- If the VLM is unavailable, Stage 2 becomes rule-based.
- ECE is discussed as a useful calibration metric, but the current code does not explicitly compute ECE.
- Conflict threshold is defined in configuration, but the current reconciliation mainly flags conflicts when Stage 1 and Stage 2 findings differ.

---

## Short Summary

This project builds a two-stage chest X-ray analysis system. A fast EfficientNet-B4 CNN first predicts disease probabilities. Temperature scaling makes the confidence scores more reliable. Confident cases are accepted directly, while uncertain cases are sent to CheXagent or a fallback Stage 2 system. The reconciliation layer combines agreement, flags disagreement, and stores feedback for future calibration.
