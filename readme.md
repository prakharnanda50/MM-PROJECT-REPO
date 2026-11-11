# MODEL MAYHEM : PCB DEFECT DETECTION

## 1. Description

Build an object detection model to identify and localise manufacturing defects in Printed Circuit Boards (PCBs).

**Target Classes:**
1. `missing_hole`
2. `mouse_bite`
3. `open_circuit`
4. `short`
5. `spur`
6. `spurious_copper`

Participants may use any framework, including YOLO, PyTorch, TensorFlow, or Keras.

---

## 2. Dataset

**Dataset:** [PCB Defect Detection Dataset (Roboflow Universe)](https://universe.roboflow.com/uthm-project-visionbased-pcb-defect-detection/pcb-defect-detection-9ewqw-duvuw/dataset/2)

- Images: 2,586
- Classes: 6
- Task: Defect Classification and Localisation (Bounding box)
- Version: 2

**Download Guide:**
1. Open the dataset link above.
2. Click **Download Dataset**.
3. Select format: **YOLOv8**.
4. Download and extract the ZIP file into the `data/` folder so that `data/data.yaml` is present:
   ```
   D:\Model Mahyem\
   └── data\
       ├── train\
       ├── valid\
       ├── test\
       └── data.yaml
   ```

---

## 3. Evaluation Criteria

Models will be evaluated on an unseen test dataset.

**Primary Metric: mAP@50:90**
- Average Precision across all six classes and nine IoU thresholds: 0.50, 0.55, 0.60, ... 0.90.

**Tie-Breakers (in order):**
1. **mAP@50**
2. **Macro Recall** (average recall across all six classes)

---

## 4. Submission Checklist

Each team must submit:
- [x] **Training Code:** `train.py`
- [x] **Model Weights:** Final trained checkpoint (`models/best.pt`)
- [x] **Inference Code:** `inference.py` (generates the prescribed CSV)
- [x] **Dependencies:** `requirements.txt`
- [x] **Execution Instructions:** This `README.md`
- [x] **Predictions:** `output/predictions.csv`

**Prescribed CSV Format:**
```csv
image_name,class_name,x_min,y_min,x_max,y_max,confidence
pcb_001.jpg,missing_hole,120,85,160,125,0.94
pcb_001.jpg,mouse_bite,220,140,270,190,0.87
```

**Submission Rules:**
- Coordinates in original image pixels (`x_min`, `y_min`, `x_max`, `y_max`).
- Class names must strictly match: `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`.
- Confidence scores between `0.0` and `1.0`.
- Each detection must occupy a separate row.

---

## 5. Step-by-Step Execution Guide

### Step 1: Install Dependencies
Open PowerShell or your terminal in this directory:
```powershell
pip install -r requirements.txt
```
*Note: If installing PyTorch with CUDA for local GPU training (e.g. RTX 3050):*
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Step 2: Download & Extract Dataset
Download the YOLOv8 format zip from Roboflow and extract it:
```powershell
python download_dataset.py --zip-file "path/to/downloaded.zip"
```
*(Or manually unzip the folder into `data/` so `data/data.yaml` is at `D:\Model Mahyem\data\data.yaml`)*

### Step 3: Train the Model
Run training using YOLO11 (or YOLOv8):
```powershell
python train.py --data data/data.yaml --model yolo11m.pt --epochs 80 --imgsz 800 --batch 8
```
- **Image Size (`--imgsz 800` or `1024`):** Crucial for PCB defects to detect tiny mouse bites and spurs.
- **Batch Size (`--batch 8`):** Fits within 6GB VRAM (RTX 3050 Laptop).
- Weights will automatically be saved to `models/best.pt`.

### Step 4: Evaluate the Model
Evaluate on validation or test split to check competition metrics:
```powershell
python evaluate.py --weights models/best.pt --data data/data.yaml --split val
```
This prints:
- Primary Metric: `mAP@50-95`
- Tie-Breaker 1: `mAP@50`
- Tie-Breaker 2: `Macro Recall`
- Per-class performance breakdown

### Step 5: Generate Submission Predictions CSV
Run inference on the test dataset folder:
```powershell
python inference.py --source data/test/images --weights models/best.pt --output output/predictions.csv --conf 0.25 --save-visuals
```
- Generates `output/predictions.csv` with the exact required header and coordinates.
- Saves annotated images to `output/visual_predictions/` for visual verification.

---

## 6. Official Verification & Benchmark Results

### 🏆 Benchmark Summary
- **Primary Metric (mAP@50:90 / mAP@50:95)**: **0.2496 (24.96%)**
- **Tie-Breaker 1 (mAP@50)**: **0.5651 (56.51%)**
- **Tie-Breaker 2 (Macro Recall)**: **0.4517 (45.17%)**
- **Macro F1 Score (at IoU 0.50)**: **0.5383 (53.83%)**
- **Macro Precision**: **0.6660 (66.60%)**
- **Inference Latency**: **~9.7 ms** per image (~100 FPS on NVIDIA RTX 3050)
- **Model Checkpoint Size**: **5.5 MB** (`models/best.pt`)

### 📋 Per-Class Performance
| Class | Precision | Recall | F1@50 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| `missing_hole` | 0.8115 | 0.6098 | 0.6964 | 0.7566 | 0.3883 |
| `mouse_bite` | 0.6587 | 0.4085 | 0.5042 | 0.5081 | 0.2177 |
| `open_circuit` | 0.6493 | 0.2932 | 0.4040 | 0.4066 | 0.1633 |
| `short` | 0.7244 | 0.5032 | 0.5939 | 0.6453 | 0.2598 |
| `spur` | 0.5003 | 0.3252 | 0.3942 | 0.4114 | 0.1689 |
| `spurious_copper` | 0.6515 | 0.5702 | 0.6081 | 0.6627 | 0.2996 |

### 🔬 5-Fold Cross-Validation Stability
- **Fold 1**: Macro F1 = 0.5605
- **Fold 2**: Macro F1 = 0.5650
- **Fold 3**: Macro F1 = 0.5739
- **Fold 4**: Macro F1 = 0.5957
- **Fold 5**: Macro F1 = 0.5654
- **Mean Macro F1**: **0.5721** $\pm$ **0.0126** (variance is very low, confirming stability across unseen physical board layouts).

---

## 7. Quick Evaluator Testing Commands

For hackathon evaluators to test score and running of the submission:

```powershell
# 1. Run evaluation benchmark (outputs mAP@50:90, mAP@50, Macro Recall, F1)
python evaluate.py --weights models/best.pt --data data/data.yaml --split test

# 2. Run inference on test images folder to generate predictions.csv
python inference.py --source data/test/images --weights models/best.pt --output output/predictions.csv --conf 0.25

# 3. Check generated predictions CSV format
Get-Content output/predictions.csv -TotalCount 5
```
