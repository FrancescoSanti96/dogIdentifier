# 🐕 Dog Breed Identifier

AI University exam

## 🎯 Project Overview

This project implements a two-phase dog identification system:

1. **Breed Classifier**: Multi-class classification (started with 120+ breeds, optimized to 5→10 breeds)
2. **Personal Dog Identifier**: Binary classification for personal Australian Shepherd recognition

## 📦 Project Structure

```
dogIdentifier/
├── 📄 README.md                    # Project documentation
├── 📄 requirements.txt             # Dependencies

│
├── 📁 src/                         # Main source code
│   ├── 📄 train.py                 # Unified training script
│   ├── 📄 prepare_data.py          # Unified data preparation
│   ├── 📄 evaluate.py              # Model evaluation and confusion matrix
│   └── 📄 my_dog_train.py          # Phase 2: Personal dog training
│
├── 📁 scripts/                     # Utility scripts
│   └── 📄 launch_tensorboard.py    # TensorBoard launcher
│
├── 📁 models/                      # Neural network architectures
├── 📁 utils/                       # Helper functions
├── 📁 docs/                        # Documentation
│   └── 📄 PROCESSO.md              # Complete experimental process
│
├── 📁 data/                        # Datasets (5, 10, 30, 60, 90, 121 breeds)
├── 📁 outputs/                     # Models and analysis results
├── 📁 experiments/                 # Research experiments
└── 📁 tests/                       # Validation tests
```

## 🚀 Core Scripts

### **🎯 Unified Training (Recommended)**

```bash
python src/train.py --breeds {5,10,30,60,90,121}
```

### **🔧 Unified Data Preparation**

```bash
python src/prepare_data.py --breeds {10,30,121}
```

### **📊 Unified Model Evaluation**

```bash
python src/evaluate.py --model MODEL --data DATA --outdir OUTPUT
```

### **🐕 Personal Dog Training (Phase 2)**

```bash
python src/my_dog_train.py
```

### **📈 TensorBoard Monitoring**

```bash
python scripts/launch_tensorboard.py
```

---

## 📋 **Legacy Information**

This project previously used individual training scripts for each breed scale. These have been consolidated into unified scripts in the `src/` directory for better maintainability.

**Legacy scripts are preserved in:**

- `experiments/legacy_scripts/training/` - Individual training scripts (`quick*_tensorboard_train.py`)
- `experiments/legacy_scripts/preparation/` - Individual preparation scripts (`prepare_*.py`)
- `experiments/archive/` - Historical experiments and development code

See `experiments/legacy_scripts/README.md` for migration guide and detailed information.

## 🚀 **How to Use This Project**

### **📋 Quick Start Guide**

#### **1️⃣ Setup Environment**

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python src/train.py --help
```

#### **2️⃣ Training Models (Unified Scripts - Recommended)**

```bash
# 🎯 COMPLETE WORKFLOW EXAMPLES

# 5 breeds (baseline - dataset pre-prepared)
python src/train.py --breeds 5

# 10 breeds (prepare dataset first)
python src/prepare_data.py --breeds 10
python src/train.py --breeds 10

# 30 breeds (prepare + train)
python src/prepare_data.py --breeds 30
python src/train.py --breeds 30

# 60 breeds (dataset pre-prepared)
python src/train.py --breeds 60

# 90 breeds (dataset pre-prepared)
python src/train.py --breeds 90

# 121 breeds (full dataset)
python src/prepare_data.py --breeds 121
python src/train.py --breeds 121
```

#### **3️⃣ Model Evaluation**

```bash
# Evaluate any trained model
python src/evaluate.py \
  --model outputs/breeds_121/best_model.pth \
  --data data/full121_balanced \
  --batch-size 64 \
  --outdir outputs/analysis/evaluation_$(date +%Y%m%d_%H%M%S)
```

#### **4️⃣ Personal Dog Training (Phase 2)**

```bash
# Binary classification: your dog vs others
python src/my_dog_train.py
```

#### **5️⃣ Monitoring with TensorBoard**

```bash
# Launch TensorBoard dashboard
python scripts/launch_tensorboard.py
# Open: http://localhost:6006
```

### **⚙️ Advanced Configuration**

#### **Environment Variables Override**

```bash
# Customize training parameters
USE_TL=1 EPOCHS=45 PATIENCE=10 LR=0.0008 python src/train.py --breeds 121
BATCH_SIZE=64 DROPOUT=0.3 python src/train.py --breeds 30
```

#### **Available Parameters**

| Variable     | Description       | Default | Example         |
| ------------ | ----------------- | ------- | --------------- |
| `USE_TL`     | Transfer Learning | 1       | `USE_TL=1`      |
| `EPOCHS`     | Training epochs   | Auto    | `EPOCHS=45`     |
| `BATCH_SIZE` | Batch size        | 32      | `BATCH_SIZE=64` |
| `LR`         | Learning rate     | Auto    | `LR=0.0008`     |
| `PATIENCE`   | Early stopping    | Auto    | `PATIENCE=10`   |
| `DROPOUT`    | Dropout rate      | 0.4     | `DROPOUT=0.3`   |

### TensorBoard

```bash
python scripts/launch_tensorboard.py
# or manually: tensorboard --logdir outputs/tensorboard
# apri http://localhost:6006
```

## 📊 How to evaluate

### **🎯 Unified Evaluation (Recommended)**

```bash
# Evaluate any trained model
python src/evaluate.py \
  --model outputs/breeds_121/best_model.pth \
  --data data/full121_balanced \
  --batch-size 64 \
  --outdir outputs/analysis/breeds_121_$(date +%Y%m%d_%H%M%S)
```

**Examples for different scales:**

```bash
# 5 breeds
python src/evaluate.py --model outputs/breeds_5/best_model.pth --data data/breeds_5

# 90 breeds
python src/evaluate.py --model outputs/breeds_90/best_model.pth --data data/top90_balanced

# 121 breeds
python src/evaluate.py --model outputs/breeds_121/best_model.pth --data data/full121_balanced
```

### **🐕 Phase 2: Personal Dog Training**

```bash
# Binary classification: your dog vs others
python src/my_dog_train.py
```

### **🔄 Legacy Scripts (Alternative)**

If you prefer the original individual scripts, they're preserved and functional:

```bash
# 🗃️ LEGACY TRAINING (from project root)
python experiments/legacy_scripts/training/quick121_tensorboard_train.py
python experiments/legacy_scripts/training/quick90_tensorboard_train.py
python experiments/legacy_scripts/training/quick60_tensorboard_train.py

# 🗃️ LEGACY PREPARATION (from project root)
python experiments/legacy_scripts/preparation/prepare_full121_balanced.py
python experiments/legacy_scripts/preparation/prepare_top30_balanced.py
python experiments/legacy_scripts/preparation/prepare_top10_balanced.py
```

**⚠️ Important:** Legacy scripts must be run from the project root directory.

### **📊 Results and Outputs**

After training, you'll find:

#### **🎯 For Final Results (University Submission):**

- **📁 `outputs/results/`**: Complete project deliverables
  - **`best_models/breeds_121_best.pth`**: Final production model
  - **`project_summary.md`**: Complete project overview
  - **`performance_table.csv`**: All model performance data
  - **`final_analysis/`**: Confusion matrices and metrics

#### **🔧 For Development:**

- **📁 `outputs/models/breeds_N/`**: All models (best + final) by scale
- **📁 `outputs/tensorboard/breeds_N/`**: TensorBoard logs organized by scale
- **📁 `outputs/analysis/`**: Detailed analysis with timestamps
- **📁 `outputs/archive/`**: Historical experiments

### **🧪 Example Complete Workflow**

```bash
# Complete example: 30 breeds from scratch to evaluation
echo "🚀 Starting 30 breeds workflow..."

# 1. Prepare dataset
python src/prepare_data.py --breeds 30

# 2. Train model
python src/train.py --breeds 30

# 3. Evaluate model
python src/evaluate.py \
  --model outputs/models/breeds_30/best_model.pth \
  --data data/top30_balanced \
  --outdir outputs/analysis/breeds_30_final

# 4. Monitor with TensorBoard
python scripts/launch_tensorboard.py
```

## 🔧 **Troubleshooting**

### **Common Issues**

#### **ImportError with Legacy Scripts**

```bash
# ❌ Error: ModuleNotFoundError: No module named 'utils'
# ✅ Solution: Always run legacy scripts from project root
cd /path/to/dogIdentifier
python experiments/legacy_scripts/training/quick121_tensorboard_train.py
```

#### **Dataset Not Found**

```bash
# ❌ Error: FileNotFoundError: data/top30_balanced
# ✅ Solution: Prepare dataset first
python src/prepare_data.py --breeds 30
```

#### **CUDA/GPU Issues**

```bash
# The project works on both CPU and GPU
# GPU will be used automatically if available
# For CPU-only training, no special configuration needed
```

#### **TensorBoard Not Loading**

```bash
# If TensorBoard doesn't start automatically:
tensorboard --logdir outputs/tensorboard --port 6006
# Then open: http://localhost:6006
```

### **File Locations**

- **Trained Models**: `outputs/breeds_N/best_model.pth`
- **Training Logs**: `outputs/tensorboard/breeds_N_TIMESTAMP/`
- **Evaluation Results**: `outputs/analysis/breeds_N_TIMESTAMP/`
- **Dataset Stats**: `data/DATASET_NAME/dataset_stats.json`

### **Performance Tips**

- **Use GPU**: Automatic if CUDA available
- **Batch Size**: Increase if you have more memory (`BATCH_SIZE=64`)
- **Early Stopping**: Adjust patience for longer training (`PATIENCE=15`)
- **Transfer Learning**: Always enabled by default (`USE_TL=1`)

### **📈 Monitor Training**

```bash
# Launch TensorBoard to monitor training progress
python scripts/launch_tensorboard.py
# Open: http://localhost:6006
```

## 📊 **Project Results**

### **Model Performance Summary**

| **Scale** | **Breeds** | **Validation Accuracy** | **Top-5 Accuracy** | **Status**      |
| --------- | ---------- | ----------------------- | ------------------ | --------------- |
| Baseline  | 5          | ~66.2%                  | ~95%               | ✅ Complete     |
| Small     | 10         | ~75%                    | ~97%               | ✅ Complete     |
| Medium    | 30         | ~78%                    | ~97%               | ✅ Complete     |
| Large     | 60         | ~79%                    | ~97%               | ✅ Complete     |
| XL        | 90         | ~81.4%                  | ~98%               | ✅ Complete     |
| **Full**  | **121**    | **78.83%**              | **~97%**           | ✅ **Complete** |

### **Key Achievements**

- **🎯 Progressive Scaling**: Successfully scaled from 5 to 121 dog breeds
- **🏆 High Accuracy**: 78.83% validation accuracy on 121 breeds
- **⚡ Transfer Learning**: Effective use of pre-trained ResNet18
- **📊 Comprehensive Analysis**: Detailed confusion matrices and per-class metrics
- **🔧 Production Ready**: Clean, unified codebase with full documentation

### **Best Performing Model**

**121 Breeds Model** (`outputs/breeds_121/best_model.pth`):

- **Test Accuracy**: 77.2%
- **Validation Accuracy**: 78.83%
- **Top-5 Accuracy**: ~97%
- **Australian Shepherd**: 100% accuracy (project target)

For detailed experimental process and results, see `docs/PROCESSO.md`.
