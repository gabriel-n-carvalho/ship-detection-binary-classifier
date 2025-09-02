# 🚢 Airbus Ship Detection - Binary Classifier

A production-ready, deterministic binary classifier for detecting ships in satellite imagery using the Airbus Ship Detection Challenge dataset. Built with PyTorch and optimized for CPU-only training in Docker containers.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training Details](#training-details)
- [Data Processing](#data-processing)
- [Model Architecture](#model-architecture)
- [Reproducibility](#reproducibility)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project converts the complex Airbus Ship Detection segmentation challenge into a simpler binary classification problem: **ship vs. no-ship**. It's designed for production environments where GPU resources aren't available, using advanced CPU optimization techniques and deterministic training for full reproducibility.

**Key Use Cases:**

- Maritime surveillance and monitoring
- Satellite image analysis
- Ship detection in remote sensing data
- Educational/research purposes in computer vision

## ✨ Features

### 🚀 **Core Capabilities**

- **Binary Classification**: Ship detection (1) vs. no ship (0)
- **Deterministic Training**: Full reproducibility across runs
- **Early Stopping**: Based on PR-AUC with configurable patience
- **Checkpoint Resuming**: Resume training from any checkpoint
- **CPU Optimization**: ARM64 optimized with threading optimizations

### 🔧 **Technical Features**

- **Docker Containerization**: Production-ready deployment
- **Data Augmentation**: Training-time transformations for robustness
- **Class Imbalance Handling**: Automatic positive weight calculation
- **Stratified Sampling**: Maintains class balance in data subsets
- **Comprehensive Logging**: Real-time progress and metrics
- **Resource Management**: Configurable memory and CPU limits

### 📊 **Metrics & Monitoring**

- **Training Metrics**: Loss, learning rate, batch timing
- **Validation Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- **Progress Tracking**: ETA estimates, real-time progress bars
- **Checkpoint Management**: Best and last model saving

## 📁 Project Structure

```
final-project/
├── airbus-ship-detection/          # Dataset directory (You need to download the dataset from Kaggle)
│   ├── train_v2/                   # Training images
├── labels/                         # Label files and data splits
│   ├── binary_labels.csv           # Binary ship/no-ship labels
│   ├── segmentations_labels.csv    # Original segmentation data (This file comes with the dataset from Kaggle)
│   └── splits/                     # Train/validation splits
│       ├── train.csv               # Training set
│       └── val.csv                 # Validation set
├── utils/                          # Data preprocessing utilities
│   ├── make_binary_labels.py       # Convert segmentation to binary
│   └── make_train_val_split.py     # Create train/val splits
├── outputs/                        # Model outputs and checkpoints
│   └── models/                     # Saved model checkpoints
├── docker-compose.yml              # Container orchestration
├── Dockerfile                      # Container definition
├── requirements.txt                # Python dependencies
├── train_binary_classifier.py      # Main training script
└── README.md                       # This file
```

## 📋 Prerequisites

### **System Requirements**

- **OS**: Linux, macOS, or Windows with Docker
- **RAM**: Minimum 16GB, recommended 64GB+
- **CPU**: Multi-core processor (ARM64 or x86_64)
- **Storage**: 10GB+ free space for dataset and models

### **Software Requirements**

- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **Git**: For cloning the repository

### **Dataset**

- **Airbus Ship Detection Challenge**: Download from [Kaggle](https://www.kaggle.com/c/airbus-ship-detection)

## 🚀 Run the training

### **1. Clone and Setup**

```bash
git clone <your-repo-url>
cd ship-detection-binary-classifier
```

### **2. Download the dataset**

- Download the dataset from Kaggle
- Copy the `train_v2/` folder from the `airbus-ship-detection` dataset into the root directory of your project, so your folder structure matches the example above.
- Place the segmentation labels file `train_ship_segmentations_v2.csv` into `labels/train_ship_segmentations_v2.csv`.
- Run the following commands to convert the segmentation labels to binary labels and create a stratified split of the data into train and val sets.

### **3. Prepare Data**

```bash
# Convert segmentation labels to binary (if needed). This script converts the original segmentation labels to binary labels.
python utils/make_binary_labels.py

# Create label splits (if not already done). This script creates a stratified split of the data into train and val sets.
python utils/make_train_val_split.py
```

### **4. Start Training with Docker**

```bash
# This command will start the training process and you will see the progress bars in the foreground.
docker-compose run --rm efficientnet-training

```

**Note:** Using `docker-compose up` can cause tqdm progress bars to be buffered and only display after training completes, due to how output is handled ([see tqdm issue #771](https://github.com/tqdm/tqdm/issues/771)). For real-time progress updates, it is recommended to use `docker-compose run` instead.

### **5. Monitor Training**

```bash
# With `docker-compose run`, you will see live tqdm progress in the foreground.

# If you still choose to use `up`, logs can be tailed but tqdm may not update live:
docker-compose logs -f

# Check container status
docker-compose ps
```

## 📖 Usage

### **Command Line Interface**

The training script supports extensive command-line configuration:

```bash
python train_binary_classifier.py \
    --seed 42 \
    --fold 0 \
    --batch-size 32 \
    --epochs 100 \
    --lr 3e-4 \
    --img-size 256 \
    --grad-accum-steps 1 \
    --early-stop-patience 20 \
    --model-name tf_efficientnetv2_b0.in1k \
    --subset-size 1000
```

### **Docker Environment Variables**

Configure training parameters via environment variables:

```bash
# Override default settings
export SEED=123
export BATCH_SIZE=64
export EPOCHS=200
export LR=1e-4

# Start training
docker-compose up
```

### **Data Subsetting for Development**

Use data subsets for faster iteration during development:

```bash
# Use 10% of data
python train_binary_classifier.py --subset-fraction 0.1

# Use specific number of samples
python train_binary_classifier.py --subset-size 1000
```

## ⚙️ Configuration

### **Training Hyperparameters**

| Parameter             | Default | Description                     |
| --------------------- | ------- | ------------------------------- |
| `SEED`                | 42      | Random seed for reproducibility |
| `BATCH_SIZE`          | 32      | Training batch size             |
| `EPOCHS`              | 100     | Maximum training epochs         |
| `LR`                  | 3e-4    | Learning rate                   |
| `IMG_SIZE`            | 256     | Input image size                |
| `GRAD_ACCUM_STEPS`    | 1       | Gradient accumulation steps     |
| `EARLY_STOP_PATIENCE` | 20      | Early stopping patience         |

### **Model Architecture**

- **Base Model**: EfficientNet-V2-B0 (ImageNet pretrained)
- **Input Size**: 256x256 RGB images
- **Output**: Single sigmoid output (0-1 probability)
- **Loss Function**: BCEWithLogitsLoss with class balancing

### **Data Augmentation**

**Training Transforms:**

- Resize to 256x256
- Random horizontal flip (50%)
- Random vertical flip (20%)
- Random rotation (±5°)
- Color jittering
- ImageNet normalization

**Validation Transforms:**

- Resize to 256x256
- ImageNet normalization

## 🎓 Training Details

### **Training Loop**

1. **Epoch Initialization**: Set random seeds, create progress bars
2. **Training Phase**: Forward pass, loss calculation, backpropagation
3. **Validation Phase**: Model evaluation, metric computation
4. **Checkpointing**: Save best model (PR-AUC) and last checkpoint
5. **Early Stopping**: Monitor PR-AUC improvement

### **Loss Function**

```python
# Automatic class balancing
pos_weight = max(1.0, negative_samples / positive_samples)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### **Optimization**

- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine annealing learning rate
- **Gradient Clipping**: Norm clipping at 1.0
- **Gradient Accumulation**: Configurable for effective larger batches

### **Metrics**

- **Accuracy**: Overall classification accuracy
- **Precision**: Ship detection precision
- **Recall**: Ship detection recall
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve (primary metric)

## 🔄 Data Processing

### **Data Pipeline**

1. **Image Loading**: PIL-based RGB conversion with error handling
2. **Transformation**: PyTorch transforms with data augmentation
3. **Batching**: Deterministic DataLoader with worker seeding
4. **Device Transfer**: CPU-optimized data movement

### **Label Processing**

- **Binary Conversion**: Segmentation masks → binary labels
- **Stratified Splitting**: Maintains class balance in train/val sets
- **Type Stability**: Consistent float32 dtype for labels

### **Data Validation**

- **Class Balance Check**: Ensures both positive and negative samples
- **Image Loading**: Graceful handling of corrupted images
- **Label Consistency**: Validation of label format and values

## 🏗️ Model Architecture

### **EfficientNet-V2-B0**

- **Architecture**: Compound scaling with inverted residuals
- **Parameters**: ~7.1M trainable parameters
- **Input**: 256x256x3 RGB images
- **Output**: Single logit for binary classification
- **Pretraining**: ImageNet-1K weights

### **Model Modifications**

```python
model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=1  # Binary classification
)
```

## 🔒 Reproducibility

### **Deterministic Training**

- **Seed Control**: All random generators seeded consistently
- **RNG State**: Complete state saving/restoration for resuming
- **Worker Seeding**: Deterministic DataLoader worker initialization
- **Algorithm Selection**: PyTorch deterministic algorithms enforced

### **Checkpoint System**

- **Best Model**: Saved based on PR-AUC improvement
- **Last Checkpoint**: Always saved for resuming
- **RNG State**: Complete random state preservation
- **Metadata**: Model configuration and training history

## ⚡ Performance Optimization

### **CPU Optimization**

- **Threading**: Optimal OpenMP/MKL/BLAS thread configuration
- **Memory Management**: Conservative shared memory settings
- **Data Loading**: Single worker in Docker to avoid conflicts
- **Batch Processing**: Efficient tensor operations

### **Docker Optimizations**

- **Resource Limits**: Configurable memory and CPU allocation
- **Shared Memory**: 16GB shm_size for DataLoader stability
- **Volume Mounts**: Efficient data access and persistence
- **Environment Variables**: Optimized PyTorch settings

### **Memory Management**

- **Gradient Accumulation**: Effective larger batch sizes
- **Checkpoint Cleanup**: Automatic garbage collection
- **Tensor Optimization**: CPU-specific tensor operations
- **Shared Memory**: Docker container memory optimization

## 🐛 Troubleshooting

### **Common Issues**

#### **Out of Memory**

```bash
# Reduce batch size
export BATCH_SIZE=16

# Reduce image size
export IMG_SIZE=128

# Use data subsetting
--subset-size 500
```

#### **Slow Training**

```bash
# Increase batch size (if memory allows)
export BATCH_SIZE=64

# Reduce image size
export IMG_SIZE=128

# Use gradient accumulation
export GRAD_ACCUM_STEPS=2
```

#### **Docker Issues**

```bash
# Check container logs
docker-compose logs

# Restart container
docker-compose restart

# Check resource usage
docker stats
```

### **Debug Mode**

Enable verbose logging for debugging:

```bash
# Set environment variable
export PYTHONUNBUFFERED=1

# Run with debug output
python -u train_binary_classifier.py --subset-size 100
```

