# -*- coding: utf-8 -*-
"""
Reproducible Single-Eye ResNet18 Training (5-fold CV -> select best fold -> TEST + TFLite Re-eval)
-------------------------------------------------------------------------------
- FIXED: No data leakage (CV uses only original TRAIN data)
- Strict determinism: fixed seeds, cuDNN deterministic, no TF32, single-thread OpenCV
- Single image per patient
- Early stop if P & R >= 0.90 on validation
- TFLite conversion + re-evaluation on same test set with auto-detected input size
- Added detailed prediction CSV with file IDs, predictions, probabilities, and confusion matrix indicators
- Added comprehensive plotting: ROC curves, confusion matrices, metrics comparison

RIGHT_EYE_1 (Chronological Split)
"""

import os
import random
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from torch.cuda.amp import GradScaler, autocast
import warnings
warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
SEED = 42
NUM_WORKERS = 0           # safest for reproducibility
PIN_MEMORY = False
USE_AMP = True
SAVE_EVERY_FOLD_MODEL = True
N_SPLITS = 5
RESOLUTION = 780
EPOCHS_CV = 300
BATCH_CV = 14
LR_CV = 0.00003
EARLY_STOP_PR = 0.90      # stop training if P & R >= 0.90


BASE_PATH = "/home/ubuntu/anemia-storage/hb_mobilenet/mat_conjunctiva_all_consistent_deletion/"
DATA_DIR = "tri_right_eye/right_eye_1_hb_less_than_9_0/conjunctiva_extracted/"
OUTPUT_DIR = os.path.join(BASE_PATH, "RIGHT_EYE_1_eye_original_repro")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =========================
# DETERMINISM
# =========================
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

os.environ["PYTHONHASHSEED"] = str(SEED)
try:
    cv2.setNumThreads(0)
except Exception:
    pass

set_global_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# =========================
# PATHS
# =========================
dirs = {
    'anemic_train': os.path.join(BASE_PATH, DATA_DIR, "anemic_train_roi/"),
    'non_train': os.path.join(BASE_PATH, DATA_DIR, "anemic_not_train_roi/"),
    'anemic_val': os.path.join(BASE_PATH, DATA_DIR, "anemic_val_roi/"),
    'non_val': os.path.join(BASE_PATH, DATA_DIR, "anemic_not_val_roi/"),
    'anemic_test': os.path.join(BASE_PATH, DATA_DIR, "anemic_test_roi/"),
    'non_test': os.path.join(BASE_PATH, DATA_DIR, "anemic_not_test_roi/")
}

# =========================
# DATA LOADING WITH FILENAMES
# =========================
def load_images_with_filenames(folder, label):
    imgs, lbls, filenames = [], [], []
    if os.path.exists(folder):
        for f in sorted(os.listdir(folder)):
            if f.endswith(".png"):
                im = cv2.imread(os.path.join(folder, f))
                if im is not None:
                    imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    lbls.append(label)
                    filenames.append(f)
    return imgs, lbls, filenames

train_imgs, train_lbls, train_filenames = [], [], []
val_imgs, val_lbls, val_filenames = [], [], []
test_imgs, test_lbls, test_filenames = [], [], []

for folder, label in [(dirs['anemic_train'],1),(dirs['non_train'],0)]:
    i,l,f = load_images_with_filenames(folder,label)
    train_imgs+=i; train_lbls+=l; train_filenames+=f
for folder, label in [(dirs['anemic_val'],1),(dirs['non_val'],0)]:
    i,l,f = load_images_with_filenames(folder,label)
    val_imgs+=i; val_lbls+=l; val_filenames+=f
for folder, label in [(dirs['anemic_test'],1),(dirs['non_test'],0)]:
    i,l,f = load_images_with_filenames(folder,label)
    test_imgs+=i; test_lbls+=l; test_filenames+=f

print(f"TEST: {sum(test_lbls)} anemic / {len(test_lbls)} total")

# =========================
# DATASET
# =========================
class SingleEyeDataset(Dataset):
    def __init__(self, imgs, labels, transform):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        return self.transform(self.imgs[idx]), self.labels[idx]

def seed_worker(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)
    torch.manual_seed(SEED + worker_id)

# =========================
# MODEL
# =========================
class SingleResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(512,1)
    def forward(self,x):
        return self.backbone(x)

# =========================
# METRICS WITH PREDICTIONS
# =========================
@torch.no_grad()
def evaluate_with_predictions(model, loader, filenames):
    model.eval()
    preds, probs, labels_all = [], [], []
    all_filenames = []
    
    # Get all filenames in loader order
    batch_size = loader.batch_size
    for i in range(0, len(filenames), batch_size):
        batch_end = min(i + batch_size, len(filenames))
        all_filenames.extend(filenames[i:batch_end])
    
    for imgs, labels in loader:
        imgs = imgs.to(device).float()
        labels = labels.to(device).float().unsqueeze(1)
        out = model(imgs)
        p = torch.sigmoid(out).cpu().numpy().flatten()
        pred = (p > 0.5).astype(int)
        preds.extend(pred.tolist())
        probs.extend(p.tolist())
        labels_all.extend(labels.cpu().numpy().flatten().tolist())
    if len(set(labels_all))<2:
        return float("nan"),float("nan"),float("nan"),float("nan"),float("nan"),0,0,0,0, labels_all, probs, all_filenames, preds
    P,R,F1,_ = precision_recall_fscore_support(labels_all,preds,average='binary')
    acc = accuracy_score(labels_all,preds)
    auc = roc_auc_score(labels_all,probs)
    tn,fp,fn,tp = confusion_matrix(labels_all,preds,labels=[0,1]).ravel()
    return P,R,F1,acc,auc,tp,tn,fp,fn, labels_all, probs, all_filenames, preds

# =========================
# PLOTTING FUNCTIONS
# =========================
def plot_roc_curve(y_true, y_scores, title, save_path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Anemic', 'Anemic'],
                yticklabels=['Non-Anemic', 'Anemic'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(pytorch_metrics, tflite_metrics, save_path):
    import matplotlib.pyplot as plt
    
    metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'AUC']
    pytorch_vals = [pytorch_metrics['Test_Precision'], pytorch_metrics['Test_Recall'], 
                    pytorch_metrics['Test_F1'], pytorch_metrics['Test_Accuracy'], 
                    pytorch_metrics['Test_AUC']]
    tflite_vals = [tflite_metrics[0], tflite_metrics[1], 
                   tflite_metrics[2], tflite_metrics[3], 
                   tflite_metrics[4]]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, pytorch_vals, width, label='PyTorch', color='steelblue')
    plt.bar(x + width/2, tflite_vals, width, label='TFLite', color='darkorange')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('PyTorch vs TFLite Performance Comparison')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# =========================
# SAVE PREDICTIONS TO CSV
# =========================
def save_predictions_to_csv(filenames, true_labels, pred_labels, pred_probs, output_path):
    # Convert labels to readable format
    true_labels_str = ['Anemic' if label == 1 else 'Non-Anemic' for label in true_labels]
    pred_labels_str = ['Anemic' if label == 1 else 'Non-Anemic' for label in pred_labels]
    
    # Calculate confusion matrix indicators
    tp = [1 if (t == 1 and p == 1) else 0 for t, p in zip(true_labels, pred_labels)]
    tn = [1 if (t == 0 and p == 0) else 0 for t, p in zip(true_labels, pred_labels)]
    fp = [1 if (t == 0 and p == 1) else 0 for t, p in zip(true_labels, pred_labels)]
    fn = [1 if (t == 1 and p == 0) else 0 for t, p in zip(true_labels, pred_labels)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'file_id': filenames,
        'actual_value': true_labels_str,
        'predicted_value': pred_labels_str,
        'predicted_probability': pred_probs,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to: {output_path}")

# =========================
# TRAINING LOOP
# =========================
def train_and_eval_single():
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((RESOLUTION,RESOLUTION)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    eval_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((RESOLUTION,RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    X = train_imgs
    y = train_lbls
    filenames = train_filenames
    
    if len(y) < N_SPLITS:
        raise RuntimeError("Not enough training samples for CV")

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    results = []
    
    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X, y), 1):
        print(f"\n--- Fold {fold} ---")
        
        train_subset_imgs = [X[i] for i in tr_idx]
        train_subset_lbls = [y[i] for i in tr_idx]
        train_subset_filenames = [filenames[i] for i in tr_idx]
        val_subset_imgs = [X[i] for i in vl_idx]
        val_subset_lbls = [y[i] for i in vl_idx]
        val_subset_filenames = [filenames[i] for i in vl_idx]
        
        tr_loader = DataLoader(
            SingleEyeDataset(train_subset_imgs, train_subset_lbls, train_tf),
            batch_size=BATCH_CV, shuffle=True, num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
            generator=torch.Generator().manual_seed(SEED)
        )
        vl_loader = DataLoader(
            SingleEyeDataset(val_subset_imgs, val_subset_lbls, eval_tf),
            batch_size=BATCH_CV, shuffle=False, num_workers=NUM_WORKERS
        )

        model = SingleResNet18().to(device)
        opt = optim.Adam(model.parameters(), lr=LR_CV)
        loss_fn = nn.BCEWithLogitsLoss()
        scaler = GradScaler(enabled=USE_AMP and device.type == "cuda")

        for ep in range(EPOCHS_CV):
            model.train()
            total_loss = 0.0
            for imgs, labels in tr_loader:
                imgs = imgs.to(device).float()
                labels = labels.to(device).float().unsqueeze(1)
                opt.zero_grad(set_to_none=True)
                with autocast(enabled=USE_AMP and device.type == "cuda"):
                    out = model(imgs)
                    loss = loss_fn(out, labels)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total_loss += loss.item()
            
            if (ep + 1) % 20 == 0 or ep == EPOCHS_CV - 1:
                print(f"Epoch {ep+1}/{EPOCHS_CV} Loss: {total_loss:.4f}")
            
            if EARLY_STOP_PR:
                P, R, _, _, _, _, _, _, _, _, _, _, _ = evaluate_with_predictions(model, vl_loader, val_subset_filenames)
                if P >= EARLY_STOP_PR and R >= EARLY_STOP_PR:
                    print(f"✅ Early stop at epoch {ep+1}: P={P:.3f}, R={R:.3f}")
                    break

        val_metrics = evaluate_with_predictions(model, vl_loader, val_subset_filenames)
        results.append({
             'Fold': fold,
             'Val_Precision': val_metrics[0],
             'Val_Recall': val_metrics[1],
             'Val_F1': val_metrics[2],
             'Val_Accuracy': val_metrics[3],
             'Val_AUC': val_metrics[4],
             'Val_TP': val_metrics[5],
             'Val_TN': val_metrics[6],
             'Val_FP': val_metrics[7],
             'Val_FN': val_metrics[8],
         })
        print(f"Fold {fold} → P={val_metrics[0]:.3f}, R={val_metrics[1]:.3f}")
        
        if SAVE_EVERY_FOLD_MODEL:
             torch.save({
                 'model_state': model.state_dict(),
                 'fold': fold,
                 'val_metrics': {
                     'precision': val_metrics[0],
                     'recall': val_metrics[1],
                     'f1': val_metrics[2],
                     'accuracy': val_metrics[3],
                     'auc': val_metrics[4],
                     'tp': val_metrics[5],
                     'tn': val_metrics[6],
                     'fp': val_metrics[7],
                     'fn': val_metrics[8],
                 }
             }, os.path.join(OUTPUT_DIR, f"fold_{fold}.pt"))

    _cv_cols = ['Fold','Val_Precision','Val_Recall','Val_F1','Val_Accuracy','Val_AUC','Val_TP','Val_TN','Val_FP','Val_FN']
    pd.DataFrame(results)[_cv_cols].to_csv(os.path.join(OUTPUT_DIR, "cv_results.csv"), index=False)
    
    df = pd.DataFrame(results)
    df['minPR'] = df[['Val_Precision', 'Val_Recall']].min(axis=1)
    candidates = df[(df.Val_Precision >= 0.90) & (df.Val_Recall >= 0.90)]
    
    if len(candidates) > 0:
        best = candidates.sort_values(['Val_F1', 'Val_AUC', 'minPR'], ascending=False).iloc[0]
    else:
        best = df.sort_values(['minPR', 'Val_F1', 'Val_AUC'], ascending=False).iloc[0]
    
    best_fold = int(best['Fold'])
    print(f"✅ Best fold = {best_fold} | P={best['Val_Precision']:.3f}, R={best['Val_Recall']:.3f}")

    # Final test evaluation
    test_loader = DataLoader(
        SingleEyeDataset(test_imgs, test_lbls, eval_tf),
        batch_size=BATCH_CV, shuffle=False, num_workers=NUM_WORKERS
    )

    checkpoint = torch.load(
        os.path.join(OUTPUT_DIR, f"fold_{best_fold}.pt"),
        map_location=device,
        weights_only=False
    )
    model = SingleResNet18().to(device)
    model.load_state_dict(checkpoint['model_state'])

    test_metrics = evaluate_with_predictions(model, test_loader, test_filenames)

    print("\n📊 FINAL TEST RESULTS (Original Test Set):")
    print(f"Precision: {test_metrics[0]:.4f}")
    print(f"Recall:    {test_metrics[1]:.4f}")
    print(f"F1 score:  {test_metrics[2]:.4f}")
    print(f"Accuracy:  {test_metrics[3]:.4f}")
    print(f"AUC:       {test_metrics[4]:.4f}")
    print(f"TP, TN, FP, FN: {int(test_metrics[5])}, {int(test_metrics[6])}, {int(test_metrics[7])}, {int(test_metrics[8])}")
  
    _test_row = [{
         'Test_Precision': test_metrics[0],
         'Test_Recall': test_metrics[1],
         'Test_F1': test_metrics[2],
         'Test_Accuracy': test_metrics[3],
         'Test_AUC': test_metrics[4],
         'Test_TP': test_metrics[5],
         'Test_TN': test_metrics[6],
         'Test_FP': test_metrics[7],
         'Test_FN': test_metrics[8],
         'Best_Fold': best_fold
    }]
    _test_cols = ['Test_Precision','Test_Recall','Test_F1','Test_Accuracy','Test_AUC','Test_TP','Test_TN','Test_FP','Test_FN','Best_Fold']
    pd.DataFrame(_test_row)[_test_cols].to_csv(os.path.join(OUTPUT_DIR, "test_results.csv"), index=False)
    
    # Save detailed predictions to CSV
    save_predictions_to_csv(
        test_metrics[11],  # filenames
        test_metrics[9],   # true labels
        test_metrics[12],  # pred labels
        test_metrics[10],  # pred probs
        os.path.join(OUTPUT_DIR, "detailed_predictions_pytorch.csv")
    )
    
    # Plot ROC curve and confusion matrix for PyTorch model
    plot_roc_curve(test_metrics[9], test_metrics[10], 
                   "ROC Curve - PyTorch Model (Original Chronological Test Set)", 
                   os.path.join(OUTPUT_DIR, "roc_curve_pytorch.png"))
    plot_confusion_matrix(test_metrics[9], 
                          test_metrics[12], 
                          "Confusion Matrix - PyTorch Model", 
                          os.path.join(OUTPUT_DIR, "confusion_matrix_pytorch.png"))
    
    return model

# =========================
# TFLITE CONVERSION
# =========================
def convert_to_tflite(model, output_dir, resolution):
    import torch.onnx
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    model.eval()
    dummy_input = torch.randn(1, 3, resolution, resolution, device=device)
    onnx_path = os.path.join(output_dir, "model.onnx")
    tf_path = os.path.join(output_dir, "tf_model")
    tflite_path = os.path.join(output_dir, "single_eye_resnet18.tflite")
    
    print("\n--- Starting TFLite Conversion Pipeline ---")
    
    # PyTorch → ONNX
    print("1. Converting PyTorch model to ONNX...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            # No dynamic_axes → fixes input size
        )
        print(f"   ✅ ONNX model saved to: {onnx_path}")
    except Exception as e:
        print(f"   ❌ PyTorch to ONNX failed: {e}")
        return

    # ONNX → TensorFlow
    print("2. Converting ONNX model to TensorFlow SavedModel...")
    try:
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(tf_path)
        print(f"   ✅ TensorFlow SavedModel saved to: {tf_path}")
    except Exception as e:
        print(f"   ❌ ONNX to TensorFlow failed: {e}")
        return

    # TensorFlow → TFLite
    print("3. Converting TensorFlow SavedModel to TFLite...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"   ✅ TFLite model saved to: {tflite_path}")
        print(f"   TFLite Model Size: {os.path.getsize(tflite_path) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"   ❌ TensorFlow to TFLite failed: {e}")
        return

# =========================
# TFLITE EVALUATION (AUTO-DETECT INPUT SIZE) WITH PREDICTIONS
# =========================
def evaluate_tflite_on_test_with_predictions(tflite_path, test_imgs, test_lbls, test_filenames):
    import tensorflow as tf
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, confusion_matrix
    import numpy as np
    from PIL import Image
    from torchvision.transforms.functional import to_tensor, normalize

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    expected_shape = input_details[0]['shape']
    print(f"🔍 TFLite model input shape: {expected_shape}")

    if len(expected_shape) != 4:
        raise ValueError(f"Expected 4D input, got {expected_shape}")
    
    batch = expected_shape[0]
    assert batch == 1, "Batch size must be 1"

    # Detect layout: NCHW if shape[1] == 3, NHWC if shape[3] == 3
    if expected_shape[1] == 3 and expected_shape[3] != 3:
        layout = 'NCHW'
        _, _, h, w = expected_shape
        resize_h, resize_w = int(h), int(w)
        print(f"   ➤ Detected layout: NCHW")
    elif expected_shape[3] == 3 and expected_shape[1] != 3:
        layout = 'NHWC'
        _, h, w, _ = expected_shape
        resize_h, resize_w = int(h), int(w)
        print(f"   ➤ Detected layout: NHWC")
    else:
        # Fallback: assume NHWC if last dim is 3
        if expected_shape[-1] == 3:
            layout = 'NHWC'
            _, h, w, _ = expected_shape
            resize_h, resize_w = int(h), int(w)
        elif expected_shape[1] == 3:
            layout = 'NCHW'
            _, _, h, w = expected_shape
            resize_h, resize_w = int(h), int(w)
        else:
            raise ValueError(f"Cannot determine layout from shape {expected_shape}")

    preds, probs, labels_all = [], [], []

    # Use PIL + torchvision to EXACTLY match PyTorch preprocessing
    def preprocess_pil_style(img_rgb, target_size):
        """
        Reproduce: 
          transforms.ToPILImage() → Resize → ToTensor → Normalize
        """
        # img_rgb: numpy array (H, W, C), uint8, RGB
        pil_img = Image.fromarray(img_rgb)
        # Resize with PIL BILINEAR (same as torchvision)
        resized_pil = pil_img.resize((target_size[1], target_size[0]), Image.BILINEAR)  # (W, H)
        # ToTensor: (H, W, C) uint8 → (C, H, W) float32 [0,1]
        tensor = to_tensor(resized_pil)  # shape: (C, H, W)
        # Normalize with ImageNet stats
        normalized = normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return normalized.numpy()  # (C, H, W), float32

    for img, label in zip(test_imgs, test_lbls):
        # img is RGB numpy array (H, W, C), uint8 — same as loaded by cv2.cvtColor(..., cv2.COLOR_BGR2RGB)
        img_norm_nchw = preprocess_pil_style(img, (resize_h, resize_w))  # (C, H, W)

        if layout == 'NCHW':
            input_data = np.expand_dims(img_norm_nchw, axis=0)  # (1, C, H, W)
        else:  # NHWC
            img_norm_nhwc = np.transpose(img_norm_nchw, (1, 2, 0))  # (H, W, C)
            input_data = np.expand_dims(img_norm_nhwc, axis=0)  # (1, H, W, C)

        input_data = input_data.astype(input_details[0]['dtype'])

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])  # [1, 1]
        logit = output[0][0]
        prob = 1.0 / (1.0 + np.exp(-logit))
        pred = int(prob > 0.5)

        preds.append(pred)
        probs.append(prob)
        labels_all.append(label)

    if len(set(labels_all)) < 2:
        print("⚠️ Only one class in test set!")
        return None

    P, R, F1, _ = precision_recall_fscore_support(labels_all, preds, average='binary')
    acc = accuracy_score(labels_all, preds)
    auc = roc_auc_score(labels_all, probs)
    tn, fp, fn, tp = confusion_matrix(labels_all, preds, labels=[0, 1]).ravel()

    return P, R, F1, acc, auc, tp, tn, fp, fn, labels_all, probs, test_filenames, preds

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    # Train and evaluate
    zz = train_and_eval_single()
    print("\n✅ Single-eye model finished reproducibly with NO DATA LEAKAGE.")
    
    # Convert to TFLite
    convert_to_tflite(zz, OUTPUT_DIR, RESOLUTION)
    print("✅ TFLite conversion pipeline complete.")

    # Re-evaluate TFLite on same test set
    print("\n🔍 Loading TFLite model and re-evaluating on original test set...")
    tflite_file = os.path.join(OUTPUT_DIR, "single_eye_resnet18.tflite")
    if not os.path.exists(tflite_file):
        raise FileNotFoundError(f"TFLite model not found at {tflite_file}")

    tflite_metrics = evaluate_tflite_on_test_with_predictions(tflite_file, test_imgs, test_lbls, test_filenames)

    if tflite_metrics:
        P, R, F1, acc, auc, tp, tn, fp, fn, tflite_labels, tflite_probs, tflite_filenames, tflite_preds = tflite_metrics
        print("\n📊 TFLITE TEST RESULTS (Same test set):")
        print(f"Precision: {P:.4f}")
        print(f"Recall:    {R:.4f}")
        print(f"F1 score:  {F1:.4f}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"AUC:       {auc:.4f}")
        print(f"TP, TN, FP, FN: {int(tp)}, {int(tn)}, {int(fp)}, {int(fn)}")

        # Save detailed TFLite predictions to CSV
        save_predictions_to_csv(
            tflite_filenames,
            tflite_labels,
            tflite_preds,
            tflite_probs,
            os.path.join(OUTPUT_DIR, "detailed_predictions_tflite.csv")
        )

        # Plot ROC curve and confusion matrix for TFLite model
        plot_roc_curve(tflite_labels, tflite_probs, 
                       "ROC Curve - TFLite Model (Original Chronological Test Set)", 
                       os.path.join(OUTPUT_DIR, "roc_curve_tflite.png"))
        plot_confusion_matrix(tflite_labels, 
                              tflite_preds, 
                              "Confusion Matrix - TFLite Model", 
                              os.path.join(OUTPUT_DIR, "confusion_matrix_tflite.png"))

        test_results_path = os.path.join(OUTPUT_DIR, "test_results.csv")
        if os.path.exists(test_results_path):
            orig = pd.read_csv(test_results_path).iloc[0]
            print("\n🔍 Comparing with original PyTorch test results:")
            print(f"PyTorch → P: {orig['Test_Precision']:.4f}, R: {orig['Test_Recall']:.4f}, AUC: {orig['Test_AUC']:.4f}")
            print(f"TFLite  → P: {P:.4f}, R: {R:.4f}, AUC: {auc:.4f}")
            
            # Create metrics comparison plot
            plot_metrics_comparison(orig.to_dict(), tflite_metrics, 
                                   os.path.join(OUTPUT_DIR, "metrics_comparison.png"))
            
            tol = 1e-3
            p_ok = abs(P - orig['Test_Precision']) < tol
            r_ok = abs(R - orig['Test_Recall']) < tol
            auc_ok = abs(auc - orig['Test_AUC']) < tol
            
            if p_ok and r_ok and auc_ok:
                print("✅ TFLite results match PyTorch within tolerance (1e-3).")
            else:
                print("⚠️ TFLite results differ from PyTorch (check normalization or export).")
        else:
            print("⚠️ Original test results not found for comparison.")
    else:
        print("❌ TFLite evaluation failed.")

    print("\n✅ Analysis complete: Original chronological test set evaluated with detailed CSV predictions and comprehensive plots!")
    print(f"✅ Detailed prediction CSVs and plots saved to: {OUTPUT_DIR}")
