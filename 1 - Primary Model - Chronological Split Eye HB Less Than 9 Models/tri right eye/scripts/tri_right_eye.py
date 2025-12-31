# -*- coding: utf-8 -*-
"""
Reproducible Tri-right-Eye ResNet18 with SHARED BACKBONE & TFLite Verification
----------------------------------------------------------------------------------
✅ Shared ResNet18 across 3 inputs → ~45 MB TFLite
✅ TFLite export with SELECT_TF_OPS (GELU via FlexDelegate)
✅ Re-evaluates .tflite file and compares results with PyTorch
✅ Ensures no silent divergence between frameworks
✅ Added detailed prediction CSV with file IDs, predictions, probabilities, and confusion matrix indicators
✅ Added comprehensive plotting: ROC curves, confusion matrices, metrics comparison
"""

import os
import json
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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
SEED = 42
NUM_WORKERS = 0
PIN_MEMORY = False
USE_AMP = True
SAVE_EVERY_FOLD_MODEL = True
N_SPLITS = 5
RESOLUTION = 224  # Only one resolution used now
EPOCHS_CV = 150
BATCH_CV = 28
LR_CV = 0.00022

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

try:
    cv2.setNumThreads(0)
except Exception:
    pass

os.environ["PYTHONHASHSEED"] = str(SEED)
set_global_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except Exception:
        pass

# =========================
# PATHS
# =========================
base_path = "/home/ubuntu/anemia-storage/hb_mobilenet/mat_conjunctiva_all_consistent_deletion/"
output_dir = os.path.join(base_path, "tri_right_eye_hb_90_repro_bestfold_only_shared")
os.makedirs(output_dir, exist_ok=True)

def make_full_path(subdirs):
    return {k: os.path.join(base_path, v) for k, v in subdirs.items()}

train_dirs_anemic = make_full_path({
    'right1': 'tri_right_eye/right_eye_1_hb_less_than_9_0/conjunctiva_extracted/anemic_train_roi/',
    'right2': 'tri_right_eye/right_eye_2_hb_less_than_9_0/conjunctiva_extracted/anemic_train_roi/',
    'right3': 'tri_right_eye/right_eye_3_hb_less_than_9_0/conjunctiva_extracted/anemic_train_roi/'
})
train_dirs_non = make_full_path({
    'right1': 'tri_right_eye/right_eye_1_hb_less_than_9_0/conjunctiva_extracted/anemic_not_train_roi/',
    'right2': 'tri_right_eye/right_eye_2_hb_less_than_9_0/conjunctiva_extracted/anemic_not_train_roi/',
    'right3': 'tri_right_eye/right_eye_3_hb_less_than_9_0/conjunctiva_extracted/anemic_not_train_roi/'
})
val_dirs_anemic = make_full_path({
    'right1': 'tri_right_eye/right_eye_1_hb_less_than_9_0/conjunctiva_extracted/anemic_val_roi/',
    'right2': 'tri_right_eye/right_eye_2_hb_less_than_9_0/conjunctiva_extracted/anemic_val_roi/',
    'right3': 'tri_right_eye/right_eye_3_hb_less_than_9_0/conjunctiva_extracted/anemic_val_roi/'
})
val_dirs_non = make_full_path({
    'right1': 'tri_right_eye/right_eye_1_hb_less_than_9_0/conjunctiva_extracted/anemic_not_val_roi/',
    'right2': 'tri_right_eye/right_eye_2_hb_less_than_9_0/conjunctiva_extracted/anemic_not_val_roi/',
    'right3': 'tri_right_eye/right_eye_3_hb_less_than_9_0/conjunctiva_extracted/anemic_not_val_roi/'
})
test_dirs_anemic = make_full_path({
    'right1': 'tri_right_eye/right_eye_1_hb_less_than_9_0/conjunctiva_extracted/anemic_test_roi/',
    'right2': 'tri_right_eye/right_eye_2_hb_less_than_9_0/conjunctiva_extracted/anemic_test_roi/',
    'right3': 'tri_right_eye/right_eye_3_hb_less_than_9_0/conjunctiva_extracted/anemic_test_roi/'
})
test_dirs_non = make_full_path({
    'right1': 'tri_right_eye/right_eye_1_hb_less_than_9_0/conjunctiva_extracted/anemic_not_test_roi/',
    'right2': 'tri_right_eye/right_eye_2_hb_less_than_9_0/conjunctiva_extracted/anemic_not_test_roi/',
    'right3': 'tri_right_eye/right_eye_3_hb_less_than_9_0/conjunctiva_extracted/anemic_not_test_roi/'
})

# =========================
# UTILS
# =========================
def base_from(fname, suffix):
    return fname[:-len(suffix)] if fname.endswith(suffix) else None

def common_bases_right(dirs_map):
    suffixes = {'right1': '_right_eye_1.png', 'right2': '_right_eye_2.png', 'right3': '_right_eye_3.png'}
    bases_sets = []
    for k in ['right1', 'right2', 'right3']:
        folder = dirs_map[k]
        if not os.path.isdir(folder):
            return []
        names = [f for f in os.listdir(folder) if f.endswith(suffixes[k])]
        bases = {base_from(f, suffixes[k]) for f in names if base_from(f, suffixes[k]) is not None}
        bases_sets.append(bases)
    if not bases_sets:
        return []
    inter = set.intersection(*bases_sets)
    return sorted(inter)

def load_tri_images_by_bases_with_filenames(dirs_map, bases):
    out = {'r1': [], 'r2': [], 'r3': [], 'filenames': []}
    key_map = {'r1': ('right1', '_right_eye_1.png'), 'r2': ('right2', '_right_eye_2.png'), 'r3': ('right3', '_right_eye_3.png')}
    for b in bases:
        imgs, failed = {}, False
        for short_k, (long_k, suf) in key_map.items():
            path = os.path.join(dirs_map[long_k], b + suf)
            if not os.path.isfile(path):
                failed = True
                break
            img = cv2.imread(path)
            if img is None:
                failed = True
                break
            imgs[short_k] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not failed:
            out['r1'].append(imgs['r1'])
            out['r2'].append(imgs['r2'])
            out['r3'].append(imgs['r3'])
            out['filenames'].append(b)  # Use base name as identifier
    return out

def prepare_dataset_right_with_filenames(anemic_dirs, non_dirs, split_name="(split)"):
    bases_a = common_bases_right(anemic_dirs)
    bases_n = common_bases_right(non_dirs)
    imgs_a = load_tri_images_by_bases_with_filenames(anemic_dirs, bases_a)
    imgs_n = load_tri_images_by_bases_with_filenames(non_dirs, bases_n)
    data = {
        'r1': imgs_a['r1'] + imgs_n['r1'],
        'r2': imgs_a['r2'] + imgs_n['r2'],
        'r3': imgs_a['r3'] + imgs_n['r3'],
        'filenames': imgs_a['filenames'] + imgs_n['filenames'],
        'label': [1]*len(imgs_a['r1']) + [0]*len(imgs_n['r1'])
    }
    print(f"✅ {split_name}: anemic={len(imgs_a['r1'])}, non-anemic={len(imgs_n['r1'])}, total={len(data['label'])}")
    try:
        df_bases = pd.DataFrame({'class': ['anemic']*len(bases_a) + ['non_anemic']*len(bases_n),
                                 'base_id': bases_a + bases_n})
        df_bases.to_csv(os.path.join(output_dir, f"{split_name.lower()}_used_base_ids_right.csv"), index=False)
    except Exception:
        pass
    return data

def count_files(d):
    return sum(1 for f in sorted(os.listdir(d)) if f.endswith(".png")) if os.path.isdir(d) else 0

def print_dir_stats(title, dirs_map):
    print(f"\n📂 {title}")
    for k in ['right1','right2','right3']:
        p = dirs_map[k]; c = count_files(p)
        print(f"{k:7s} | {p} | files={c}")

# =========================
# LOAD DATA
# =========================

print_dir_stats("TEST  anemic (right)", test_dirs_anemic)
print_dir_stats("TEST  non-anemic (right)", test_dirs_non)

train_data = prepare_dataset_right_with_filenames(train_dirs_anemic, train_dirs_non, split_name="TRAIN")
val_data   = prepare_dataset_right_with_filenames(val_dirs_anemic,   val_dirs_non,   split_name="VAL")
test_data  = prepare_dataset_right_with_filenames(test_dirs_anemic,  test_dirs_non,  split_name="TEST")

if len(train_data['label']) == 0:
    raise RuntimeError("No tri-right-eye TRAIN samples found.")

# =========================
# DATASET
# =========================
class TrirightDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data['label'])
    def __getitem__(self, idx):
        images = [self.data[k][idx] for k in ['r1','r2','r3']]
        images = [self.transform(img) for img in images]
        label = self.data['label'][idx]
        return images, label

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def make_loader(dataset, batch_size, shuffle):
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and (device.type=='cuda'),
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
        generator=g,
    )

# =========================
# MODEL: SHARED RESNET18 BACKBONE
# =========================
class TriResNetright(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

        self.fusion = nn.Sequential(
            nn.Linear(3 * 512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1)
        )

    def forward(self, x1, x2, x3):
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        f3 = self.backbone(x3)
        x = torch.cat([f1, f2, f3], dim=1)
        return self.fusion(x)

# =========================
# EVALUATION (PyTorch) WITH PREDICTIONS
# =========================
@torch.no_grad()
def evaluate_with_predictions(model, loader, filenames):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    all_filenames = []
    
    # Get all filenames in loader order
    batch_size = loader.batch_size
    for i in range(0, len(filenames), batch_size):
        batch_end = min(i + batch_size, len(filenames))
        all_filenames.extend(filenames[i:batch_end])
    
    for imgs, labels in loader:
        x1, x2, x3 = [img.to(device).float() for img in imgs]
        labels = labels.to(device).float().unsqueeze(1)
        out = model(x1, x2, x3)
        prob = torch.sigmoid(out).cpu().numpy().flatten()
        pred = (prob > 0.5).astype(int)
        all_preds.extend(pred.tolist())
        all_probs.extend(prob.tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())

    if len(all_labels) == 0 or len(set(all_labels)) < 2:
        return [float('nan')] * 9 + [all_labels, all_probs, all_filenames, all_preds]

    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0,1]).ravel()
    return p, r, f1, acc, auc, tp, tn, fp, fn, all_labels, all_probs, all_filenames, all_preds

# =========================
# TFLITE CONVERSION
# =========================
def convert_to_tflite(best_model: nn.Module, output_dir: str, resolution: int, tflite_filename: str):
    import torch.onnx
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    best_model.eval().to('cpu')
    dummy_inputs = tuple(torch.randn(1, 3, resolution, resolution) for _ in range(3))
    onnx_path = os.path.join(output_dir, "tri_right_model.onnx")
    tf_path = os.path.join(output_dir, "tri_right_tf_model")
    tflite_path = os.path.join(output_dir, tflite_filename)

    print("\n--- Starting TFLite Conversion Pipeline ---")

    # Step 1: PyTorch -> ONNX
    print("1. Converting to ONNX...")
    try:
        torch.onnx.export(
            best_model,
            dummy_inputs,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input1', 'input2', 'input3'],
            output_names=['output']
        )
        print(f"   ✅ ONNX saved: {onnx_path}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

    # Step 2: ONNX -> TensorFlow
    print("2. ONNX -> TensorFlow SavedModel...")
    try:
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        if os.path.exists(tf_path):
            import shutil
            shutil.rmtree(tf_path)
        tf_rep.export_graph(tf_path)
        print(f"   ✅ TF SavedModel saved: {tf_path}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

    # Step 3: TF -> TFLite with SELECT_TF_OPS
    print("3. TF -> TFLite (SELECT_TF_OPS)...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"   ✅ TFLite saved: {tflite_path} ({size_mb:.2f} MB)")
        return tflite_path
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None

# =========================
# TFLITE RE-EVALUATION WITH PREDICTIONS
# =========================
def evaluate_tflite_model_with_predictions(tflite_path, test_data, resolution):
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    from torchvision.transforms.functional import to_tensor, normalize

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    sorted_inputs = sorted(input_details, key=lambda x: x['name'])
    print(f"🔍 TFLite input names: {[d['name'] for d in sorted_inputs]}")

    first_shape = sorted_inputs[0]['shape']
    if len(first_shape) == 4:
        if first_shape[1] == 3:  # [B,C,H,W]
            layout = 'NCHW'
        else:  # [B,H,W,C]
            layout = 'NHWC'

    resize_h, resize_w = int(first_shape[1] if layout == 'NHWC' else first_shape[2]), \
                         int(first_shape[2] if layout == 'NHWC' else first_shape[3])

    print(f"   ➤ Detected layout: {layout}, size: {resize_h}x{resize_w}")

    def preprocess_pil_style(img_rgb):
        img_pil = Image.fromarray(img_rgb)
        img_resized = img_pil.resize((resize_w, resize_h), Image.BILINEAR)
        tensor = to_tensor(img_resized)
        normalized = normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return normalized.numpy()

    all_preds, all_probs, all_labels, all_filenames = [], [], [], test_data['filenames']

    for i in range(len(test_data['label'])):
        imgs = [test_data['r1'][i], test_data['r2'][i], test_data['r3'][i]]
        label = test_data['label'][i]

        for idx, detail in enumerate(sorted_inputs):
            raw_img = imgs[idx]
            processed = preprocess_pil_style(raw_img)

            if layout == 'NCHW':
                model_input = np.expand_dims(processed, axis=0).astype(detail['dtype'])
            else:
                nhwc = np.transpose(processed, (1, 2, 0))
                model_input = np.expand_dims(nhwc, axis=0).astype(detail['dtype'])

            interpreter.set_tensor(detail['index'], model_input)

        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        logit = float(np.array(output).reshape(-1)[0])
        prob = 1.0 / (1.0 + np.exp(-logit))
        pred = int(prob > 0.5)

        all_preds.append(pred)
        all_probs.append(prob)
        all_labels.append(label)

    if len(set(all_labels)) < 2:
        auc = float('nan')
    else:
        auc = roc_auc_score(all_labels, all_probs)

    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0,1]).ravel()

    return p, r, f1, acc, auc, tp, tn, fp, fn, all_labels, all_probs, all_filenames, all_preds

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
# MAIN TRAINING LOOP
# =========================
if __name__ == "__main__":
    resolution = RESOLUTION
    results = []
    cv_index_records = []

    print(f"\n===== Processing right resolution: {resolution} =====")
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((resolution, resolution)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    labels_np = np.array(train_data['label'])

    fold = 1
    for train_idx, val_idx in kf.split(np.zeros_like(labels_np), labels_np):
        print(f"\n--- right Fold {fold} ---")
        cv_index_records.append({"fold": fold, "train_indices": train_idx.tolist(), "val_indices": val_idx.tolist()})

        train_subset = {k: [v[i] for i in train_idx] for k, v in train_data.items()}
        val_subset   = {k: [v[i] for i in val_idx]   for k, v in train_data.items()}

        train_loader = make_loader(TrirightDataset(train_subset, train_transform), BATCH_CV, True)
        val_loader   = make_loader(TrirightDataset(val_subset,   test_transform),  BATCH_CV, False)

        model = TriResNetright().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR_CV)
        scaler = GradScaler(enabled=(USE_AMP and device.type == "cuda"))

        for epoch in range(EPOCHS_CV):
            model.train()
            total_loss = 0.0
            for imgs, labels in train_loader:
                x1, x2, x3 = [img.to(device).float() for img in imgs]
                labels = labels.to(device).float().unsqueeze(1)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=(USE_AMP and device.type == "cuda")):
                    out = model(x1, x2, x3)
                    loss = criterion(out, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS_CV}] Loss: {total_loss:.6f}")

        val_metrics = evaluate_with_predictions(model, val_loader, val_subset['filenames'])
        result_row = {
            'EyeSet': 'right',
            'Resolution': resolution,
            'Fold': fold,
            'Val_Precision': val_metrics[0],
            'Val_Recall': val_metrics[1],
            'Val_F1': val_metrics[2],
            'Val_Accuracy': val_metrics[3],
            'Val_AUC': val_metrics[4],
            'Val_TP': val_metrics[5],
            'Val_TN': val_metrics[6],
            'Val_FP': val_metrics[7],
            'Val_FN': val_metrics[8]
        }
        results.append(result_row)
        print(results)
        if SAVE_EVERY_FOLD_MODEL:
            fold_path = os.path.join(output_dir, f"right_cv_fold_{fold}_res{resolution}.pt")
            torch.save({'model_state': model.state_dict()}, fold_path)

        fold += 1

    # Save CV results
    pd.DataFrame(results).to_csv(os.path.join(output_dir, "right_val_cross_validation_results.csv"), index=False)
    with open(os.path.join(output_dir, "right_cv_indices.json"), "w") as f:
        json.dump(cv_index_records, f, indent=2)

    # Select best fold
    df = pd.DataFrame(results)
    df['minPR'] = df[['Val_Precision','Val_Recall']].min(axis=1)
    candidates = df[(df['Val_Precision'] >= 0.90) & (df['Val_Recall'] >= 0.90)]
    best = candidates.sort_values(['Val_F1'], ascending=False).iloc[0] if len(candidates) > 0 else \
           df.sort_values(['minPR','Val_F1'], ascending=False).iloc[0]
    best_fold = int(best['Fold'])
    print(f"✅ Best fold = {best_fold}")

    # Load best model
    ckpt_path = os.path.join(output_dir, f"right_cv_fold_{best_fold}_res{resolution}.pt")
    best_model = TriResNetright().to(device)
    state = torch.load(ckpt_path, map_location=device)
    best_model.load_state_dict(state['model_state'])

    # Test evaluation (PyTorch)
    test_loader = make_loader(TrirightDataset(test_data, test_transform), BATCH_CV, False)
    test_metrics = evaluate_with_predictions(best_model, test_loader, test_data['filenames'])

    test_results_df = pd.DataFrame([{
        'ChosenFold': best_fold,
        'Test_Precision': test_metrics[0],
        'Test_Recall': test_metrics[1],
        'Test_F1': test_metrics[2],
        'Test_Accuracy': test_metrics[3],
        'Test_AUC': test_metrics[4],
        'Test_TP': test_metrics[5],
        'Test_TN': test_metrics[6],
        'Test_FP': test_metrics[7],
        'Test_FN': test_metrics[8]
    }])
    test_results_df.to_csv(os.path.join(output_dir, "right_bestfold_test_results.csv"), index=False)

    print("\n📊 TEST Results (Shared Backbone):")
    print(test_results_df.to_string(index=False))

    # Save detailed PyTorch predictions to CSV
    save_predictions_to_csv(
        test_metrics[11],  # filenames
        test_metrics[9],   # true labels
        test_metrics[12],  # pred labels
        test_metrics[10],  # pred probs
        os.path.join(output_dir, "detailed_predictions_pytorch.csv")
    )

    # Plot ROC curve and confusion matrix for PyTorch model
    plot_roc_curve(test_metrics[9], test_metrics[10], 
                   "ROC Curve - PyTorch Model (Original Chronological Test Set)", 
                   os.path.join(output_dir, "roc_curve_pytorch.png"))
    plot_confusion_matrix(test_metrics[9], 
                          test_metrics[12], 
                          "Confusion Matrix - PyTorch Model", 
                          os.path.join(output_dir, "confusion_matrix_pytorch.png"))

    # Convert to TFLite
    tflite_filename = "tri_right_eye_resnet18_shared.tflite"
    tflite_path = convert_to_tflite(best_model.cpu(), output_dir, resolution, tflite_filename)

    if tflite_path:
        size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"\n🎉 SUCCESS! Final TFLite model size: {size_mb:.2f} MB")
        print(f"📍 Path: {tflite_path}")

        # --- 🔍 Re-evaluate TFLite model ---
        print("\n🔍 Re-evaluating TFLite model on test set...")
        try:
            tflite_metrics = evaluate_tflite_model_with_predictions(tflite_path, test_data, resolution)
            tflite_results_df = pd.DataFrame([{
                'Source': 'TFLite',
                'Test_Precision': tflite_metrics[0],
                'Test_Recall': tflite_metrics[1],
                'Test_F1': tflite_metrics[2],
                'Test_Accuracy': tflite_metrics[3],
                'Test_AUC': tflite_metrics[4],
                'Test_TP': tflite_metrics[5],
                'Test_TN': tflite_metrics[6],
                'Test_FP': tflite_metrics[7],
                'Test_FN': tflite_metrics[8]
            }])

            combined = pd.concat([
                test_results_df.assign(Source='PyTorch'),
                tflite_results_df
            ], ignore_index=True)

            print("\n📊 COMPARISON: PyTorch vs TFLite")
            print(combined.to_string(index=False))
            combined.to_csv(os.path.join(output_dir, "pytorch_vs_tflite_comparison.csv"), index=False)

            # Save detailed TFLite predictions to CSV
            save_predictions_to_csv(
                tflite_metrics[11],  # filenames
                tflite_metrics[9],   # true labels
                tflite_metrics[12],  # pred labels
                tflite_metrics[10],  # pred probs
                os.path.join(output_dir, "detailed_predictions_tflite.csv")
            )

            # Plot ROC curve and confusion matrix for TFLite model
            plot_roc_curve(tflite_metrics[9], tflite_metrics[10], 
                           "ROC Curve - TFLite Model (Original Chronological Test Set)", 
                           os.path.join(output_dir, "roc_curve_tflite.png"))
            plot_confusion_matrix(tflite_metrics[9], 
                                  tflite_metrics[12], 
                                  "Confusion Matrix - TFLite Model", 
                                  os.path.join(output_dir, "confusion_matrix_tflite.png"))

            # Create metrics comparison plot
            plot_metrics_comparison(test_results_df.iloc[0].to_dict(), tflite_metrics, 
                                   os.path.join(output_dir, "metrics_comparison.png"))

            tol = 1e-3
            if (abs(tflite_metrics[2] - test_metrics[2]) < tol and
                abs(tflite_metrics[4] - test_metrics[4]) < tol):
                print("✅ TFLite results MATCH PyTorch within tolerance.")
            else:
                print("⚠️ WARNING: TFLite results differ significantly from PyTorch!")
        except Exception as e:
            print(f"❌ TFLite evaluation failed: {e}")
    else:
        print("❌ TFLite conversion failed.")

    print("\n✅ Pipeline completed. Model size reduced to ~45 MB via shared backbone.")
    print(f"✅ Detailed prediction CSVs and plots saved to: {output_dir}")
