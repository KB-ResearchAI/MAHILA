''' Tongue '''
#SAM Model
''' Tongue '''
import os
import tempfile
import zipfile
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import psycopg2
from datetime import timedelta
import cv2
import random
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from torchvision.io import read_image

# -------------------------------
# Utility Functions
# -------------------------------

def load_and_resize_images(directory, size=(224, 224)):
    images = []
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist. Skipping.")
        return images
    for filename in os.listdir(directory):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(directory, filename)
            try:
                image = read_image(image_path).float() / 255.0  # Normalize to [0,1]
                image = transforms.Resize(size)(image)
                images.append((filename, image))
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
    return images


def load_data():
    newc = psycopg2.connect(
        user='b',
        password='yhq',
        host='wadia.c',
        port='5432',
        database='wa'
    )
    anemia_image = pd.read_sql_query('SELECT * FROM "anemia_image_samples"', con=newc)
    newc.close()
    # Filter only LID samples
    anemia_image = anemia_image[anemia_image['sample_id'].astype(str).str.contains("LID", na=False)]
    # Convert sample_date from Unix timestamp
    anemia_image['sample_date'] = pd.to_datetime(anemia_image['sample_date'], unit='s')
    anemia_image['sample_date'] = anemia_image['sample_date'] + timedelta(hours=5, minutes=30)
    return anemia_image


def preprocess_data(anemia_image):
    anemia_image = anemia_image[anemia_image['hb_value'] > 0].copy()
    anemia_image = anemia_image.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Assign initial train/test split (80/20)
    split_point = int(0.8 * len(anemia_image))
    anemia_image['split'] = 'test'
    anemia_image.loc[:split_point - 1, 'split'] = 'train'

    # Refine split based on anemia status (Hb < 9)
    anemia_image.loc[(anemia_image['hb_value'] < 9) & (anemia_image['split'] == 'train'), 'split'] = 'anemic_train'
    anemia_image.loc[(anemia_image['hb_value'] >= 9) & (anemia_image['split'] == 'train'), 'split'] = 'anemic_not_train'
    anemia_image.loc[(anemia_image['hb_value'] < 9) & (anemia_image['split'] == 'test'), 'split'] = 'anemic_test'
    anemia_image.loc[(anemia_image['hb_value'] >= 9) & (anemia_image['split'] == 'test'), 'split'] = 'anemic_not_test'
    
    return anemia_image


def refine_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
    return refined_mask


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def extract_and_save_roi(images, predictor, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename, image in images:
        try:
            # Convert to numpy (HWC, [0,255], uint8)
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)

            # Set image in SAM predictor
            predictor.set_image(image_np)

            # Predict mask using center point
            h, w = image_np.shape[:2]
            point_coords = np.array([[w // 2, h // 2]])
            masks, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=np.array([1]),
                multimask_output=False
            )

            mask = masks[0]  # boolean array
            mask_refined = refine_mask(mask)

            # Apply mask
            filtered_roi = cv2.bitwise_and(image_np, image_np, mask=mask_refined)

            # Resize to 224 width (maintain aspect ratio)
            filtered_roi_resized = resize_image(filtered_roi, width=224)

            # Save as BGR (OpenCV format)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, cv2.cvtColor(filtered_roi_resized, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Error processing {filename}: {e}")


def extract_images_from_zip(data, zip_folder_path, temp_extract_folder, base_directory_path):
    record_ids_in_data = set(data['record_id'].astype(str).values)

    for file_name in os.listdir(zip_folder_path):
        if not file_name.endswith('.zip'):
            continue
        zip_file_path = os.path.join(zip_folder_path, file_name)
        record_id = os.path.splitext(file_name)[0]

        if record_id not in record_ids_in_data:
            continue

        split = data.loc[data['record_id'].astype(str) == record_id, 'split'].values[0]

        # Extract zip
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_folder)
        except Exception as e:
            print(f"Failed to extract {zip_file_path}: {e}")
            continue

        # Process extracted files
        for extracted_file in os.listdir(temp_extract_folder):
            if "tongue_" not in extracted_file:
                continue
            for tongue_number in ['1', '2', '3']:
                if f"tongue_{tongue_number}" in extracted_file:
                    dest_folder = os.path.join(base_directory_path, f"tongue_{tongue_number}", split)
                    os.makedirs(dest_folder, exist_ok=True)
                    src_path = os.path.join(temp_extract_folder, extracted_file)
                    dst_path = os.path.join(dest_folder, extracted_file)
                    try:
                        shutil.move(src_path, dst_path)
                    except Exception as e:
                        print(f"Error moving {src_path} to {dst_path}: {e}")


# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":
    # Load and preprocess data
    anemia_image = load_data()
    data = preprocess_data(anemia_image)

    base_directory_path = '/media/sarfraaz/HDD/Sarfraaz_Backup/anemia_research/'
    zip_folder_path = "/media/sarfraaz/HDD/Sarfraaz_Backup/anemia_research/downloaded_files/"
    temp_extract_folder = tempfile.mkdtemp(dir=base_directory_path)

    splits = ['anemic_train', 'anemic_test', 'anemic_not_train', 'anemic_not_test']
    tongue_numbers = ['1', '2', '3']

    # Reset directories
    for tongue_num in tongue_numbers:
        for split in splits:
            img_dir = os.path.join(base_directory_path, f"tongue_{tongue_num}", split)
            roi_dir = os.path.join(base_directory_path, f"tongue_{tongue_num}", f"{split}_roi")
            if os.path.exists(img_dir):
                shutil.rmtree(img_dir)
            if os.path.exists(roi_dir):
                shutil.rmtree(roi_dir)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(roi_dir, exist_ok=True)

    # Extract images from ZIPs
    extract_images_from_zip(data, zip_folder_path, temp_extract_folder, base_directory_path)

    # Load SAM model
    sam_checkpoint = "/media/sarfraaz/HDD/Sarfraaz_Backup/segment-anything/sam_vit_b_01ec64.pth"
    model_type = "vit_b"  # ✅ Fixed: was "vit_h" but checkpoint is vit_b

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    # Process each tongue and split
    for tongue_num in tongue_numbers:
        for split in splits:
            input_dir = os.path.join(base_directory_path, f"tongue_{tongue_num}", split)
            output_dir = os.path.join(base_directory_path, f"tongue_{tongue_num}", f"{split}_roi")
            images = load_and_resize_images(input_dir, size=(224, 224))
            if images:
                extract_and_save_roi(images, predictor, output_dir)
            else:
                print(f"No images found in {input_dir}")

    # Clean up temp folder
    shutil.rmtree(temp_extract_folder, ignore_errors=True)
    print("Processing complete.")
    
''' Tongue - Improved SAM Segmentation with Post-Filtering '''
import os
import tempfile
import zipfile
import shutil
import pandas as pd
import numpy as np
from datetime import timedelta
import cv2
import torch
import torchvision.transforms as transforms
from segment_anything import sam_model_registry, SamPredictor
from torchvision.io import read_image
import psycopg2

# -------------------------------
# Utility Functions
# -------------------------------

def load_and_resize_images(directory, size=(224, 224)):
    images = []
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist. Skipping.")
        return images
    for filename in os.listdir(directory):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(directory, filename)
            try:
                image = read_image(image_path).float() / 255.0
                image = transforms.Resize(size)(image)
                images.append((filename, image))
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
    return images

def load_data():
    newc = psycopg2.connect(
        user='wb',
        password='byq',
        host='w',
        port='5432',
        database='wa'
    )
    anemia_image = pd.read_sql_query('SELECT * FROM "anemia_image_samples"', con=newc)
    newc.close()
    anemia_image = anemia_image[anemia_image['sample_id'].astype(str).str.contains("LID", na=False)]
    anemia_image['sample_date'] = pd.to_datetime(anemia_image['sample_date'], unit='s')
    anemia_image['sample_date'] = anemia_image['sample_date'] + timedelta(hours=5, minutes=30)
    return anemia_image

def preprocess_data(anemia_image):
    anemia_image = anemia_image[anemia_image['hb_value'] > 0].copy()
    anemia_image = anemia_image.sample(frac=1, random_state=42).reset_index(drop=True)
    split_point = int(0.8 * len(anemia_image))
    anemia_image['split'] = 'test'
    anemia_image.loc[:split_point - 1, 'split'] = 'train'

    anemia_image.loc[(anemia_image['hb_value'] < 9) & (anemia_image['split'] == 'train'), 'split'] = 'anemic_train'
    anemia_image.loc[(anemia_image['hb_value'] >= 9) & (anemia_image['split'] == 'train'), 'split'] = 'anemic_not_train'
    anemia_image.loc[(anemia_image['hb_value'] < 9) & (anemia_image['split'] == 'test'), 'split'] = 'anemic_test'
    anemia_image.loc[(anemia_image['hb_value'] >= 9) & (anemia_image['split'] == 'test'), 'split'] = 'anemic_not_test'
    return anemia_image

def refine_mask(mask):
    kernel = np.ones((7, 7), np.uint8)  # slightly larger kernel
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Keep only the largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # skip background
    refined = (labels == largest_label).astype(np.uint8)
    return refined

def is_tongue_color_hsv(roi):
    """Check if dominant color in ROI is pink/red (tongue-like) using HSV."""
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    # Tongue: Hue ~ 0-20 or 160-180 (red/pink), Saturation > 30, Value > 50
    lower_red1 = np.array([0, 30, 50])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 30, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2
    red_ratio = np.sum(red_mask > 0) / red_mask.size
    return red_ratio > 0.2  # at least 20% red/pink

def extract_and_save_roi(images, predictor, output_dir, fallback_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    if fallback_dir:
        os.makedirs(fallback_dir, exist_ok=True)

    for filename, image in images:
        try:
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            h, w = image_np.shape[:2]

            predictor.set_image(image_np)

            # Use central bounding box instead of point
            margin = 0.2
            x1 = int(margin * w)
            y1 = int(margin * h)
            x2 = int((1 - margin) * w)
            y2 = int((1 - margin) * h)
            bbox = np.array([x1, y1, x2, y2])

            masks, _, _ = predictor.predict(box=bbox[None, :], multimask_output=False)
            mask = masks[0]

            # Refine: largest component + morphology
            mask_refined = refine_mask(mask)

            # Apply mask
            filtered_roi = cv2.bitwise_and(image_np, image_np, mask=mask_refined)

            # Resize to 224 width
            filtered_roi_resized = resize_image(filtered_roi, width=224)

            # Save candidate
            temp_path = os.path.join(output_dir, f"temp_{filename}")
            final_path = os.path.join(output_dir, filename)
            cv2.imwrite(temp_path, cv2.cvtColor(filtered_roi_resized, cv2.COLOR_RGB2BGR))

            # Validate by file size and color
            file_size_kb = os.path.getsize(temp_path) / 1024
            is_good_size = 7 <= file_size_kb <= 20
            is_good_color = is_tongue_color_hsv(filtered_roi_resized)

            if is_good_size and is_good_color:
                os.rename(temp_path, final_path)
                # print(f"✅ Accepted: {filename} ({file_size_kb:.1f} KB)")
            else:
                os.remove(temp_path)
                if fallback_dir:
                    # Save original image as fallback
                    fallback_path = os.path.join(fallback_dir, filename)
                    cv2.imwrite(fallback_path, cv2.cvtColor(resize_image(image_np, width=224), cv2.COLOR_RGB2BGR))
                print(f"⚠️ Rejected ROI: {filename} | Size: {file_size_kb:.1f} KB | Color OK: {is_good_color}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

def extract_images_from_zip(data, zip_folder_path, temp_extract_folder, base_directory_path):
    record_ids_in_data = set(data['record_id'].astype(str).values)
    for file_name in os.listdir(zip_folder_path):
        if not file_name.endswith('.zip'):
            continue
        zip_file_path = os.path.join(zip_folder_path, file_name)
        record_id = os.path.splitext(file_name)[0]
        if record_id not in record_ids_in_data:
            continue
        split = data.loc[data['record_id'].astype(str) == record_id, 'split'].values[0]
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_folder)
        except Exception as e:
            print(f"Failed to extract {zip_file_path}: {e}")
            continue
        for extracted_file in os.listdir(temp_extract_folder):
            if "tongue_" not in extracted_file:
                continue
            for tongue_number in ['1', '2', '3']:
                if f"tongue_{tongue_number}" in extracted_file:
                    dest_folder = os.path.join(base_directory_path, f"tongue_{tongue_number}", split)
                    os.makedirs(dest_folder, exist_ok=True)
                    src_path = os.path.join(temp_extract_folder, extracted_file)
                    dst_path = os.path.join(dest_folder, extracted_file)
                    try:
                        shutil.move(src_path, dst_path)
                    except Exception as e:
                        print(f"Error moving {src_path} to {dst_path}: {e}")

# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":
    anemia_image = load_data()
    data = preprocess_data(anemia_image)

    base_directory_path = '/media/sarfraaz/HDD/Sarfraaz_Backup/anemia_research/'
    zip_folder_path = "/media/sarfraaz/HDD/Sarfraaz_Backup/anemia_research/downloaded_files/"
    temp_extract_folder = tempfile.mkdtemp(dir=base_directory_path)

    splits = ['anemic_train', 'anemic_test', 'anemic_not_train', 'anemic_not_test']
    tongue_numbers = ['1', '2', '3']

    # Reset directories
    for tongue_num in tongue_numbers:
        for split in splits:
            img_dir = os.path.join(base_directory_path, f"tongue_{tongue_num}", split)
            roi_dir = os.path.join(base_directory_path, f"tongue_{tongue_num}", f"{split}_roi")
            fallback_dir = os.path.join(base_directory_path, f"tongue_{tongue_num}", f"{split}_fallback")
            for d in [img_dir, roi_dir, fallback_dir]:
                if os.path.exists(d):
                    shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)

    extract_images_from_zip(data, zip_folder_path, temp_extract_folder, base_directory_path)

    # Load SAM
    sam_checkpoint = "/media/sarfraaz/HDD/Sarfraaz_Backup/segment-anything/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    # Process
    for tongue_num in tongue_numbers:
        for split in splits:
            input_dir = os.path.join(base_directory_path, f"tongue_{tongue_num}", split)
            output_dir = os.path.join(base_directory_path, f"tongue_{tongue_num}", f"{split}_roi")
            fallback_dir = os.path.join(base_directory_path, f"tongue_{tongue_num}", f"{split}_fallback")
            images = load_and_resize_images(input_dir, size=(224, 224))
            if images:
                extract_and_save_roi(images, predictor, output_dir, fallback_dir)
            else:
                print(f"No images found in {input_dir}")

    shutil.rmtree(temp_extract_folder, ignore_errors=True)
    print("✅ Processing complete. Check *_fallback folders for rejected ROIs.")
    