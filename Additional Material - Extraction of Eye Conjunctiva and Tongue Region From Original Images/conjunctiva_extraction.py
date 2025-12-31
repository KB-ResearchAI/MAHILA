#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 22:29:02 2025

@author: sarfraaz
"""


   
import os
from sklearn.model_selection import GridSearchCV
import pandas as pd #Importing Pandas which is a data analysis package of python
import numpy as np #Importing Numpy which is numerical package of python
from pandas import ExcelWriter #Importing ExcelWriter to save the necessary outcomes in Excel Format
import psycopg2
import os  #Import operating System
import os.path #os.path to read the file from the desired location
import datetime
from datetime import timedelta
import zipfile
import shutil
import tempfile
import cv2
import random

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))


# Specify the root directory
root_directory = '/home/sarfraaz/Downloads/drive-download/'



import os

def list_filenames_in_subdirectories(root_dir):
    filenames = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            filenames.append(file)
    return filenames



# Get the list of all filenames in subdirectories
all_filenames = list_filenames_in_subdirectories(root_directory)

# Print the list of filenames
for filename in all_filenames:
    print(filename)
import psycopg2
newc = psycopg2.connect(user='w',
                        password='K',
                        host='wam',
                        port='5432',
                        database='wa')

anemia_image = pd.read_sql_query('select * from "anemia_image_samples"', con=newc)
anemia_image = anemia_image[anemia_image['sample_id'].astype(str).str.contains("LID")]

anemia_image['sample_date'] = pd.to_datetime(anemia_image['sample_date'], origin='unix', unit='s')
anemia_image['sample_date'] = pd.DatetimeIndex(anemia_image['sample_date']) + timedelta(hours=5, minutes=30)

data = anemia_image[(anemia_image['hb_value'] > 4) & (anemia_image['hb_value'] <=18) ]
data.shape

# Shuffle the DataFrame randomly
np.random.seed(42)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate the number of samples for the first and second sets
first_samples = np.round(0.8 * len(data))
second_samples = len(data) - first_samples

# Assign first or second label based on the first four numbers of sample_id
data['split'] = np.where(data.index < first_samples, 'first', 'second')
data['split'] = np.where((data['hb_value'] < 9) & (data['split'] == "first"), 'anemic_first', data['split'])
data['split'] = np.where((data['hb_value'] >= 9) & (data['split'] == "first"), 'anemic_not_first', data['split'])
data['split'] = np.where((data['hb_value'] < 9) & (data['split'] == "second"), 'anemic_second', data['split'])
data['split'] = np.where((data['hb_value'] >= 9) & (data['split'] == "second"), 'anemic_not_second', data['split'])
data['split'].value_counts()
#data=data.tail(300)
df=data.copy()

today_timestamp = pd.Timestamp.now().timestamp()
max_valid_timestamp = today_timestamp
min_valid_timestamp = pd.Timestamp.now().replace(year=pd.Timestamp.now().year - 110).timestamp()

df = df[(df['date_of_birth'] >= min_valid_timestamp) & 
                                                  (df['date_of_birth'] <= max_valid_timestamp)]

df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], unit='s')
df['age'] = (pd.Timestamp.now() - df['date_of_birth']).dt.days


# Convert Unix timestamp (in seconds) back to datetime
df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], unit='s')


df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], unit='ns', errors='coerce')

df['date_of_birth']= pd.to_datetime(df['date_of_birth'], errors='coerce').dt.date


# Drop missing values
df = df.dropna(subset=['date_of_birth', 'sample_date'])


# Base path for directories
base_path = '/media/sarfraaz/HDD/Sarfraaz_Backup/anemia_research/classification/mobilenet/'

# Eye types and numbers
# Define eye types and numbers
eye_types = ['right_eye','left_eye']
eye_numbers = [ '3','1','2']

# Subdirectories for each eye type
subdirs = [
    'anemic_first', 'anemic_seconds',
    'anemic_not_first', 'anemic_not_seconds',
    'anemic_first_roi', 'anemic_second_rois',
    'anemic_not_first_roi', 'anemic_not_second_rois'
]

def manage_directory(directory):
    """Delete all contents of a directory and recreate it."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    return f"Directory reset and ready for use: {directory}"

# Process all directories
results = []
for eye_type in eye_types:
    for eye_number in eye_numbers:
        for subdir in subdirs:
            path = os.path.join(base_path, f"{eye_type}_{eye_number}", subdir)
            results.append(manage_directory(path))

# Print results
for result in results:
    print(result)

# Function to list folders
def list_folders(directory):
    folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
    return folders

# Base paths
base_directory_path = '/media/sarfraaz/HDD/Sarfraaz_Backup/anemia_research/classification/mobilenet/'
zip_folder_path = "/media/sarfraaz/HDD/Sarfraaz_Backup/anemia_research/downloaded_files/"

# Define eye types and numbers
eye_types = ['right_eye','left_eye']
eye_numbers = [ '3','1','2']

# Create a temporary folder to extract the files
temp_extract_folder = tempfile.mkdtemp(dir='/media/sarfraaz/HDD/Sarfraaz_Backup/anemia_research/')
import os
import zipfile
import shutil

# Loop through the zip files in the folder
for file_name in os.listdir(zip_folder_path):
    if file_name.endswith('.zip'):
        zip_file_path = os.path.join(zip_folder_path, file_name)
        record_id = os.path.splitext(file_name)[0]
        
        try:
            # Check if the record_id exists in the dataframe
            if record_id in data['record_id'].values:
                # Get the corresponding split value from the dataframe
                split = data.loc[data['record_id'] == record_id, 'split'].values[0]
                print(f"Processing: {file_name}")
                
                # Extract the files from the zip file
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_folder)
                
                # Loop through each eye type and number
                for eye_type in eye_types:
                    for eye_number in eye_numbers:
                        # Determine the destination folder based on the split value
                        destination_folder = os.path.join(base_directory_path, f"{eye_type}_{eye_number}", split)
                        if not os.path.exists(destination_folder):
                            os.makedirs(destination_folder)
                        
                        # Move the files to the destination folder if they contain the specific eye type and number
                        pattern = f"{eye_type}_{eye_number}"
                        for extracted_file in os.listdir(temp_extract_folder):
                            if pattern in extracted_file:
                                extracted_file_path = os.path.join(temp_extract_folder, extracted_file)
                                destination_file_path = os.path.join(destination_folder, extracted_file)
                                
                                # Only move the file if it exists in the source directory
                                if os.path.exists(extracted_file_path):
                                    shutil.move(extracted_file_path, destination_file_path)
        except zipfile.BadZipFile:
            print(f"Skipping: {file_name} (not a valid zip file)")
        except Exception as e:
            print(f"An error occurred while processing {file_name}: {e}")

print("Processing complete.")


# Remove the temporary extraction folder
shutil.rmtree(temp_extract_folder)

import os
import cv2
import random
import numpy as np
import pandas as pd
import shutil
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from tensorflow.keras.models import Model

def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def load_images(DIR, RESIZE):
    """Load and resize images from a directory."""
    images = []
    filenames = []
    for IMAGE_NAME in os.listdir(DIR):
        PATH = os.path.join(DIR, IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype.lower() == ".png":
            img = cv2.imread(PATH)
            img = cv2.resize(img, (RESIZE, RESIZE))
            images.append(img)
            filenames.append(IMAGE_NAME)
    return images, filenames

def preprocess_image(img_array):
    """Preprocess image for Grad-CAM input."""
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def crop_eye_region_from_image(image, bbox):
    """Crop the eye region from the image using bounding box."""
    x, y, w, h = bbox
    return image[y:y + h, x:x + w]

def extract_eye_region_from_heatmap(heatmap, threshold=0.5):
    """Extract eye region using Grad-CAM heatmap."""
    heatmap = np.uint8(255 * heatmap)
    _, mask = cv2.threshold(heatmap, int(255 * threshold), 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h), largest_contour
    return None, None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    """Generate Grad-CAM heatmap."""
    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]
    
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(output.shape[0:2], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    
    cam = cv2.resize(cam.numpy(), (img_array.shape[2], img_array.shape[1]))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    
    return heatmap


def region_of_interest(first_folder, second_folder, output_folder, resize_dim):
    """Process and save eye regions as resize_dim x resize_dim images, using Grad-CAM bounding box."""
    
    # Load and resize images
    first_images_raw, _ = load_images(first_folder, 224)
    second_images_raw, filenames_second = load_images(second_folder, 224)
    
    # Use both first and second images for bounding box estimation
    all_images_for_bbox = first_images_raw + second_images_raw
    random.shuffle(all_images_for_bbox)
    
    # Load pre-firsted model
    
    model = MobileNetV2(weights='imagenet')
    
    # Initialize accumulators for bounding box
    total_x, total_y, total_w, total_h = 0, 0, 0, 0
    valid_contours = 0
    
    # Grad-CAM and bounding box averaging
    for image in all_images_for_bbox:
        img_array = preprocess_image(image)
        
        heatmap = make_gradcam_heatmap(img_array, model, 'Conv_1', ["global_average_pooling2d", "Logits", "Predictions"])
        
        bbox, _ = extract_eye_region_from_heatmap(heatmap)
        if bbox:
            x, y, w, h = bbox
            total_x += x
            total_y += y
            total_w += w
            total_h += h
            valid_contours += 1
    
    if valid_contours > 0:
        # Compute average bounding box
        avg_x = total_x // valid_contours
        avg_y = total_y // valid_contours
        avg_w = total_w // valid_contours
        avg_h = total_h // valid_contours
        
        # Expand from all directions (20% margin)
        margin_x = int(avg_w * 0.2)
        margin_y = int(avg_h * 0.2)
        x1 = avg_x - margin_x
        y1 = avg_y - margin_y
        x2 = avg_x + avg_w + margin_x
        y2 = avg_y + avg_h + margin_y
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Apply cropping to second images only (for saving)
        for idx, image in enumerate(second_images_raw):
            h_img, w_img, _ = image.shape
            
            # Clamp within image bounds
            x1_clamped = max(0, x1)
            y1_clamped = max(0, y1)
            x2_clamped = min(w_img, x2)
            y2_clamped = min(h_img, y2)
            
            final_w = x2_clamped - x1_clamped
            final_h = y2_clamped - y1_clamped
            
            # Crop and resize
            eye_region = crop_eye_region_from_image(image, (x1_clamped, y1_clamped, final_w, final_h))
            eye_region_resized = cv2.resize(eye_region, (resize_dim, resize_dim))
            output_path = os.path.join(output_folder, filenames_second[idx])
            cv2.imwrite(output_path, eye_region_resized)


# Parameters
base_path = '/media/sarfraaz/HDD/Sarfraaz_Backup/anemia_research/classification/mobilenet/'
# Define eye types and numbers
eye_types = ['right_eye','left_eye']
eye_numbers = ['3', '1','2']
splits = ['anemic_first', 'anemic_second', 'anemic_not_first', 'anemic_not_second']

for eye_type in eye_types:
    for eye_number in eye_numbers:
        first_input_folder = f"{base_path}{eye_type}_{eye_number}/anemic_not_first/"
        for split in splits:
            first_folder = f"{base_path}{eye_type}_{eye_number}/{split}/"
            output_folder = f"{base_path}{eye_type}_{eye_number}/{split}_roi/"
            region_of_interest(first_folder, first_folder, output_folder, resize_dim=224)

print("Processing complete.")


''' Gemini Qwen FINAL'''
import os
import cv2
import numpy as np
import shutil
import pandas as pd
from datetime import datetime

# ===========================================
# 1. LOAD IMAGES (Safe & Robust)
# ===========================================
def load_images(image_dir, resize=(512, 512)):
    images, filenames = [], []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            path = os.path.join(image_dir, filename)
            try:
                img = cv2.imread(path)
                if img is None:
                    print(f"⚠️ Failed to load: {filename}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, resize)
                images.append(img)
                filenames.append(filename)
            except Exception as e:
                print(f"❌ Error loading {filename}: {str(e)}")
                continue
    return np.array(images) if images else np.array([]), filenames


# ===========================================
# 2. BROWNISH COLOR FILTER (Skin/Noise)
# ===========================================
def filter_brownish_colors(color):
    r, g, b = color
    return (70 <= r <= 150) and (30 <= g <= 110) and (10 <= b <= 90)
def filter_brownish_colors(rgb_color):
    """Detect brown skin."""
    r, g, b = rgb_color
    return (80 <= r <= 150) and (g <= 100) and (b <= 80) and (r > g + 10)

# ===========================================
# 3. UNIFIED CONJUNCTIVA SEGMENTATION (Consistent)
# ===========================================
def segment_conjunctiva_consistent(image):
    """
    Unified segmentation using the best-performing heuristic:
    - HSV: [0, 51, 120] → [10, 255, 255] and [160, 50, 100] → [180, 255, 255]
    - Morphology: k=5
    - Spatial prior: lower-central region
    """
    if image.ndim != 3:
        return np.zeros_like(image), False, 0, (0,0,0)

    h, w = image.shape[:2]
    rgb = image.copy()

    try:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    except:
        return np.zeros_like(image), False, 0, (0,0,0)

    # Define spatial prior: conjunctiva is usually in lower-central
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    #roi_mask[int(2*h/5):, int(w/4):int(3*w/4)] = 255
    roi_mask[int(1.5*h/5):, :] = 255  # Entire lower region, full width
    # Best-performing HSV range (from your H3 heuristic)
    low1, high1 = np.array([0, 51, 120]), np.array([10, 255, 255])
    low2, high2 = np.array([140, 50, 100]), np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, low1, high1)
    mask2 = cv2.inRange(hsv, low2, high2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_and(mask, mask, mask=roi_mask)  # Apply spatial prior

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Final fallback: use central rectangle
        clean_mask = np.zeros((h, w), dtype=np.uint8)
        clean_mask[int(2*h/5):int(2*h/5), int(w/3):int(2*w/3)] = 255
        area = 2000
        avg_color = (140, 100, 90)
        extracted = cv2.bitwise_and(rgb, rgb, mask=clean_mask)
        return extracted, True, area, avg_color

    # Pick largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)

    masked = cv2.bitwise_and(rgb, rgb, mask=clean_mask)
    pixels = masked[masked.sum(axis=2) > 0]

    if len(pixels) == 0:
        return rgb, False, 0, (0,0,0)

    avg_color = tuple(np.mean(pixels, axis=0).astype(int))

    return masked, True, area, avg_color


# ===========================================
# 4. SAVE IMAGE
# ===========================================
def save_extracted(save_dir, name, arr):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    try:
        bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
    except Exception as e:
        print(f"❌ Failed to save {name}: {e}")


# ===========================================
# 5. PROCESS DIRECTORY WITH LOGGING
# ===========================================
def process_dir_with_logging(input_dir, base_save, subfolder, summary_log):
    print(f"🔍 Scanning {input_dir}...")
    images, names = load_images(input_dir)

    if len(images) == 0:
        print(f"❌ No valid images loaded from {input_dir}")
        return 0

    out_dir = os.path.join(base_save, subfolder)
    success_count = 0

    for img, name in zip(images, names):
        print(f"  📷 Processing {name}...")
        extracted, is_success, area, avg_color = segment_conjunctiva_consistent(img)

        # Always save — even if weak
        save_extracted(out_dir, name, extracted)

        log_entry = {
            'filename': name,
            'subfolder': subfolder,
            'success': True,  # Always True now
            'method': 'consistent_H3_spatial',
            'contour_area': area,
            'avg_r': avg_color[0],
            'avg_g': avg_color[1],
            'avg_b': avg_color[2],
            'input_dir': input_dir,
            'saved_path': os.path.join(out_dir, name)
        }
        summary_log.append(log_entry)
        success_count += 1

    return success_count


# ===========================================
# 6. MANAGE OUTPUT DIRECTORY
# ===========================================
def manage_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


# ===========================================
# 7. MAIN EXECUTION
# ===========================================



if __name__ == "__main__":
    # --- CONFIG ---
    base_root = "/media/sarfraaz/HDD/Sarfraaz_Backup/anemia_research/classification"

    # All 6 eyes with 4 subfolders each
    input_dirs = {
        "left_eye_1": {
            "anemic_second_roi": os.path.join(base_root, "left_eye_1/anemic_second_roi/"),
            "anemic_not_second_roi": os.path.join(base_root, "left_eye_1/anemic_not_second_roi/"),
            "anemic_first_roi": os.path.join(base_root, "left_eye_1/anemic_first_roi/"),
            "anemic_not_first_roi": os.path.join(base_root, "left_eye_1/anemic_not_first_roi/")
        },
        "right_eye_1": {
            "anemic_second_roi": os.path.join(base_root, "right_eye_1/anemic_second_roi/"),
            "anemic_not_second_roi": os.path.join(base_root, "right_eye_1/anemic_not_second_roi/"),
            "anemic_first_roi": os.path.join(base_root, "right_eye_1/anemic_first_roi/"),
            "anemic_not_first_roi": os.path.join(base_root, "right_eye_1/anemic_not_first_roi/")
        },
        "left_eye_2": {
            "anemic_second_roi": os.path.join(base_root, "left_eye_2/anemic_second_roi/"),
            "anemic_not_second_roi": os.path.join(base_root, "left_eye_2/anemic_not_second_roi/"),
            "anemic_first_roi": os.path.join(base_root, "left_eye_2/anemic_first_roi/"),
            "anemic_not_first_roi": os.path.join(base_root, "left_eye_2/anemic_not_first_roi/")
        },
        "right_eye_2": {
            "anemic_second_roi": os.path.join(base_root, "right_eye_2/anemic_second_roi/"),
            "anemic_not_second_roi": os.path.join(base_root, "right_eye_2/anemic_not_second_roi/"),
            "anemic_first_roi": os.path.join(base_root, "right_eye_2/anemic_first_roi/"),
            "anemic_not_first_roi": os.path.join(base_root, "right_eye_2/anemic_not_first_roi/")
        },
        "left_eye_3": {
            "anemic_second_roi": os.path.join(base_root, "left_eye_3/anemic_second_roi/"),
            "anemic_not_second_roi": os.path.join(base_root, "left_eye_3/anemic_not_second_roi/"),
            "anemic_first_roi": os.path.join(base_root, "left_eye_3/anemic_first_roi/"),
            "anemic_not_first_roi": os.path.join(base_root, "left_eye_3/anemic_not_first_roi/")
        },
        "right_eye_3": {
            "anemic_second_roi": os.path.join(base_root, "right_eye_3/anemic_second_roi/"),
            "anemic_not_second_roi": os.path.join(base_root, "right_eye_3/anemic_not_second_roi/"),
            "anemic_first_roi": os.path.join(base_root, "right_eye_3/anemic_first_roi/"),
            "anemic_not_first_roi": os.path.join(base_root, "right_eye_3/anemic_not_first_roi/")
        },
    }
    base_root2 = "/media/sarfraaz/HDD/Sarfraaz_Backup/anemia_research/ref"
    # --- OUTPUT ROOT ---
    save_root = os.path.join(base_root2, "mat_conjunctiva_all_consistent")
    manage_directory(save_root)

    summary_log = []

    # Loop over all 6 eyes
    for eye_name, dirs in input_dirs.items():
        print(f"\n=== Processing {eye_name} ===")
        save_base = os.path.join(save_root, eye_name)
        manage_directory(save_base)

        for subfolder, input_dir in dirs.items():
            print(f"\n🚀 Processing {eye_name} → {subfolder}...")
            try:
                success_count = process_dir_with_logging(
                    input_dir, save_base, subfolder, summary_log
                )
                print(f"✅ {success_count} images processed and saved.")
            except Exception as e:
                print(f"💥 Failed to process {eye_name}/{subfolder}: {e}")

    # --- SAVE FINAL SUMMARY ---
    if summary_log:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(save_root, f"segmentation_summary_consistent_{timestamp}.csv")
        df = pd.DataFrame(summary_log)
        df.to_csv(csv_path, index=False)
        print(f"\n📊 Final summary saved to: {csv_path}")

    print("\n=== ALL DONE ===")
    
    
''' after above code, we did manual inspection and removed images where conjuctiva was not extracted '''
''' single eye random image selection approx 500 each eye'''
##### import os
import shutil
import random
from collections import defaultdict, Counter

# ---------------------------------------------------------
# 1. BALANCED IMAGE SELECTION (UNIQUE PER BENEFICIARY)
# ---------------------------------------------------------
def select_balanced_images(source_dirs, output_dir, used_beneficiaries, target_per_eye, seed=72):
    """
    Selects balanced images across 6 eye views while ensuring:
    ✅ Each beneficiary is used only once globally (across all Hb thresholds)
    ✅ Each eye view has ~target_per_eye images
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    suffix_map = {
        'left1': '_left_eye_1.png',
        'left2': '_left_eye_2.png',
        'left3': '_left_eye_3.png',
        'right1': '_right_eye_1.png',
        'right2': '_right_eye_2.png',
        'right3': '_right_eye_3.png',
    }

    # Collect images per beneficiary
    beneficiary_images = defaultdict(dict)
    for key, src_dir in source_dirs.items():
        suffix = suffix_map[key]
        if not os.path.exists(src_dir):
            continue
        for fname in os.listdir(src_dir):
            if fname.endswith(suffix):
                b_id = fname.replace(suffix, "")
                if b_id in used_beneficiaries:
                    continue
                beneficiary_images[b_id][key] = os.path.join(src_dir, fname)

    beneficiaries = list(beneficiary_images.keys())
    random.shuffle(beneficiaries)

    counts = {k: 0 for k in suffix_map.keys()}

    # Pick balanced images
    for b_id in beneficiaries:
        for eye_key in sorted(counts, key=counts.get):  # always choose least-filled eye first
            if eye_key in beneficiary_images[b_id] and counts[eye_key] < target_per_eye:
                src = beneficiary_images[b_id][eye_key]
                shutil.copy(src, os.path.join(output_dir, os.path.basename(src)))
                counts[eye_key] += 1
                used_beneficiaries.add(b_id)
                break

        # Stop early if all targets are filled
        if all(c >= target_per_eye for c in counts.values()):
            break

    print(f"✅ Copied {sum(counts.values())} images → {output_dir}")
    print(f"📊 Per-eye distribution: {counts}")
    return counts

# ---------------------------------------------------------
# 2. ENSURE DIRECTORY EXISTS
# ---------------------------------------------------------
def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

# ---------------------------------------------------------
# 3. TRIM DATASET TO EXACTLY 2990 IMAGES
# ---------------------------------------------------------
def trim_to_target_images(folders, target_total=2990, seed=42):
    random.seed(seed)
    folder_images = {
        f: [os.path.join(f, x) for x in os.listdir(f) if x.endswith(".png")]
        for f in folders
    }
    total_images = sum(len(imgs) for imgs in folder_images.values())

    print(f"📂 Current total images: {total_images}")
    if total_images <= target_total:
        print(f"✅ No trimming needed, already ≤ {target_total}.")
        return

    excess = total_images - target_total
    print(f"⚠️ Need to delete {excess} images to maintain {target_total} total.")

    per_folder_delete = excess // len(folders)
    remaining = excess % len(folders)
    deleted_count = 0

    for i, (folder, imgs) in enumerate(folder_images.items()):
        random.shuffle(imgs)
        num_to_delete = per_folder_delete + (1 if i < remaining else 0)
        for img in imgs[:num_to_delete]:
            os.remove(img)
            deleted_count += 1
        print(f"🗑️ Deleted {num_to_delete} images from: {folder}")

    print(f"✅ Done! Deleted {deleted_count} images total.")
    print(f"🎯 Final dataset size: {target_total}")

# ---------------------------------------------------------
# 4. VERIFY FINAL DATASET DISTRIBUTION
# ---------------------------------------------------------
def verify_image_distribution(base_path, folders):
    patterns = [
        "_left_eye_1.png", "_left_eye_2.png", "_left_eye_3.png",
        "_right_eye_1.png", "_right_eye_2.png", "_right_eye_3.png"
    ]
    image_counts = Counter()
    beneficiary_ids = set()

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        for fname in os.listdir(folder_path):
            for pattern in patterns:
                if fname.endswith(pattern):
                    image_counts[pattern] += 1
            b_id = fname.split("_left_eye_")[0] if "_left_eye_" in fname else fname.split("_right_eye_")[0]
            beneficiary_ids.add(b_id)

    print("\n📊 Final Image Counts by Eye Type:")
    for pattern in patterns:
        print(f"{pattern}: {image_counts[pattern]}")

    left_total = sum(image_counts[p] for p in patterns if "left_eye" in p)
    right_total = sum(image_counts[p] for p in patterns if "right_eye" in p)
    grand_total = left_total + right_total

    print("\n🔹 Final Totals:")
    print(f"  Total LEFT eye images : {left_total}")
    print(f"  Total RIGHT eye images: {right_total}")
    print(f"  Grand Total           : {grand_total}")
    print(f"  Unique Beneficiaries  : {len(beneficiary_ids)}")

# ---------------------------------------------------------
# 5. MAIN EXECUTION (AUTOMATED FOR MULTIPLE Hb THRESHOLDS)
# ---------------------------------------------------------

base_path = "/home/ubuntu/anemia-storage/hb_mobilenet/mat_conjunctiva_all_consistent_deletion/"
hb_thresholds = [    "7_0"] #and other HB thresholds
target_images_total = 3000
target_per_eye = 495  # ~490–498 range

used_beneficiaries = set()

for hb in hb_thresholds:
    print(f"\n🚀 Processing Hb threshold: {hb.replace('_', '.')} g/dL")

    output_dir = os.path.join(base_path, f"hb_{hb}")
    ensure_directory(output_dir)

    output_base = os.path.join(output_dir, "conjunctiva_extracted")
    ensure_directory(output_base)

    # Output folders for this Hb threshold
    output_paths = [
        os.path.join(output_base, 'anemic_train_roi'),
        os.path.join(output_base, 'anemic_not_train_roi'),
        os.path.join(output_base, 'anemic_val_roi'),
        os.path.join(output_base, 'anemic_not_val_roi'),
        os.path.join(output_base, 'anemic_test_roi'),
        os.path.join(output_base, 'anemic_not_test_roi')
    ]
    for path in output_paths:
        ensure_directory(path)

    def make_full_path(subdirs):
        return {k: os.path.join(base_path, v) for k, v in subdirs.items()}

    # Define mappings dynamically for each Hb threshold
    def dirs_for(split_type, anemic=True):
        prefix = "anemic" if anemic else "anemic_not"
        return make_full_path({
            'left1': f"tri_left_eye/left_eye_1_hb_less_than_{hb}/conjunctiva_extracted/{prefix}_{split_type}_roi/",
            'left2': f"tri_left_eye/left_eye_2_hb_less_than_{hb}/conjunctiva_extracted/{prefix}_{split_type}_roi/",
            'left3': f"tri_left_eye/left_eye_3_hb_less_than_{hb}/conjunctiva_extracted/{prefix}_{split_type}_roi/",
            'right1': f"tri_right_eye/right_eye_1_hb_less_than_{hb}/conjunctiva_extracted/{prefix}_{split_type}_roi/",
            'right2': f"tri_right_eye/right_eye_2_hb_less_than_{hb}/conjunctiva_extracted/{prefix}_{split_type}_roi/",
            'right3': f"tri_right_eye/right_eye_3_hb_less_than_{hb}/conjunctiva_extracted/{prefix}_{split_type}_roi/"
        })

    # Run balanced selection for all 6 splits per Hb threshold
    select_balanced_images(dirs_for("train"), os.path.join(output_base, 'anemic_train_roi'), used_beneficiaries, target_per_eye)
    select_balanced_images(dirs_for("train", anemic=False), os.path.join(output_base, 'anemic_not_train_roi'), used_beneficiaries, target_per_eye)
    select_balanced_images(dirs_for("val"), os.path.join(output_base, 'anemic_val_roi'), used_beneficiaries, target_per_eye)
    select_balanced_images(dirs_for("val", anemic=False), os.path.join(output_base, 'anemic_not_val_roi'), used_beneficiaries, target_per_eye)
    select_balanced_images(dirs_for("test"), os.path.join(output_base, 'anemic_test_roi'), used_beneficiaries, target_per_eye)
    select_balanced_images(dirs_for("test", anemic=False), os.path.join(output_base, 'anemic_not_test_roi'), used_beneficiaries, target_per_eye)

    # Trim dataset to exactly 2990 images for this Hb threshold
    trim_to_target_images(output_paths, target_total=target_images_total)

    # Verify distribution per Hb threshold
    verify_image_distribution(output_base, [
        "anemic_train_roi", "anemic_not_train_roi",
        "anemic_val_roi", "anemic_not_val_roi",
        "anemic_test_roi", "anemic_not_test_roi"
    ])



''' associating each HB threshold and its images in proper folder'''


# -*- coding: utf-8 -*-
import os
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import os
import shutil
import pandas as pd
from datetime import timedelta
import psycopg2

# SQL Connection
newc = psycopg2.connect(
    user='wab',
    password='Kha',
    host='wco',
    port='5432',
    database='waa'
)

# Load and prepare metadata
anemia_image = pd.read_sql_query('SELECT * FROM "anemia_image_samples"', con=newc)
anemia_image = anemia_image[anemia_image['sample_id'].astype(str).str.contains("LID")]
anemia_image['sample_date'] = pd.to_datetime(anemia_image['sample_date'], origin='unix', unit='s')
anemia_image['sample_date'] += timedelta(hours=5, minutes=30)


data=anemia_image[(anemia_image['hb_value']>3) & (anemia_image['hb_value']<20)]
data.shape

data=data.sort_values(['sample_date'], ascending=True)

metadata=data.copy()


metadata = metadata[['sample_id', 'sample_date', 'hb_value',
                     'right_eye_1', 'right_eye_2', 'right_eye_3',
                     'left_eye_1', 'left_eye_2', 'left_eye_3']]

# =========================
# CONFIG
# =========================
# If you have a CSV on disk, set this to its path; else set to None to use in-memory `metadata` variable
METADATA_CSV = metadata.copy()  # e.g. "/home/sarfraaz/Pictures/metadata.csv"

# -*- coding: utf-8 -*-
import os
import shutil
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# ASSUMPTION: You already prepared `metadata` like you showed:
#   - filtered hb_value (3..20)
#   - converted sample_date to IST
#   - sorted ascending by sample_date
#   - selected columns: ['sample_id','sample_date','hb_value', eyes...]
# If not, uncomment and adapt your own loading block above this script.
# ------------------------------------------------------------------

# =========================
# CONFIG (time-based split)
# =========================
# Source tree with per-eye subfolders containing the saved images
# -*- coding: utf-8 -*-
import os
import shutil
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# ASSUMPTION: You already prepared `metadata` like earlier:
#   - filtered hb_value (3..20)
#   - converted sample_date to IST
#   - sorted ascending by sample_date
#   - selected columns: ['sample_id','sample_date','hb_value', eyes...]
# If not, run your DB-loading & preprocessing code before this cell.
# ------------------------------------------------------------------

# =========================
# CONFIG (time-based split)
# =========================
SRC_TREE_ROOT = "/home/ubuntu/anemia-storage/hb_mobilenet/mat_conjunctiva_all_consistent_deletion/"

OUT_LEFT_ROOT  = "/home/ubuntu/anemia-storage/hb_mobilenet/mat_conjunctiva_all_consistent_deletion/tri_left_eye"
OUT_RIGHT_ROOT = "/home/ubuntu/anemia-storage/hb_mobilenet/mat_conjunctiva_all_consistent_deletion/tri_right_eye"

EYES = ["left_eye_1", "right_eye_1", "left_eye_2", "right_eye_2", "left_eye_3", "right_eye_3"]

THRESHOLDS = [7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]  # run all
TRAIN_FRAC = 0.75
VAL_FRAC   = 0.125
TEST_FRAC  = 0.125

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")

# sanity: ensure metadata has required columns and global order
need_cols = ["sample_id", "sample_date", "hb_value"] + EYES
missing = [c for c in need_cols if c not in metadata.columns]
if missing:
    raise ValueError(f"Missing columns in metadata: {missing}")

metadata = metadata.sort_values("sample_date", ascending=True).reset_index(drop=True)

# ensure output roots
os.makedirs(OUT_LEFT_ROOT, exist_ok=True)
os.makedirs(OUT_RIGHT_ROOT, exist_ok=True)

# combined summary
all_counts = []

for eye in EYES:
    print(f"\n========== EYE: {eye} ==========")

    # 1) map filename -> path by scanning the entire eye subtree once
    eye_root = os.path.join(SRC_TREE_ROOT, eye)
    if not os.path.isdir(eye_root):
        print(f"⚠️ Eye root not found: {eye_root} — skipping.")
        continue

    filename_to_path = {}
    for root, _, files in os.walk(eye_root):
        for f in files:
            if f.lower().endswith(VALID_EXTS):
                filename_to_path[f] = os.path.join(root, f)

    if not filename_to_path:
        print(f"⚠️ No files found under {eye_root} — skipping.")
        continue

    # 2) subset metadata rows that have a filename for this eye and match suffix
    df = metadata[metadata[eye].notna()].copy()
    if df.empty:
        print(f"⚠️ No filenames listed in metadata for {eye} — skipping.")
        continue

    df[eye] = df[eye].astype(str)
    expected_suffix = f"_{eye}.png"
    suffix_mask = df[eye].str.endswith(expected_suffix, na=False)
    bad_suffix = (~suffix_mask).sum()
    if bad_suffix:
        print(f"⚠️ {bad_suffix} rows do not end with '{expected_suffix}' — skipped.")
    df = df[suffix_mask].copy()
    if df.empty:
        print(f"❌ No rows with filenames ending '{expected_suffix}' for {eye} — skipping.")
        continue

    # 3) map to actual paths found; drop missing
    df["src_path"] = df[eye].map(filename_to_path)
    miss = df["src_path"].isna().sum()
    if miss:
        print(f"⚠️ {miss} filenames not found under {eye_root} — skipped.")
    df = df[df["src_path"].notna()].copy()
    if df.empty:
        print(f"❌ After mapping, nothing left to process for {eye}.")
        continue

    # keep chronological order within this eye too
    df = df.sort_values("sample_date", ascending=True).reset_index(drop=True)

    # 4) compute fixed sequential splits (positions) ONCE; these are time-based
    n = len(df)
    n_train = int(np.round(TRAIN_FRAC * n))
    n_val   = int(np.round(VAL_FRAC   * n))
    n_test  = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val  = n - n_train

    train_pos = list(range(0, n_train))
    val_pos   = list(range(n_train, n_train + n_val))
    test_pos  = list(range(n_train + n_val, n))

    # 5) now iterate over thresholds; labels change, splits stay
    for thr in THRESHOLDS:
        print(f"--- Threshold: {thr} ---")

        # label for this threshold
        df["label"] = (df["hb_value"] < float(thr)).astype(int)

        # decide output root (left vs right) and build tag/folders
        out_root = OUT_LEFT_ROOT if eye.startswith("left") else OUT_RIGHT_ROOT
        tag = f"{eye}_hb_less_than_{str(thr).replace('.', '_')}"
        split_root = os.path.join(out_root, tag, "conjunctiva_extracted")

        for d in [
            "anemic_train_roi", "anemic_val_roi", "anemic_test_roi",
            "anemic_not_train_roi", "anemic_not_val_roi", "anemic_not_test_roi"
        ]:
            os.makedirs(os.path.join(split_root, d), exist_ok=True)

        # COPY: train
        part = df.iloc[train_pos]
        for _, row in part.iterrows():
            src = row["src_path"]
            dst_dir = os.path.join(split_root, "anemic_train_roi" if row["label"] == 1 else "anemic_not_train_roi")
            dst = os.path.join(dst_dir, os.path.basename(src))
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"❌ Copy failed (train): {src} -> {dst} | {e}")

        # COPY: val
        part = df.iloc[val_pos]
        for _, row in part.iterrows():
            src = row["src_path"]
            dst_dir = os.path.join(split_root, "anemic_val_roi" if row["label"] == 1 else "anemic_not_val_roi")
            dst = os.path.join(dst_dir, os.path.basename(src))
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"❌ Copy failed (val): {src} -> {dst} | {e}")

        # COPY: test
        part = df.iloc[test_pos]
        for _, row in part.iterrows():
            src = row["src_path"]
            dst_dir = os.path.join(split_root, "anemic_test_roi" if row["label"] == 1 else "anemic_not_test_roi")
            dst = os.path.join(dst_dir, os.path.basename(src))
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"❌ Copy failed (test): {src} -> {dst} | {e}")

        # stats for this threshold
        train_df = df.iloc[train_pos]
        val_df   = df.iloc[val_pos]
        test_df  = df.iloc[test_pos]

        t_total = len(train_df); t_pos = int((train_df["label"]==1).sum()); t_neg = t_total - t_pos
        v_total = len(val_df);   v_pos = int((val_df["label"]==1).sum());   v_neg = v_total - v_pos
        s_total = len(test_df);  s_pos = int((test_df["label"]==1).sum());  s_neg = s_total - s_pos

        print(f"Train(seq) total={t_total}, anemic={t_pos}, non-anemic={t_neg}")
        print(f"Val  (seq) total={v_total}, anemic={v_pos}, non-anemic={v_neg}")
        print(f"Test (seq) total={s_total}, anemic={s_pos}, non-anemic={s_neg}")
        print(f"📁 Saved to: {split_root}")

        all_counts.append({
            "eye": eye, "threshold": thr,
            "train_total": t_total, "train_anemic": t_pos, "train_non_anemic": t_neg,
            "val_total": v_total,   "val_anemic": v_pos,   "val_non_anemic": v_neg,
            "test_total": s_total,  "test_anemic": s_pos,  "test_non_anemic": s_neg,
            "output_root": split_root
        })

# write combined summary
if all_counts:
    summary_df = pd.DataFrame(all_counts)
    summary_csv = os.path.join(OUT_RIGHT_ROOT, "tri_split_summary_all_eyes_all_thresholds_timebased.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✅ Combined sequential-split summary saved at: {summary_csv}")
else:
    print("\n⚠️ No splits were created (check mapping/suffixes/paths).")




