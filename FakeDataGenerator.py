import cv2
import numpy as np
import os
import random
from PIL import Image, ImageEnhance
from glob import glob
import shutil
from sklearn.model_selection import train_test_split

def make_background_transparent(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_pil = image_pil.convert("RGBA")
    newData = []
    for item in image_pil.getdata():
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    image_pil.putdata(newData)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGBA2BGRA)

def adjust_cursor_color(image, brightness_factor, saturation_factor):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    enhancer = ImageEnhance.Color(image_pil)
    image_pil = enhancer.enhance(saturation_factor)
    enhancer = ImageEnhance.Brightness(image_pil)
    image_pil = enhancer.enhance(brightness_factor)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGBA2BGRA)

def update_progress(progress):
    bar_length = 50
    block = int(round(bar_length * progress))
    text = "\rProgress: [{0}] {1:.1f}%".format("#" * block + "-" * (bar_length - block), progress * 100)
    print(text, end="", flush=True)

def prepare_directories(base_dir, dirs):
    for dir_name in dirs:
        path = os.path.join(base_dir, dir_name)
        if not os.path.exists(path):
            os.makedirs(path)

video_path = input("Enter the name of your background video: ")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Failed to open video: {video_path}")
    exit()

scale_variation = input("Vary the scale of the cursors? Y/N: ").upper() == 'Y'
if scale_variation:
    lower_bound_scale = float(input("Enter the lower bound for scale variation (default 0.5): ") or "0.5")
    upper_bound_scale = float(input("Enter the upper bound for scale variation (default 1.5): ") or "1.5")

color_variation = input("Vary the color of the cursors? Y/N: ").upper() == 'Y'
if color_variation:
    lower_bound_color = float(input("Enter the lower bound for color variation factor (default 0.9): ") or "0.9")
    upper_bound_color = float(input("Enter the upper bound for color variation factor (default 1.1): ") or "1.1")

output_folder = 'labelled_data'
train_images_dir = os.path.join(output_folder, 'images', 'train')
val_images_dir = os.path.join(output_folder, 'images', 'val')
train_labels_dir = os.path.join(output_folder, 'labels', 'train')
val_labels_dir = os.path.join(output_folder, 'labels', 'val')

prepare_directories('', [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir])

cursor_templates_path = 'cursor_templates/*.png'
cursor_files = glob(cursor_templates_path)
cursor_templates = []
class_names = []

for idx, file in enumerate(cursor_files):
    class_name = os.path.basename(file).split('.')[0]
    class_names.append(class_name)
    cursor_templates.append((make_background_transparent(cv2.imread(file, cv2.IMREAD_UNCHANGED)), idx))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_paths = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cursor, class_id = random.choice(cursor_templates)

    if scale_variation:
        scale_factor = random.uniform(lower_bound_scale, upper_bound_scale)
        cursor = cv2.resize(cursor, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    if color_variation:
        brightness_factor = random.uniform(lower_bound_color, upper_bound_color)
        saturation_factor = random.uniform(lower_bound_color, upper_bound_color)
        cursor = adjust_cursor_color(cursor, brightness_factor, saturation_factor)

    y_offset = random.randint(0, frame.shape[0] - cursor.shape[0])
    x_offset = random.randint(0, frame.shape[1] - cursor.shape[1])

    for c in range(0, 3):
        frame[y_offset:y_offset+cursor.shape[0], x_offset:x_offset+cursor.shape[1], c] = (
            cursor[:, :, 3] / 255.0 * cursor[:, :, c] +
            (1 - cursor[:, :, 3] / 255.0) * frame[y_offset:y_offset+cursor.shape[0], x_offset:x_offset+cursor.shape[1], c])

    modified_frame_path = os.path.join(train_images_dir, f"frame_{frame_count:05d}.png")
    cv2.imwrite(modified_frame_path, frame)
    frame_paths.append(modified_frame_path)

    label_path = os.path.join(train_labels_dir, f"frame_{frame_count:05d}.txt")
    with open(label_path, 'w') as label_file:
        center_x = (x_offset + cursor.shape[1] / 2) / frame.shape[1]
        center_y = (y_offset + cursor.shape[0] / 2) / frame.shape[0]
        width = cursor.shape[1] / frame.shape[1]
        height = cursor.shape[0] / frame.shape[0]
        label_file.write(f"{class_id} {center_x} {center_y} {width} {height}\n")

    frame_count += 1
    update_progress(frame_count / total_frames)

cap.release()

if frame_paths:
    train_paths, val_paths = train_test_split(frame_paths, test_size=0.2, random_state=42)

    for path in val_paths:
        shutil.move(path, os.path.join(val_images_dir, os.path.basename(path)))
        label_path = path.replace('.png', '.txt').replace(train_images_dir, train_labels_dir)
        shutil.move(label_path, os.path.join(val_labels_dir, os.path.basename(label_path)))

    dataset_yaml = 'dataset.yaml'
    with open(os.path.join(output_folder, dataset_yaml), 'w') as yaml_file:
        yaml_content = f"""path: ../{output_folder}
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""
        yaml_file.write(yaml_content)

    print("\nData generation complete.")
else:
    print("No frames were processed. Please check the video file and try again.")