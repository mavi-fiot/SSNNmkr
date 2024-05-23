import albumentations as A
import cv2
import os

transform = A.Compose([
    A.HorizontalFlip(p=0.8),
    A.RandomBrightnessContrast(p=0.7),
    A.Rotate(p=1)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def augment_and_save_image(image_path, bboxes, classes):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    augmented = transform(image=image, bboxes=bboxes, class_labels=classes)
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']
    augmented_classes = augmented['class_labels']

    folder, filename = os.path.split(image_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_aug{ext}"
    new_image_path = os.path.join(folder, new_filename)

    cv2.imwrite(new_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

    new_label_filename = f"{name}_aug.txt"
    new_label_path = os.path.join(folder.replace('images', 'labels'), new_label_filename)

    with open(new_label_path, 'w') as label_file:
        for cls, bbox in zip(augmented_classes, augmented_bboxes):
            label_file.write(f"{cls} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

def read_yolo_labels(label_path):
    bboxes = []
    classes = []
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])
            bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
            classes.append(cls)
            bboxes.append(bbox)
    return bboxes, classes

def process_images_in_directory(input_image_dir, input_label_dir):
    for root, dirs, files in os.walk(input_image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                label_path = os.path.join(input_label_dir, file.replace(os.path.splitext(file)[1], '.txt'))
                bboxes, classes = read_yolo_labels(label_path)
                augment_and_save_image(image_path, bboxes, classes)

train_dataset_path = 'data/train/images'
train_label_dataset_path = 'data/train/labels'
process_images_in_directory(train_dataset_path, train_label_dataset_path)
