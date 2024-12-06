from pathlib import Path
import shutil

train_txt = Path('/data1/DATA_126/hqj/MAR20/ImageSets/Main/train.txt')
val_txt = Path('/data1/DATA_126/hqj/MAR20/ImageSets/Main/test.txt')
img_dir = Path('/data1/DATA_126/hqj/MAR20/JPEGImages/')
label_dir = Path('/data1/DATA_126/hqj/MAR20/Annotations/hbb/')
train_save_image_dir = Path('/data1/DATA_126/hqj/MAR20/train/images/')
train_save_label_dir = Path('/data1/DATA_126/hqj/MAR20/train/voc/')
val_save_image_dir = Path('/data1/DATA_126/hqj/MAR20/val/images/')
val_save_label_dir = Path('/data1/DATA_126/hqj/MAR20/val/voc/')

with open(train_txt, 'r') as f:
    train_id = f.readlines()

with open(val_txt, 'r') as f:
    val_id = f.readlines()

train_id = [i.strip() for i in train_id]
val_id = [i.strip() for i in val_id]

# for train
for i in train_id:
    img_path = img_dir / f'{i}.jpg'
    label_path = label_dir / f'{i}.xml'
    save_img_path = train_save_image_dir / f'{i}.jpg'
    save_label_path = train_save_label_dir / f'{i}.xml'
    # copy、
    shutil.copyfile(img_path, save_img_path)
    shutil.copyfile(label_path, save_label_path)

# for val
for i in val_id:
    img_path = img_dir / f'{i}.jpg'
    label_path = label_dir / f'{i}.xml'
    save_img_path = val_save_image_dir / f'{i}.jpg'
    save_label_path = val_save_label_dir / f'{i}.xml'
    # copy、
    shutil.copyfile(img_path, save_img_path)
    shutil.copyfile(label_path, save_label_path)


