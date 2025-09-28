import os
import shutil
from glob import glob
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import math


CROP_BOX = (407, 40, 407 + 1060, 40 + 1060)
RANDOM_SEED = 42
IMG_SIZE = (224, 224)
AUG_MULTIPLIER = 2


folders = {
    'original': 'images',
    'generated': 'box_dataset'
}

unified_dir = 'unified_dataset'

def ensure_rgb_no_alpha(img: Image.Image) -> Image.Image:
    if img.mode in ('RGBA', 'LA'):
        bg = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'RGBA':
            bg.paste(img, mask=img.split()[3])
        else:
            tmp = img.convert('RGBA')
            bg.paste(tmp, mask=tmp.split()[3])
        return bg
    if img.mode == 'P':
        return img.convert('RGB')
    return img.convert('RGB')

def crop_resize(image: Image.Image) -> Image.Image:
    cropped = image.crop(CROP_BOX)
    resized = cropped.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    return resized.convert('RGB')

def crop_resize_save(image_path: str, output_path: str) -> bool:
    try:
        with Image.open(image_path) as img:
            img = ensure_rgb_no_alpha(img)
            out = crop_resize(img)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out.save(output_path, format='JPEG', quality=95)
            return True
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        return False

def parse_pallet_name(pallet_name: str):
    try:
        parts = pallet_name.split('_')
        if len(parts) == 4 and parts[0] == 'pallet':
            return {
                'laptop_count': int(parts[1]),
                'tablet_count': int(parts[2]),
                'group_box_count': int(parts[3])
            }
    except:
        pass
    return None

def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)

def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)

def color_temperature(img: Image.Image, factor: float) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    arr[..., 0] = np.clip(arr[..., 0] * factor, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] / factor, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

def add_soft_shadow(img: Image.Image, strength: float = 0.1, direction: str = 'bottom') -> Image.Image:
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[0], arr.shape[1]
    if direction == 'bottom':
        mask = np.linspace(1.0, 1.0 - strength, h)[:, None]
        mask = np.tile(mask, (1, w))
    elif direction == 'top':
        mask = np.linspace(1.0 - strength, 1.0, h)[:, None]
        mask = np.tile(mask, (1, w))
    elif direction == 'right':
        mask = np.linspace(1.0, 1.0 - strength, w)[None, :]
        mask = np.tile(mask, (h, 1))
    else:  # left
        mask = np.linspace(1.0 - strength, 1.0, w)[None, :]
        mask = np.tile(mask, (h, 1))
    if arr.ndim == 3:
        mask = mask[..., None]
    shaded = arr * mask
    shaded = np.clip(shaded, 0, 255)
    return Image.fromarray(shaded.astype(np.uint8))

def slight_rotate(img: Image.Image, degrees: float) -> Image.Image:
    return img.rotate(degrees, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(255, 255, 255))

def slight_perspective(img: Image.Image, dx: int = 5, dy: int = 5) -> Image.Image:
    w, h = img.size
    dst = [(0+dx,0+dy), (w-dx,0+dy), (w-dx,h-dy), (0+dx,h-dy)]
    return img.transform(img.size, Image.Transform.QUAD, data=sum(dst, ()), resample=Image.Resampling.BICUBIC)

def add_gaussian_noise(img: Image.Image, std: float = 3.0) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def augment_presets(img: Image.Image):
    variants = []

    v1 = adjust_brightness(img, 1.10)
    v1 = adjust_contrast(v1, 1.05)
    variants.append(v1)

    v2 = color_temperature(img, 1.06)
    variants.append(v2)

    v3 = add_soft_shadow(img, strength=0.08, direction='bottom')
    v3 = slight_rotate(v3, degrees=2)
    variants.append(v3)

    v4 = slight_perspective(img, dx=4, dy=4)
    v4 = add_gaussian_noise(v4, std=2.0)
    variants.append(v4)
    return variants

def save_image(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.convert('RGB').save(path, format='JPEG', quality=95)


def process_dataset():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if os.path.exists(unified_dir):
        shutil.rmtree(unified_dir)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(unified_dir, split), exist_ok=True)

    all_pallets = []
    original_pallets = []

    for folder_type, folder_path in folders.items():
        if not os.path.exists(folder_path):
            print(f"Предупреждение: папка {folder_path} не существует")
            continue
        pallet_dirs = glob(os.path.join(folder_path, "pallet_*_*_*"))
        for pallet_dir in pallet_dirs:
            pallet_name = os.path.basename(pallet_dir)
            target_info = parse_pallet_name(pallet_name)
            if target_info is None:
                continue
            image_files = glob(os.path.join(pallet_dir, "*.jpg")) \
                        + glob(os.path.join(pallet_dir, "*.jpeg")) \
                        + glob(os.path.join(pallet_dir, "*.png"))
            if not image_files:
                continue
            pallet_info = {
                'pallet_name': pallet_name,
                'folder_type': folder_type,
                'original_path': pallet_dir,
                'images': sorted(image_files),
                **target_info
            }
            all_pallets.append(pallet_info)
            if folder_type == 'original':
                original_pallets.append(pallet_info)

    print(f"Найдено паллет: всего {len(all_pallets)}, оригинальных {len(original_pallets)}")

    candidates = list(all_pallets)
    random.shuffle(candidates)

    n_total = len(candidates)
    n_train_base = max(1, int(0.9 * n_total))
    train_base = candidates[:n_train_base]
    val_base = candidates[n_train_base:]

    splits = {'train': list(train_base), 'val': list(val_base)}

    generated = {'train': [], 'val': []}

    for split_name, split_data in splits.items():
        rows = []
        out_dir = os.path.join(unified_dir, split_name)

        for pallet_info in split_data:
            pallet_name = pallet_info['pallet_name']
            counts = (pallet_info['laptop_count'],
                      pallet_info['tablet_count'],
                      pallet_info['group_box_count'])

            for idx, image_path in enumerate(pallet_info['images']):
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                out_name = f"{pallet_name}_{base_name}_{idx}_orig.jpg"
                out_path = os.path.join(out_dir, out_name)
                ok = crop_resize_save(image_path, out_path)
                if ok:
                    rows.append({
                        'filename': out_name,
                        'pallet_name': pallet_name,
                        'laptop_count': counts[0],
                        'tablet_count': counts[1],
                        'group_box_count': counts[2],
                        'folder_type': pallet_info['folder_type'],
                        'original_path': image_path
                    })
                    generated[split_name].append(out_path)
                else:
                    continue

                try:
                    with Image.open(out_path) as base_img:
                        base_img = base_img.convert('RGB')
                        variants = augment_presets(base_img)
                        for aug_i, vimg in enumerate(variants):
                            aug_name = f"{pallet_name}_{base_name}_{idx}_aug{aug_i+1}.jpg"
                            aug_path = os.path.join(out_dir, aug_name)
                            save_image(vimg, aug_path)
                            rows.append({
                                'filename': aug_name,
                                'pallet_name': pallet_name,
                                'laptop_count': counts[0],
                                'tablet_count': counts[1],
                                'group_box_count': counts[2],
                                'folder_type': pallet_info['folder_type'],
                                'original_path': image_path
                            })
                            generated[split_name].append(aug_path)
                except Exception as e:
                    print(f"Ошибка при аугментации {out_path}: {e}")

        pd.DataFrame(rows).to_csv(os.path.join(unified_dir, f'{split_name}_metadata.csv'), index=False)
        print(f"{split_name}: {len(rows)} изображений")

    train_files = generated['train']
    random.shuffle(train_files)
    n_test = max(1, int(0.10 * len(train_files)))
    test_pick = train_files[:n_test]

    test_rows = []
    test_out_dir = os.path.join(unified_dir, 'test')
    os.makedirs(test_out_dir, exist_ok=True)

    train_meta = pd.read_csv(os.path.join(unified_dir, 'train_metadata.csv'))

    meta_map = {row['filename']: row for _, row in train_meta.iterrows()}

    for src_path in test_pick:
        src_name = os.path.basename(src_path)
        dst_name = src_name
        dst_path = os.path.join(test_out_dir, dst_name)
        shutil.copy2(src_path, dst_path)

        row = meta_map.get(src_name, None)
        if row is not None:
            test_rows.append(dict(row))
        else:
            test_rows.append({
                'filename': dst_name,
                'pallet_name': 'unknown',
                'laptop_count': np.nan,
                'tablet_count': np.nan,
                'group_box_count': np.nan,
                'folder_type': 'duplicated_from_train',
                'original_path': 'unknown'
            })

    pd.DataFrame(test_rows).to_csv(os.path.join(unified_dir, 'test_metadata.csv'), index=False)
    print(f"test: {len(test_rows)} изображений (дубликаты из train ~10%)")

    print(f"Датасет обработан и сохранен в {unified_dir}")
    return True

if __name__ == "__main__":
    process_dataset()

