import os
import re
import time
from statistics import mean
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms
import numpy as np

ORIG_ROOT = "images"
OUT_DIR = "test_visualization_three_heads"
MODEL_PATH = "best_model_advanced.pth"

try:
    from train_advanced import AdvancedBoxCounter
except ImportError as e:
    print("Не удалось импортировать ThreeHeadBoxCounter из train_advanced.py. Поместите файл рядом и повторите.")
    raise e

DEVICE = torch.device('cpu')

CROP_BOX = (407, 40, 407 + 1060, 40 + 1060)
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

try:
    DEFAULT_FONT = ImageFont.truetype("arial.ttf", 20)
except:
    DEFAULT_FONT = ImageFont.load_default()

PALLET_DIR_RE = re.compile(r'^pallet_(\d+)_(\d+)_(\d+)$', re.IGNORECASE)


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


def crop_and_prepare(path: str) -> Image.Image:
    img = Image.open(path)
    img = ensure_rgb_no_alpha(img)
    img = img.crop(CROP_BOX)
    return img.convert('RGB')


def detect_view_key(stem: str):
    low = stem.lower()
    for key in ['left', 'right', 'front', 'back', 'top']:
        if re.search(r'(?:^|[_-])' + key + r'(?:_|-|$)', low):
            return key
    m = re.match(r'^(.*?)(?:_(\d+))?$', stem)
    return m.group(1).lower() if m else low


def group_images_by_view(img_paths):
    groups = defaultdict(list)
    for p in img_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        key = detect_view_key(stem)
        groups[key].append(p)
    return groups


def forward_pair_cpu(model, img1: Image.Image, img2: Image.Image):
    t1 = transform(img1).unsqueeze(0).to(DEVICE)
    t2 = transform(img2).unsqueeze(0).to(DEVICE)
    t0 = time.perf_counter()

    with torch.no_grad():
        pred = model(t1, t2).cpu().numpy()[0]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Возвращаем только 3 головы + вычисляем total как сумму
    return {
        'l': int(max(0, round(pred[0]))),
        't': int(max(0, round(pred[1]))),
        'g': int(max(0, round(pred[2]))),
        'total': int(max(0, round(pred[0] + pred[1] + pred[2])))
    }, elapsed_ms


def annotate_pair(image1, image2, gt, pred, save_path,
                  elapsed_ms=None, subtitle=None):
    w1, h1 = image1.size
    w2, h2 = image2.size
    pad = 10
    title_h = 120

    W = w1 + w2 + pad * 3
    H = max(h1, h2) + title_h + pad * 3

    canvas = Image.new('RGB', (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    time_line = f" | time: {elapsed_ms:.1f} ms (CPU)" if elapsed_ms is not None else ""
    title = (f"GT L:{gt['l']} T:{gt['t']} G:{gt['g']} Total:{gt['total']} | "
             f"Pred L:{pred['l']} T:{pred['t']} G:{pred['g']} Total:{pred['total']}{time_line}")

    draw.text((pad, pad), title, fill=(20, 20, 20), font=DEFAULT_FONT)

    if subtitle:
        draw.text((pad, pad + 28), subtitle, fill=(60, 60, 60), font=DEFAULT_FONT)

    draw.text((pad, title_h - 20), 'View A (Left)', fill=(0, 0, 0), font=DEFAULT_FONT)
    draw.text((w1 + pad * 2, title_h - 20), 'View B (Right)', fill=(0, 0, 0), font=DEFAULT_FONT)

    canvas.paste(image1, (pad, title_h))
    canvas.paste(image2, (w1 + pad * 2, title_h))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.save(save_path, quality=95)


def accuracy_at_k(y_true, y_pred, k=1):
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    return float(np.mean(np.abs(y_true - y_pred) <= k))


def compute_metrics(gt_list, pr_list, name):
    if not gt_list:
        return f"{name}: нет данных"

    gt = np.array(gt_list)
    pr = np.array(pr_list)
    err = pr - gt

    mae = np.mean(np.abs(err))
    mse = np.mean(err ** 2)
    rmse = np.sqrt(mse)
    acc = np.mean(pr == gt)
    acc0 = accuracy_at_k(gt, pr, 0)
    acc1 = accuracy_at_k(gt, pr, 1)
    acc2 = accuracy_at_k(gt, pr, 2)

    return f"{name}: MAE={mae:.3f}, MSE={mse:.3f}, RMSE={rmse:.3f}, ACC={acc:.3f}, ACC0={acc0:.3f}, ACC1={acc1:.3f}, ACC2={acc2:.3f}"


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Не найдена модель {MODEL_PATH}. Сначала обучите модель с 3 головами.")
        return

    model = AdvancedBoxCounter(backbone='efficientnet', use_attention=True).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    pallet_dirs = []
    for name in os.listdir(ORIG_ROOT):
        full = os.path.join(ORIG_ROOT, name)
        if os.path.isdir(full) and PALLET_DIR_RE.match(name):
            pallet_dirs.append(full)

    print(f"[INFO] Найдено паллет: {len(pallet_dirs)}")

    saved = 0
    times = []

    gts_l, gts_t, gts_g, gts_total = [], [], [], []
    prs_l, prs_t, prs_g, prs_total = [], [], [], []

    for pdir in pallet_dirs:
        pname = os.path.basename(pdir)
        m = PALLET_DIR_RE.match(pname)
        if not m:
            continue

        L, T, G = int(m.group(1)), int(m.group(2)), int(m.group(3))
        total_gt = L + T + G
        gt = {'l': L, 't': T, 'g': G, 'total': total_gt}

        files = [os.path.join(pdir, f) for f in os.listdir(pdir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(files) < 2:
            continue

        by_view = group_images_by_view(files)
        pairs = []

        for view, paths in by_view.items():
            if len(paths) >= 2:
                p_sorted = sorted(paths)
                for i in range(len(p_sorted) - 1):
                    for j in range(i + 1, len(p_sorted)):
                        pairs.append((p_sorted[i], p_sorted[j]))

        if not pairs:
            allp = sorted(files)
            for i in range(len(allp) - 1):
                for j in range(i + 1, len(allp)):
                    pairs.append((allp[i], allp[j]))

        if not pairs:
            continue

        path1, path2 = pairs[0]

        # Детерминированный порядок: всегда алфавитный
        if path1 > path2:
            path1, path2 = path2, path1

        try:
            img1 = crop_and_prepare(path1)
            img2 = crop_and_prepare(path2)
        except Exception as e:
            print(f"[WARN] Ошибка обработки {path1}, {path2}: {e}")
            continue

        pred, tm = forward_pair_cpu(model, img1, img2)
        times.append(tm)

        # Сохраняем для метрик
        gts_l.append(gt['l']);
        prs_l.append(pred['l'])
        gts_t.append(gt['t']);
        prs_t.append(pred['t'])
        gts_g.append(gt['g']);
        prs_g.append(pred['g'])
        gts_total.append(gt['total']);
        prs_total.append(pred['total'])

        subtitle = f"Pallet: {pname}"
        save_path = os.path.join(OUT_DIR, f"{pname}.png")
        annotate_pair(img1, img2, gt, pred, save_path, elapsed_ms=tm, subtitle=subtitle)
        saved += 1

    avg_time = mean(times) if times else 0.0
    print(f"Готово. Сохранено: {saved} визуализаций. Среднее время: {avg_time:.1f} ms")

    # Выводим метрики для каждой головы отдельно
    print(compute_metrics(gts_l, prs_l, "Laptop"))
    print(compute_metrics(gts_t, prs_t, "Tablet"))
    print(compute_metrics(gts_g, prs_g, "Group"))
    print(compute_metrics(gts_total, prs_total, "Total"))


if __name__ == "__main__":
    main()
