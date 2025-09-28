import os
import csv
from PIL import Image
import torch
import torchvision.transforms as transforms
import time
import numpy as np
import onnxruntime as ort


IMAGES_ROOT = "images"
RESULT_CSV  = "results/result.csv"
MODEL_PATH  = "models/best_model_advanced.pth"
ONNX_PATH = "models/best_model_advanced.onnx"

try:
    from train_advanced import AdvancedBoxCounter
except ImportError as e:
    print("Не удалось импортировать AdvancedBoxCounter из train_advanced.py. Поместите файл рядом и повторите.")
    raise e

DEVICE = torch.device('cpu')

CROP_BOX = (407, 40, 407 + 1060, 40 + 1060)
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

def load_model(path):
    model = AdvancedBoxCounter(backbone='efficientnet', use_attention=True).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

def prepare(path):
    img = Image.open(path).convert('RGB')
    return img.crop(CROP_BOX)

def predict_pair(model, img1, img2):
    t1 = transform(img1).unsqueeze(0).to(DEVICE)
    t2 = transform(img2).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(t1, t2).cpu().numpy()[0]
    return np.round(np.clip(out, 0, None)).astype(int)

def order_pair(img_list):
    left_idxs = [i for i,f in enumerate(img_list) if 'left' in f.lower()]
    if left_idxs:
        i_left = left_idxs[0]
        i_other = 1 - i_left
        return [img_list[i_left], img_list[i_other]]
    return sorted(img_list)

def preprocess_np(img_pil):
    img = img_pil.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0  # HWC, [0,1]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    arr = np.expand_dims(arr, 0)        # NCHW
    return arr

def predict_pair_onnx(onnx_path, img1, img2):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inpA = preprocess_np(img1)
    inpB = preprocess_np(img2)
    inputs = {
        sess.get_inputs()[0].name: inpA.astype(np.float32),
        sess.get_inputs()[1].name: inpB.astype(np.float32),
    }
    out_name = sess.get_outputs()[0].name
    out = sess.run([out_name], inputs)[0][0]
    return np.round(np.clip(out, 0, None)).astype(int)


def export_to_onnx(model, onnx_path=ONNX_PATH):
    model.eval()
    # dummy входы соответствуют вашему пайпу: две картинки RGB 224x224
    dummyA = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    dummyB = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    torch.onnx.export(
        model,
        (dummyA, dummyB),
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["inputA", "inputB"],
        output_names=["counts"],
        dynamic_axes={
            "inputA": {0: "batch"},
            "inputB": {0: "batch"},
            "counts": {0: "batch"},
        },
    )
    print(f"ONNX модель сохранена в {onnx_path}")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Модель {MODEL_PATH} не найдена. Обучите модель и поместите файл здесь.")
        return

    use_onnx = os.getenv("USE_ONNX", "0") == "1"
    do_export = os.getenv("EXPORT_ONNX", "0") == "1"

    model = load_model(MODEL_PATH)

    if do_export:
        export_to_onnx(model, ONNX_PATH)

    rows = []
    infer_times = []
    prep_times = []

    t0_total = time.perf_counter()
    for pallet_dir in os.listdir(IMAGES_ROOT):
        full_dir = os.path.join(IMAGES_ROOT, pallet_dir)
        if not os.path.isdir(full_dir):
            continue
        imgs = [f for f in os.listdir(full_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if len(imgs) != 2:
            print(f"Не совпадает количество изображений в папке {pallet_dir}")
            continue

        viewA, viewB = order_pair(imgs)

        t0_prep = time.perf_counter()
        imgA = prepare(os.path.join(full_dir, viewA))
        imgB = prepare(os.path.join(full_dir, viewB))
        prep_times.append(time.perf_counter() - t0_prep)

        t0_inf = time.perf_counter()
        if use_onnx:
            pred = predict_pair_onnx(ONNX_PATH, imgA, imgB)
        else:
            pred = predict_pair(model, imgA, imgB)
        infer_times.append(time.perf_counter() - t0_inf)

        rows.append([pallet_dir, int(pred[0]), int(pred[1]), int(pred[2])])

    total_sec = time.perf_counter() - t0_total

    with open(RESULT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dir_name', 'laptop', 'tablet', 'group_box'])
        writer.writerows(rows)

    # Сводка по времени
    def fmt(s): return f"{s*1000:.1f} ms"
    # if infer_times:
    #     print(f"Infer: mean={fmt(stats.mean(infer_times))}, median={fmt(stats.median(infer_times))}, n={len(infer_times)}")
    # if prep_times:
    #     print(f"Prep:  mean={fmt(stats.mean(prep_times))}, median={fmt(stats.median(prep_times))}, n={len(prep_times)}")
    # print(f"Total: {total_sec:.3f} s")

    print(f"Готово. Результаты сохранены в {RESULT_CSV}")

if __name__ == "__main__":
    main()