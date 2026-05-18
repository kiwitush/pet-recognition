"""
推理与测试集评估
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

from config import Config
from dataset import get_breed_names, get_transforms


def evaluate(model: nn.Module, dataloader: DataLoader, device: str) -> dict:
    """
    在给定 dataloader 上评估模型, 返回 acc 和 per-class acc
    """
    model.eval()
    model.to(device)

    correct, total = 0, 0
    num_classes = len(get_breed_names())
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for p, lab in zip(preds, labels):
                class_correct[lab] += (p == lab).item()
                class_total[lab] += 1

    breed_names = get_breed_names()
    per_class_acc = {
        breed_names[i]: (class_correct[i] / class_total[i]).item()
        for i in range(num_classes) if class_total[i] > 0
    }

    return {"accuracy": correct / total, "per_class_acc": per_class_acc}


def predict_image(model: nn.Module, image_path: str, device: str,
                  image_size: int = 224) -> tuple:
    """
    对单张图片进行推理, 返回 (品种名, 置信度)
    """
    model.eval()
    model.to(device)

    transform = get_transforms(image_size, is_train=False)
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        prob = torch.softmax(output, dim=1)
        conf, pred = prob.max(dim=1)

    breed_names = get_breed_names()
    return breed_names[pred.item()], conf.item()


def load_checkpoint(model: nn.Module, path: str, device: str) -> int:
    """加载 checkpoint, 返回 epoch"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt.get("epoch", 0)


if __name__ == "__main__":
    import sys
    import random
    from models.baseline import build_baseline_model
    from dataset import get_dataloaders, get_breed_names

    if len(sys.argv) < 2:
        print("用法: python inference.py <image_path>")
        print("  或: python inference.py --test <checkpoint_path>")
        print("  或: python inference.py --random [checkpoint_path]")
        sys.exit(1)

    cfg = Config()

    def _build_model_for_ckpt(ckpt_path: str):
        """尝试从 checkpoint 读取模型名，否则默认 resnet18"""
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg_data = ckpt.get("config")
            if cfg_data is not None:
                if isinstance(cfg_data, dict):
                    model_name = cfg_data.get("model_name", "resnet18")
                else:
                    model_name = getattr(cfg_data, "model_name", "resnet18")
                return build_baseline_model(model_name, cfg.num_classes, pretrained=False)
        except Exception:
            pass
        return build_baseline_model("resnet18", cfg.num_classes, pretrained=False)

    if sys.argv[1] == "--test":
        ckpt_path = sys.argv[2]
        model = _build_model_for_ckpt(ckpt_path)
        load_checkpoint(model, ckpt_path, cfg.device)
        model = model.to(cfg.device)
        _, _, test_loader = get_dataloaders(
            data_root=cfg.data_root, batch_size=cfg.batch_size,
            num_workers=cfg.num_workers, image_size=cfg.image_size,
        )
        metrics = evaluate(model, test_loader, cfg.device)
        print(f"测试集 accuracy: {metrics['accuracy']:.4f}")

    elif sys.argv[1] == "--random":
        ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/baseline_resnet18/best.pth"
        model = _build_model_for_ckpt(ckpt_path)
        try:
            load_checkpoint(model, ckpt_path, cfg.device)
            model = model.to(cfg.device)
        except FileNotFoundError:
            print(f"[warn] checkpoint 未找到: {ckpt_path}, 使用随机权重")

        _, _, test_loader = get_dataloaders(
            data_root=cfg.data_root, batch_size=1,
            num_workers=cfg.num_workers, image_size=cfg.image_size,
        )
        test_ds = test_loader.dataset
        idx = random.randint(0, len(test_ds) - 1)
        image, true_label = test_ds[idx]
        breed_names = get_breed_names()

        with torch.no_grad():
            model.eval()
            output = model(image.unsqueeze(0).to(cfg.device))
            prob = torch.softmax(output, dim=1)
            conf, pred = prob.max(dim=1)

        pred_label = pred.item()
        print(f"真实品种: {breed_names[true_label]}")
        print(f"预测品种: {breed_names[pred_label]}  (置信度: {conf.item():.3f})")
        print(f"结果: {'正确' if pred_label == true_label else '错误'}")

        # 生成标注图片
        from PIL import ImageDraw, ImageFont
        from dataset import IMAGENET_MEAN, IMAGENET_STD
        import numpy as np

        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img = image.cpu() * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)

        draw = ImageDraw.Draw(pil_img)
        text_lines = [
            f"True: {breed_names[true_label]}",
            f"Pred: {breed_names[pred_label]} ({conf.item():.2f})",
        ]
        color = (0, 255, 0) if pred_label == true_label else (255, 80, 80)

        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            font = ImageFont.load_default()

        # 右下角标注
        W, H = pil_img.size
        line_h = 22
        margin = 12
        for i, line in enumerate(reversed(text_lines)):
            bbox = draw.textbbox((0, 0), line, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = W - tw - margin
            y = H - (i + 1) * line_h - margin
            draw.text((x, y), line, fill=color, font=font)

        out_path = "prediction_result.png"
        pil_img.save(out_path)
        print(f"图片已保存: {out_path}")

    else:
        ckpt_path = "checkpoints/baseline_resnet18/best.pth"
        model = _build_model_for_ckpt(ckpt_path)
        try:
            load_checkpoint(model, ckpt_path, cfg.device)
            model = model.to(cfg.device)
        except FileNotFoundError:
            print(f"[warn] checkpoint 未找到: {ckpt_path}, 使用随机权重")
        breed, conf = predict_image(model, sys.argv[1], cfg.device, cfg.image_size)
        print(f"预测: {breed}  (置信度: {conf:.3f})")
