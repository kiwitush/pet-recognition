"""
ViT 训练脚本: 支持 ViT-Tiny/16 和 ViT-Small/16
用法:
  python experiments/train_vit.py model_name=vit_small   # ViT-Small
  python experiments/train_vit.py model_name=vit_tiny    # ViT-Tiny (默认)
  python experiments/train_vit.py model_name=vit_small epochs=80 lr=5e-5 lr_head=5e-4
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from dataset import get_dataloaders
from models.vit import build_vit_model
from trainer import Trainer, set_seed, init_logger
from inference import evaluate


def main():
    cfg = Config(
        experiment_name="vit_tiny",
        model_name="vit_tiny",
        pretrained=True,
        epochs=80,
        lr=5e-5,
        lr_head=5e-4,
        weight_decay=1e-3,
        label_smoothing=0.15,
    )

    # CLI 参数覆盖
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            if hasattr(cfg, k):
                default_type = type(getattr(cfg, k))
                if default_type == bool:
                    v = v.lower() in ("true", "1", "yes")
                else:
                    v = default_type(v)
                setattr(cfg, k, v)
            else:
                print(f"警告: 未知参数 {k}，已跳过")

    # experiment_name 跟随 model_name
    cfg.experiment_name = cfg.model_name

    set_seed(cfg.seed)
    logger = init_logger(cfg)

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=cfg.data_root, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, image_size=cfg.image_size,
        val_split=cfg.val_split, seed=cfg.seed,
    )

    model = build_vit_model(
        model_name=cfg.model_name,
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained,
    )

    trainer = Trainer(model, cfg, train_loader, val_loader, logger=logger)
    result = trainer.fit()
    print(f"ViT best val_acc = {result['best_val_acc']:.4f}")

    # 测试集评估
    best_path = os.path.join(cfg.checkpoint_dir, cfg.experiment_name, "best.pth")
    if os.path.exists(best_path):
        trainer.load_checkpoint(best_path)
        test_metrics = evaluate(trainer.model, test_loader, cfg.device)
        logger.log({"test_acc": test_metrics["accuracy"]})
        print(f"测试集 accuracy: {test_metrics['accuracy']:.4f}")

    logger.finish()


if __name__ == "__main__":
    main()
