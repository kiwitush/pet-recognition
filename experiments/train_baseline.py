"""
Baseline 模型训练: ImageNet 预训练 ResNet-18 微调
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import baseline_config
from dataset import get_dataloaders
from models.baseline import build_baseline_model
from trainer import Trainer, set_seed, init_logger
from inference import evaluate


def main():
    cfg = baseline_config()

    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=")
            if hasattr(cfg, k):
                orig_type = type(getattr(cfg, k))
                if orig_type is bool:
                    setattr(cfg, k, v.lower() in ("true", "1", "yes"))
                else:
                    setattr(cfg, k, orig_type(v))

    set_seed(cfg.seed)
    logger = init_logger(cfg)

    print(f"设备: {cfg.device}")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=cfg.data_root, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, image_size=cfg.image_size,
        val_split=cfg.val_split, seed=cfg.seed,
    )

    model = build_baseline_model(cfg.model_name, cfg.num_classes, cfg.pretrained)
    trainer = Trainer(model, cfg, train_loader, val_loader, logger=logger)
    trainer.fit()

    # 加载最佳模型并在测试集上评估
    best_path = os.path.join(cfg.checkpoint_dir, cfg.experiment_name, "best.pth")
    if os.path.exists(best_path):
        trainer.load_checkpoint(best_path)
        test_metrics = evaluate(trainer.model, test_loader, cfg.device)
        print(f"测试集 accuracy: {test_metrics['accuracy']:.4f}")
        if logger is not None:
            logger.log({"test_acc": test_metrics["accuracy"]})

    if logger is not None:
        logger.finish()


if __name__ == "__main__":
    main()
