"""
进阶模型训练: ResNet + 注意力机制 (SE / CBAM)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import attention_config
from dataset import get_dataloaders
from models.attention import build_advanced_model
from trainer import Trainer, set_seed, init_logger
from inference import evaluate


def main():
    cfg = attention_config()

    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=")
            if hasattr(cfg, k):
                orig_type = type(getattr(cfg, k))
                if orig_type is bool:
                    setattr(cfg, k, v.lower() in ("true", "1", "yes"))
                else:
                    setattr(cfg, k, orig_type(v))

    # 确保 experiment_name 与 attention_type 一致
    cfg.experiment_name = f"attention_resnet18_{cfg.attention_type}"

    set_seed(cfg.seed)
    logger = init_logger(cfg)

    print(f"设备: {cfg.device}")
    print(f"注意力模块: {cfg.attention_type}")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=cfg.data_root, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, image_size=cfg.image_size,
        val_split=cfg.val_split, seed=cfg.seed,
    )

    model = build_advanced_model(
        cfg.model_name, cfg.num_classes,
        pretrained=cfg.pretrained, attention_type=cfg.attention_type,
    )
    trainer = Trainer(model, cfg, train_loader, val_loader, logger=logger)
    result = trainer.fit()

    # 测试集评估
    best_path = os.path.join(cfg.checkpoint_dir, cfg.experiment_name, "best.pth")
    if os.path.exists(best_path):
        trainer.load_checkpoint(best_path)
        test_metrics = evaluate(trainer.model, test_loader, cfg.device)
        result["test_acc"] = test_metrics["accuracy"]

    print(f"进阶模型 best val_acc = {result['best_val_acc']:.4f}")
    if result.get("test_acc"):
        print(f"测试集 accuracy: {result['test_acc']:.4f}")

    if logger is not None:
        logger.finish()


if __name__ == "__main__":
    main()
