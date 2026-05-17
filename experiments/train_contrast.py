"""
预训练消融实验: 随机初始化 vs ImageNet 预训练
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from dataset import get_dataloaders, NUM_CLASSES
from models.baseline import build_baseline_model
from trainer import Trainer, set_seed, init_logger
from inference import evaluate


def run_experiment(cfg: Config, label: str):
    print(f"\n{'='*50}")
    print(f"对比实验: {label}")
    print(f"{'='*50}")

    set_seed(cfg.seed)
    logger = init_logger(cfg)

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=cfg.data_root, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, image_size=cfg.image_size,
        val_split=cfg.val_split, seed=cfg.seed,
    )

    model = build_baseline_model(cfg.model_name, NUM_CLASSES, pretrained=cfg.pretrained)
    trainer = Trainer(model, cfg, train_loader, val_loader, logger=logger)
    result = trainer.fit()

    # 测试集评估
    best_path = os.path.join(cfg.checkpoint_dir, cfg.experiment_name, "best.pth")
    if os.path.exists(best_path):
        trainer.load_checkpoint(best_path)
        test_metrics = evaluate(trainer.model, test_loader, cfg.device)
        result["test_acc"] = test_metrics["accuracy"]
        logger.log({"test_acc": test_metrics["accuracy"]})
        print(f"[{label}] 最佳 val_acc = {result['best_val_acc']:.4f}  |  测试集 acc = {result['test_acc']:.4f}")
    else:
        print(f"[{label}] 最佳 val_acc = {result['best_val_acc']:.4f}")

    logger.finish()
    return result, trainer


def main():
    # 随机初始化
    cfg_scratch = Config(
        experiment_name="contrast_scratch",
        pretrained=False,
        epochs=80,
        lr=1e-4,
        lr_head=1e-4,
    )
    result_scratch, _ = run_experiment(cfg_scratch, "随机初始化")

if __name__ == "__main__":
    main()
