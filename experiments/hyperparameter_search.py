"""
超参数搜索: 固定 epochs=20, 对比不同 lr / weight_decay 组合 (3×3=9组)
"""

import itertools
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from dataset import get_dataloaders
from models.baseline import build_baseline_model
from trainer import Trainer, set_seed, init_logger


def main():
    lr_list = [1e-4]  # 5e-5, 2e-4, 1e-4
    wd_list = [1e-3]  # 1e-3, 1e-4, 5e-4

    results = []

    for lr, wd in itertools.product(lr_list, wd_list):
        lr_head = lr * 10    # 经验法则: head 学习率通常设置为 backbone 的 10 倍
        cfg = Config(
            experiment_name=f"hparam_lr{lr}_wd{wd}",
            model_name="resnet18",
            pretrained=True,
            epochs=20,
            lr=lr,
            lr_head=lr_head,
            weight_decay=wd,
            early_stopping_patience=20,
            logger="swanlab",
        )

        print(f"\n>>> lr={lr}, lr_head={lr_head}, weight_decay={wd}")

        set_seed(cfg.seed)
        logger = init_logger(cfg)

        train_loader, val_loader, _ = get_dataloaders(
            data_root=cfg.data_root, batch_size=cfg.batch_size,
            num_workers=cfg.num_workers, image_size=cfg.image_size,
            val_split=cfg.val_split, seed=cfg.seed,
        )

        model = build_baseline_model(cfg.model_name, cfg.num_classes, cfg.pretrained)
        trainer = Trainer(model, cfg, train_loader, val_loader, logger=logger)
        result = trainer.fit()
        logger.finish()

        print(f"    best_val_acc (20 epoch) = {result['best_val_acc']:.4f}")
        results.append((lr, lr_head, wd, result["best_val_acc"]))

    results.sort(key=lambda x: x[3], reverse=True)

    print(f"\n{'='*60}")
    print("结果汇总 (epochs=20, AdamW + Cosine + label_smoothing=0.1)")
    print(f"{'='*60}")
    print(f"{'lr':<10} {'lr_head':<10} {'weight_decay':<14} {'best_val_acc':<12}")
    print("-" * 46)
    for lr, lr_head, wd, acc in results:
        print(f"{lr:<10.0e} {lr_head:<10.0e} {wd:<14.0e} {acc:.4f}")

    best = results[0]
    print(f"\n最优: lr={best[0]}, lr_head={best[1]}, weight_decay={best[2]}  →  val_acc={best[3]:.4f}")


if __name__ == "__main__":
    main()
