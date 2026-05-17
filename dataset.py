"""
Oxford-IIIT Pet Dataset 加载与预处理.

37 类宠物品种, 官方划分 trainval/test. 这里将 trainval 进一步切分为 train/val.
"""

from typing import Tuple, Optional, List, Callable

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import InterpolationMode


NUM_CLASSES = 37
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class PetDataset(Dataset):
    """对数据集子集独立应用 transform, 解决 random_split 后 Subset transform 不一致的问题."""

    def __init__(self, base_dataset: OxfordIIITPet, indices: List[int],
                 transform: Optional[Callable] = None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.base_dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_transforms(image_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """返回数据增强流水线. is_train=True 含增强, 否则仅 Resize + CenterCrop + Normalize."""
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.10), value='random'),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 256 / 224), interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_dataloaders(
    data_root: str = "./data",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """返回 (train_loader, val_loader, test_loader)."""
    # 加载原始 trainval (不含 transform, 后续由 wrapper 按需应用)
    base_trainval = OxfordIIITPet(
        root=data_root,
        split="trainval",
        target_types="category",
        transform=None,
        download=True,
    )

    # 训练/验证集划分
    generator = torch.Generator().manual_seed(seed)
    val_size = int(len(base_trainval) * val_split)
    train_size = len(base_trainval) - val_size
    train_subset, val_subset = random_split(
        base_trainval, [train_size, val_size], generator=generator
    )

    train_ds = PetDataset(base_trainval, train_subset.indices,
                          transform=get_transforms(image_size, is_train=True))
    val_ds   = PetDataset(base_trainval, val_subset.indices,
                          transform=get_transforms(image_size, is_train=False))

    # 测试集 (官方 test 划分)
    test_ds = OxfordIIITPet(
        root=data_root,
        split="test",
        target_types="category",
        transform=get_transforms(image_size, is_train=False),
        download=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_breed_names() -> List[str]:
    """返回 37 个宠物品种名称 (按标签索引顺序)"""
    return [
        "Abyssinian", "American Bulldog", "American Pit Bull Terrier",
        "Basset Hound", "Beagle", "Bengal", "Birman", "Bombay",
        "Boxer", "British Shorthair", "Chihuahua", "Egyptian Mau",
        "English Cocker Spaniel", "English Setter", "German Shorthaired",
        "Great Pyrenees", "Havanese", "Japanese Chin", "Keeshond",
        "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland",
        "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue",
        "Saint Bernard", "Samoyed", "Scottish Terrier", "Shiba Inu",
        "Siamese", "Sphynx", "Staffordshire Bull Terrier", "Wheaten Terrier",
        "Yorkshire Terrier",
    ]


# 快速自检
if __name__ == "__main__":
    print("正在加载 Oxford-IIIT Pet 数据集... (首次运行自动下载)")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root="./data", batch_size=32, num_workers=0, val_split=0.2, seed=42
    )
    print(f"训练集批次数: {len(train_loader)}  |  总样本: {len(train_loader.dataset)}")
    print(f"验证集批次数: {len(val_loader)}    |  总样本: {len(val_loader.dataset)}")
    print(f"测试集批次数: {len(test_loader)}  |  总样本: {len(test_loader.dataset)}")

    images, labels = next(iter(train_loader))
    print(f"一个 batch 的图像 shape:  {images.shape}")
    print(f"一个 batch 的标签: min={labels.min().item()}, max={labels.max().item()}")
    print(f"归一化统计 — mean: {images.mean():.3f}  std: {images.std():.3f}")
    print(f"品种名称数量: {len(get_breed_names())}")

    # 验证整个训练集覆盖全部 37 类
    all_labels = torch.cat([y for _, y in train_loader])
    print(f"训练集覆盖类别数: {all_labels.unique().numel()} / 37   (应为 37)")
    print(f"训练集标签范围: {all_labels.min().item()} – {all_labels.max().item()}   (应为 0–36)")
