from .split import Split
from .transform import TransformBuilder
from .dataset import MammoBenchDataset
from torch.utils.data import DataLoader
import torch


def builder_dataloader(
        split : Split,
        train_transform_builder : TransformBuilder,
        eval_transform_builder : TransformBuilder,
        batch_size: int = 16,
        num_workers: int = 1
)-> dict[str, DataLoader[tuple[torch.Tensor,int]]]:


    train_transform = train_transform_builder.build()
    eval_transform = eval_transform_builder.build()

    train_ds = MammoBenchDataset(df= split.train_df(), transform=train_transform)
    val_ds = MammoBenchDataset(df= split.val_df(), transform=eval_transform)
    test_ds = MammoBenchDataset(df= split.test_df(), transform=eval_transform)

    return {

        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    }