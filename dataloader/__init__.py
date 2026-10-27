"""Dataset-specific data loader implementations."""

from importlib import import_module


def generate_dataset_loader(cfg):
    """Build train and validation loaders for the configured dataset."""
    dataset_name = cfg["dataset"]
    supported = {"GenVideo", "AIGVDBench"}
    if dataset_name not in supported:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}; choose one of {sorted(supported)}"
        )

    dataset_module = import_module(f".{dataset_name}", package=__name__)
    return dataset_module.generate_dataset_loader(cfg)
