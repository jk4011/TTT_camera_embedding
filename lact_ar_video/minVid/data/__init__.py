from minVid.utils.config_utils import get_obj_from_str


def get_data_module(data_config, data_seed=0):
    """Instantiate the data module referenced by config.dataset_train.

    Expected config layout:
        dataset_train:
          target: minVid.data.simple_video_dataset.SimpleVideoDataModule
          params:
            batch_size: 1
            num_workers: 4
            ...

    The resolved class is called as cls(params, data_seed=data_seed) and
    must expose a .train_dataloader() method.
    """
    target = data_config["target"]
    params = data_config.get("params", None) or {}
    cls = get_obj_from_str(target)
    return cls(params, data_seed=data_seed)
