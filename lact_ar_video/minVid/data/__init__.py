import inspect

from minVid.utils.config_utils import get_obj_from_str


def get_data_module(data_config, data_seed=0, consumed_batches=0):
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

    consumed_batches > 0 requests a deterministic, stream-aligned resume:
    the module should reproduce the exact index stream a fresh loader with
    the same data_seed would emit, minus the first consumed_batches batches.
    Only passed through when the module supports it (currently
    MultiCamPairDataModule); otherwise a warning is printed and the module
    is built without fast-forward.
    """
    target = data_config["target"]
    params = data_config.get("params", None) or {}
    cls = get_obj_from_str(target)
    if consumed_batches:
        if "consumed_batches" in inspect.signature(cls.__init__).parameters:
            return cls(params, data_seed=data_seed,
                       consumed_batches=consumed_batches)
        print(f"WARNING: {target} does not support consumed_batches; "
              f"resume will NOT be stream-aligned to the fresh run.")
    return cls(params, data_seed=data_seed)
