import torch
import torch.nn as nn
from mmengine import Config
from mmengine.runner import Runner
from mmseg.apis import init_model
from dynaconf import Dynaconf
from cvsdk.mmsegmentation.config import TrainingConfig
from collections import OrderedDict
from rich.pretty import pprint
from structlog import get_logger

logger = get_logger()


class MMSegModels:
  """MMSegmentation models class.
  """
  models = {
    "fcn": ["fcn_r50-d8_512x512_20k_voc12aug", "fcn_r101-d8_512x512_20k_voc12aug"],
    "pspnet": ["pspnet_r50-d8_512x512_20k_voc12aug", "pspnet_r101-d8_512x512_20k_voc12aug"],
    "deeplabv3": ["deeplabv3_r50-d8_512x512_20k_voc12aug", "deeplabv3_r101-d8_512x512_20k_voc12aug"],
    "deeplabv3plus": ["deeplabv3plus_r50-d8_512x512_20k_voc12aug", "deeplabv3plus_r101-d8_512x512_20k_voc12aug"],
  }

  @staticmethod
  def get_available_models() -> dict[str, list[str]]:
    """Get available models.

    Returns:
        dict[str, list[str]]: Dictionary of available models.
    """
    return MMSegModels.models
  
  @staticmethod
  def get_config(config_file: str, envvar_prefix: str = "", load_from: str | None = None) -> Config:
    """Load a configuration.

    Args:
        config_file (str): YAML training configuration file
        envvar_prefix (str, optional): Prefix for environment variables. Defaults to "".
        load_from (str | None, optional): Path to checkpoint file with pretrained weights. Defaults to None.

    Returns:
        Config: MMSegmentation configuration
    """
    settings = Dynaconf(
        envvar_prefix=envvar_prefix,
        settings_files=[config_file],
        lowercase_envvars=True,
    )
    
    config_data = {k.lower(): v for k, v in settings.items()}
    config = TrainingConfig(**config_data)

    DATASET_DIR=config.dataset_dir
    DATASET_CLASSES=config.dataset_classes

    MODEL_TYPE=config.model_type
    MODEL_NAME=config.model_name
    BATCH_SIZE=config.batch_size
    NUM_CLASSES=len(config.dataset_classes)
    EPOCHS=config.epochs
    WORK_DIR=config.work_dir

    OPTIMIZER=config.optimizer

    cfg = Config.fromfile(f"{config.config_path}/{config.model_type}/{config.model_name}.py")
    print("LOAD FROM", load_from)
    cfg.load_from = load_from

    if OPTIMIZER=="sgd":
      cfg.optim_wrapper.optimizer = {
        'type': 'SGD',
        'lr': config.lr,
        'momentum': config.momentum,
        'weight_decay': config.weight_decay,
      }
    elif OPTIMIZER=="adamw":
      cfg.optim_wrapper.optimizer = {
        'type': 'AdamW',
        'lr': config.lr,
        'weight_decay': config.weight_decay,
      }

    # Here we can define image augmentations used for training.
    train_pipeline = [
      dict(type='LoadImageFromFile'),
      dict(type='LoadAnnotations'),
    ]

    train_pipeline += config.augmentations

    train_pipeline += [
      dict(type='PackSegInputs')
    ]

    cfg.train_dataloader.dataset.data_root=DATASET_DIR
    cfg.train_dataloader.dataset.data_prefix.img_path=f"{config.train_dir}/"
    cfg.train_dataloader.dataset.data_prefix.seg_map_path=f"{config.train_seg_dir}/"
    cfg.train_dataloader.dataset.pipeline = train_pipeline
    
    cfg.val_dataloader.dataset.data_root=DATASET_DIR
    cfg.val_dataloader.dataset.data_prefix.img_path=f"{config.val_dir}/"
    cfg.val_dataloader.dataset.data_prefix.seg_map_path=f"{config.val_seg_dir}/"
    
    cfg.test_dataloader.dataset.data_root=DATASET_DIR
    cfg.test_dataloader.dataset.data_prefix.img_path=f"{config.test_dir}/"
    cfg.test_dataloader.dataset.data_prefix.seg_map_path=f"{config.test_seg_dir}/"
    
    cfg.train_cfg.max_iters=EPOCHS
    cfg.default_hooks.logger.interval=10

    if cfg.model.decode_head:
      cfg.model.decode_head.num_classes=NUM_CLASSES
    
    if hasattr(cfg.model, 'auxiliary_head') and cfg.model.auxiliary_head:
      cfg.model.auxiliary_head.num_classes=NUM_CLASSES

    cfg.train_dataloader.batch_size=BATCH_SIZE
    cfg.val_dataloader.batch_size=BATCH_SIZE
    cfg.test_dataloader.batch_size=BATCH_SIZE
    cfg.work_dir=WORK_DIR
    cfg.resume = True if load_from is None else False
    return cfg

  @staticmethod
  def train(config_file: str, load_from: str | None = None):
    cfg = MMSegModels.get_config(config_file=config_file, load_from=load_from)
    runner = Runner.from_cfg(cfg)
    runner.train()

  @staticmethod
  def extract_backbone(config_file: str, output_file: str, load_from: str | None = None):
    config: Config = Config.fromfile(config_file)
    model: nn.Module = init_model(config, load_from, device="cpu").eval()
    torch.save(model.backbone.state_dict(), output_file)
