import torch
import torch.nn as nn
from mmengine import Config
from mmengine.runner import Runner
from mmpretrain.apis import init_model
from dynaconf import Dynaconf
from cvsdk.mm.pretrain.config import TrainingConfig
from collections import OrderedDict
from rich.pretty import pprint
from structlog import get_logger

logger = get_logger()


class MMPretrainModels:
    """MMPretrain models class for image classification and feature extraction.
    """
    models = {
        "resnet": ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        "vit": ["vit-small-p16", "vit-base-p16", "vit-large-p16"],
        "swin": ["swin-tiny", "swin-small", "swin-base", "swin-large"],
        "efficientnet": ["efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3"],
    }

    @staticmethod
    def get_available_models() -> dict[str, list[str]]:
        """Get available models.

        Returns:
            dict[str, list[str]]: Dictionary of available models.
        """
        return MMPretrainModels.models
    
    @staticmethod
    def get_config(config_file: str, envvar_prefix: str = "", load_from: str | None = None) -> Config:
        """Load a configuration.

        Args:
            config_file (str): YAML training configuration file
            envvar_prefix (str, optional): Prefix for environment variables. Defaults to "".
            load_from (str | None, optional): Path to checkpoint file with pretrained weights. Defaults to None.

        Returns:
            Config: MMPretrain configuration
        """
        settings = Dynaconf(
            envvar_prefix=envvar_prefix,
            settings_files=[config_file],
            lowercase_envvars=True,
        )
        
        config_data = {k.lower(): v for k, v in settings.items()}
        config = TrainingConfig(**config_data)

        DATASET_DIR = config.dataset_dir
        DATASET_CLASSES = config.dataset_classes

        MODEL_TYPE = config.model_type
        MODEL_NAME = config.model_name
        BATCH_SIZE = config.batch_size
        NUM_CLASSES = len(config.dataset_classes)
        EPOCHS = config.epochs
        WORK_DIR = config.work_dir

        ANN_TRAIN = config.annotations_train
        ANN_VAL = config.annotations_val
        ANN_TEST = config.annotations_test

        OPTIMIZER = config.optimizer

        cfg = Config.fromfile(f"{config.config_path}/{config.model_type}/{config.model_name}.py")
        logger.info("LOAD FROM", load_from=load_from)
        cfg.load_from = load_from

        if OPTIMIZER == "sgd":
            cfg.optim_wrapper.optimizer = {
                'type': 'SGD',
                'lr': config.lr,
                'momentum': config.momentum,
                'weight_decay': config.weight_decay,
            }
        elif OPTIMIZER == "adamw":
            cfg.optim_wrapper.optimizer = {
                'type': 'AdamW',
                'lr': config.lr,
                'weight_decay': config.weight_decay,
            }

        # Configure data pipeline for training
        train_pipeline = [
            dict(type='LoadImageFromFile'),
        ]

        train_pipeline += config.augmentations

        train_pipeline += [
            dict(type='PackInputs')
        ]

        # Configure datasets
        cfg.train_dataloader.dataset.data_root = DATASET_DIR
        cfg.train_dataloader.dataset.ann_file = ANN_TRAIN
        cfg.train_dataloader.dataset.data_prefix = config.train_dir
        cfg.train_dataloader.dataset.pipeline = train_pipeline
        
        cfg.val_dataloader.dataset.data_root = DATASET_DIR
        cfg.val_dataloader.dataset.data_prefix = config.val_dir
        cfg.val_dataloader.dataset.ann_file = ANN_VAL
        
        cfg.test_dataloader.dataset.data_root = DATASET_DIR
        cfg.test_dataloader.dataset.data_prefix = config.test_dir
        cfg.test_dataloader.dataset.ann_file = ANN_TEST
        
        cfg.train_cfg.max_epochs = EPOCHS
        cfg.default_hooks.logger.interval = 10

        # Update model head for number of classes
        if hasattr(cfg.model, 'head'):
            cfg.model.head.num_classes = NUM_CLASSES

        cfg.train_dataloader.batch_size = BATCH_SIZE
        cfg.val_dataloader.batch_size = BATCH_SIZE
        cfg.test_dataloader.batch_size = BATCH_SIZE
        cfg.work_dir = WORK_DIR
        cfg.resume = load_from is not None
        return cfg

    @staticmethod
    def train(config_file: str, load_from: str | None = None):
        """Train a model.

        Args:
            config_file (str): Path to configuration file
            load_from (str | None, optional): Path to checkpoint file. Defaults to None.
        """
        cfg = MMPretrainModels.get_config(config_file=config_file, load_from=load_from)
        runner = Runner.from_cfg(cfg)
        runner.train()

    @staticmethod
    def extract_backbone(config_file: str, output_file: str, load_from: str | None = None):
        """Extract and save the backbone from a trained model.

        Args:
            config_file (str): Path to configuration file
            output_file (str): Path to save the backbone weights
            load_from (str | None, optional): Path to checkpoint file. Defaults to None.
        """
        config: Config = Config.fromfile(config_file)
        model: nn.Module = init_model(config, load_from, device="cpu").eval()
        torch.save(model.backbone.state_dict(), output_file)
        logger.info("Backbone extracted", output_file=output_file)
