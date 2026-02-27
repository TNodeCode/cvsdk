import torch
import torch.nn as nn
from mmengine import Config
from mmengine.runner import Runner
from mmdet.apis import init_detector
from dynaconf import Dynaconf
from cvsdk.mm.det.config import TrainingConfig
from collections import OrderedDict
from rich.pretty import pprint
from structlog import get_logger
from cvsdk.mm.det.backbones import *
from cvsdk.mm.det.necks import *
from cvsdk.mm.det.vitdet.vitdet import *

logger = get_logger()


class MMDetModels:
  """MMDetection models class.
  """
  models = {
    "faster_rcnn": ["faster-rcnn_r50_fpn_1x_coco", "faster-rcnn_r101_fpn_1x_coco", "faster-rcnn_x101-32x4d_fpn_1x_coco", "faster-rcnn_x101-64x4d_fpn_1x_coco"],
    "cascade_rcnn": ["cascade-rcnn_r50_fpn_1x_coco", "cascade-rcnn_r101_fpn_1x_coco", "cascade-rcnn_x101-32x4d_fpn_1x_coco", "cascade-rcnn_x101-64x4d_fpn_1x_coco"],
    "deformable_detr": ["deformable-detr_r50_16xb2-50e_coco", "deformable-detr-refine_r50_16xb2-50e_coco", "deformable-detr-refine-twostage_r50_16xb2-50e_coco"],
    "yolox": ["yolox_nano_8xb8-300e_coco", "yolox_tiny_8xb8-300e_coco", "yolox_s_8xb8-300e_coco", "yolox_m_8xb8-300e_coco", "yolox_l_8xb8-300e_coco", "yolox_x_8xb8-300e_coco"],
  }

  @staticmethod
  def get_available_models() -> dict[str, list[str]]:
    """Get available models.

    Returns:
        dict[str, list[str]]: Dictionary of available models.
    """
    return MMDetModels.models
  
  @staticmethod
  def get_config(config_file: str, envvar_prefix: str = "", load_from: str | None = None) -> Config:
    """Load a configuration.

    Args:
        config_file (str): YAML training configuration file
        envvar_prefix (str, optional): Prefix for environment variables. Defaults to "".
        load_from (str | None, optional): Path to checkpoint file with pretrained weights. Defaults to None.

    Returns:
        Config: MMDetection configuration
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

    ANN_TRAIN=config.annotations_train
    ANN_VAL=config.annotations_val
    ANN_TEST=config.annotations_test

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
    # see: https://mmdetection.readthedocs.io/en/v2.19.1/tutorials/data_pipeline.html
    train_pipeline = [
      dict(type='LoadImageFromFile', backend_args=None),
      dict(type='LoadAnnotations', with_bbox=True),
    ]

    train_pipeline += config.augmentations

    train_pipeline += [
      dict(type='PackDetInputs')
    ]

    if MODEL_TYPE == "yolox":
      # YoloX uses MultiImageMixDataset, has to be configured differently
      cfg.train_dataloader.dataset.dataset.data_root=DATASET_DIR
      cfg.train_dataloader.dataset.dataset.ann_file=f"{ANN_TRAIN}"
      cfg.train_dataloader.dataset.dataset.data_prefix.img=f"{config.train_dir}/"
      cfg.train_dataloader.dataset.dataset.update({'metainfo': {'classes': DATASET_CLASSES}})
      cfg.train_dataloader.dataset.dataset.pipeline = train_pipeline
    else:
      cfg.train_dataloader.dataset.data_root=DATASET_DIR
      cfg.train_dataloader.dataset.ann_file=f"{ANN_TRAIN}"
      cfg.train_dataloader.dataset.data_prefix.img=f"{config.train_dir}/"
      cfg.train_dataloader.dataset.update({'metainfo': {'classes': DATASET_CLASSES}})
      cfg.train_dataloader.dataset.pipeline = train_pipeline
    cfg.val_dataloader.dataset.data_root=DATASET_DIR
    cfg.val_dataloader.dataset.data_prefix.img=f"{config.val_dir}/"
    cfg.val_dataloader.dataset.ann_file=f"{ANN_VAL}"
    cfg.val_evaluator.ann_file=f"{DATASET_DIR}{ANN_VAL}"
    cfg.val_dataloader.dataset.update({'metainfo': {'classes': DATASET_CLASSES}})
    cfg.test_dataloader.dataset.data_root=DATASET_DIR
    cfg.test_dataloader.dataset.data_prefix.img=f"{config.test_dir}/"
    cfg.test_dataloader.dataset.ann_file=f"{ANN_TEST}"
    cfg.test_evaluator.ann_file=f"{DATASET_DIR}{ANN_TEST}"
    cfg.train_cfg.max_epochs=EPOCHS
    cfg.default_hooks.logger.interval=10

    if cfg.model.backbone and cfg.model.backbone.type == "ResNet" and config.backbone.type == "ResNet":
      cfg.model.backbone.init_cfg.checkpoint = config.backbone.checkpoint
      cfg.model.backbone.frozen_stages = config.backbone.frozen_stages
      cfg.model.backbone.depth = config.backbone.depth
      cfg.model.backbone.out_indices = config.backbone.out_indices
      cfg.model.neck.in_channels = config.backbone.out_channels
    elif cfg.model.backbone and cfg.model.backbone.type == "mmpretrain.EfficientNetV2":
      cfg.model.backbone.init_cfg.checkpoint = config.backbone.checkpoint
      cfg.model.backbone.arch = config.backbone.arch
      cfg.model.backbone.out_indices = config.backbone.out_indices
      cfg.model.neck.in_channels = config.backbone.out_channels
    elif cfg.model.backbone and cfg.model.backbone.type == "SwinTransformer":
      cfg.model.backbone.init_cfg.checkpoint = config.backbone.checkpoint
      cfg.model.backbone.pretrain_img_size = config.backbone.pretrain_img_size
      cfg.model.backbone.attn_drop_rate = config.backbone.attn_drop_rate
      cfg.model.backbone.embed_dims = config.backbone.embed_dims
      cfg.model.backbone.drop_path_rate = config.backbone.drop_path_rate
      cfg.model.backbone.drop_rate = config.backbone.drop_rate
      cfg.model.backbone.num_heads = config.backbone.num_heads
      cfg.model.backbone.out_indices = config.backbone.out_indices

    if MODEL_TYPE in ["faster_rcnn", "cascade_rcnn"]:
      if type(cfg.model.roi_head) is list:
        for bbox_head in cfg.model.roi_head:
          bbox_head.num_classes=NUM_CLASSES
      else:
        cfg.model.roi_head.bbox_head.num_classes=NUM_CLASSES
    if MODEL_TYPE in ["deformable_detr", "dino"]:
      cfg.model.bbox_head.num_classes=NUM_CLASSES
    elif MODEL_TYPE in ["codino"]:
      ffn_cfgs=dict(
        type='FFN',
        embed_dims=config.detr_encoder.embed_dims,
        feedforward_channels=1024,
        num_fcs=2,
        ffn_drop=0.,
        act_cfg=dict(type='ReLU', inplace=True),
      )
      if cfg.model.backbone.type == "ViT":
        cfg.model.backbone.finetuning = config.backbone.finetuning
        cfg.model.backbone.depth = config.backbone.depth
        cfg.model.backbone.embed_dim = config.backbone.embed_dims
        cfg.model.backbone.num_heads = config.backbone.num_heads
        cfg.model.backbone.patch_size = config.backbone.patch_size
        cfg.model.backbone.window_size = config.backbone.window_size
        cfg.model.neck.backbone_channel = config.backbone.embed_dims
        cfg.model.neck.in_channels = [config.backbone.embed_dims // 4, config.backbone.embed_dims // 2, config.backbone.embed_dims, config.backbone.embed_dims]
        cfg.model.neck.out_channels = config.backbone.neck_out_channels
      cfg.model.neck.out_channels=config.detr_encoder.embed_dims
      cfg.model.bbox_head[0].num_classes=NUM_CLASSES
      cfg.model.bbox_head[0].in_channels=config.detr_encoder.embed_dims
      cfg.model.bbox_head[0].feat_channels=config.detr_encoder.embed_dims
      cfg.model.neck.out_channels=config.detr_encoder.embed_dims
      cfg.model.query_head.num_classes=NUM_CLASSES
      cfg.model.roi_head[0].bbox_head.num_classes=NUM_CLASSES
      cfg.model.roi_head[0].bbox_head.in_channels=config.detr_encoder.embed_dims
      cfg.model.roi_head[0].bbox_roi_extractor.out_channels=config.detr_encoder.embed_dims
      cfg.model.rpn_head.feat_channels=config.detr_encoder.embed_dims
      cfg.model.rpn_head.in_channels=config.detr_encoder.embed_dims
      cfg.model.query_head.positional_encoding.num_feats=config.detr_encoder.embed_dims // 2
      cfg.model.query_head.transformer.encoder.num_layers=config.detr_encoder.num_layers
      cfg.model.query_head.transformer.encoder.transformerlayers.attn_cfgs.embed_dims=config.detr_encoder.embed_dims
      cfg.model.query_head.transformer.encoder.transformerlayers.attn_cfgs.dropout=config.detr_encoder.attn_dropout
      cfg.model.query_head.transformer.encoder.transformerlayers.ffn_cfgs=ffn_cfgs
      cfg.model.query_head.transformer.encoder.transformerlayers.ffn_dropout=config.detr_encoder.ffn_dropout
      cfg.model.query_head.transformer.decoder.num_layers=config.detr_decoder.num_layers
      cfg.model.query_head.transformer.decoder.transformerlayers.ffn_cfgs=ffn_cfgs
      cfg.model.query_head.transformer.decoder.transformerlayers.attn_cfgs[0].embed_dims=config.detr_decoder.embed_dims
      cfg.model.query_head.transformer.decoder.transformerlayers.attn_cfgs[0].dropout=config.detr_decoder.attn_dropout
      cfg.model.query_head.transformer.decoder.transformerlayers.attn_cfgs[1].embed_dims=config.detr_decoder.embed_dims
      cfg.model.query_head.transformer.decoder.transformerlayers.attn_cfgs[1].dropout=config.detr_decoder.attn_dropout
      cfg.model.query_head.transformer.decoder.transformerlayers.ffn_dropout=config.detr_decoder.ffn_dropout
    elif MODEL_TYPE == "yolox":
      cfg.model.bbox_head.num_classes=NUM_CLASSES

    cfg.train_dataloader.batch_size=BATCH_SIZE
    cfg.val_dataloader.batch_size=BATCH_SIZE
    cfg.test_dataloader.batch_size=BATCH_SIZE
    cfg.work_dir=WORK_DIR
    cfg.resume = True if load_from is None else False
    return cfg

  @staticmethod
  def train(config_file: str, load_from: str | None = None):
    cfg = MMDetModels.get_config(config_file=config_file, load_from=load_from)
    runner = Runner.from_cfg(cfg)
    runner.train()

  @staticmethod
  def extract_backbone(config_file: str, output_file: str, load_from: str | None = None):
    config: Config = Config.fromfile(config_file)
    model: nn.Module = init_detector(config, load_from, device="cpu").eval()
    torch.save(model.backbone.state_dict(), output_file)
  
  @staticmethod
  def copy_backbone(
    source_config_file: str,
    target_config_file: str,
    output_file: str,
    load_source_from: str | None = None
  ):
    _state = torch.load(load_source_from)
    pprint(_state.keys())
    config_source: Config = Config.fromfile(source_config_file)
    config_target: Config = Config.fromfile(target_config_file)
    logger.info("Loading source model", config_file=source_config_file, source=load_source_from)
    model_source: nn.Module = init_detector(config_source, load_source_from, device="cpu").eval()
    logger.info("Loading target model", config_file=target_config_file, load_from=None)
    model_target: nn.Module = init_detector(config_target, None, device="cpu").eval()
    source_layers = set(model_source.backbone.state_dict().keys())
    target_layers = set(model_target.backbone.state_dict().keys())
    if not source_layers - target_layers and not target_layers - source_layers:
      logger.info("Source and target backbone are identical")
    if source_layers - target_layers:
      logger.warning("Source layer contains layers that do not exist in the target layer")
      pprint(source_layers - target_layers)
    if target_layers - source_layers:
      logger.warning("Target layer contains layers that do not exist in the source layer")
      pprint(target_layers - source_layers)
    source_backbone_state_dict: OrderedDict = model_source.backbone.state_dict()
    model_target.backbone.load_state_dict(source_backbone_state_dict)
    logger.info("Storing target model with copied backbone weights from source model", output_file=output_file)
    torch.save(OrderedDict({'state_dict': model_target.state_dict()}), output_file)
