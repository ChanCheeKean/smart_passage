import os
import yaml
import argparse
import torch
import super_gradients
from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val)

def main(opt):
    ### load config ###
    config_file = os.path.join(".", "config", opt.cfg)
    with open(config_file, "r") as stream:
        config = yaml.safe_load(stream)

    ### set up experiment name and checkpoint location
    CHECKPOINT_DIR = config['experiment']['checkpoint_dir']
    super_gradients.setup_device(
        device=config['experiment']['device'], 
    )
    
    trainer = Trainer(
        experiment_name=config['experiment']['name'], 
        ckpt_root_dir=CHECKPOINT_DIR,
    )

    ### dataset
    dataset_params = {
        'data_dir': config['dataset']['data_dir'],
        'train_images_dir': config['dataset']['train_images_dir'],
        'train_labels_dir': config['dataset']['train_labels_dir'],
        'val_images_dir': config['dataset']['val_images_dir'],
        'val_labels_dir': config['dataset']['val_labels_dir'],
        'test_images_dir': config['dataset']['test_images_dir'],
        'test_labels_dir': config['dataset']['test_labels_dir'],
        'classes':  config['dataset']['classes'],
    }

    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': config['dataset']['data_dir'],
            'images_dir': config['dataset']['train_images_dir'],
            'labels_dir': config['dataset']['train_labels_dir'],
            'classes': config['dataset']['classes'],
        },
        dataloader_params={
            'batch_size': config['dataset']['batch_size'],
            'num_workers': config['dataset']['num_workers'],
        }
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': config['dataset']['data_dir'],
            'images_dir': config['dataset']['val_images_dir'],
            'labels_dir': config['dataset']['val_labels_dir'],
            'classes': config['dataset']['classes'],
        },
        dataloader_params={
            'batch_size': config['dataset']['batch_size'],
            'num_workers': config['dataset']['num_workers'],
        }
    )

    test_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': config['dataset']['data_dir'],
            'images_dir': config['dataset']['test_images_dir'],
            'labels_dir': config['dataset']['test_labels_dir'],
            'classes': config['dataset']['classes'],
        },
        dataloader_params={
            'batch_size': config['dataset']['batch_size'],
            'num_workers': config['dataset']['num_workers'],
        }
    )

    # plot augmented train dataset
    # train_data.dataset.plot()

    ### model and parameters ###
    model = models.get(
        config['model']['name'],
        num_classes=len(config['dataset']['classes']), 
        pretrained_weights="coco",
    )
    mixed_precision = True if config['experiment']['device'] == 'cuda' else False

    train_params = {
        # ENABLING SILENT MODE
        'silent_mode': config['model']['silent_mode'],
        "average_best_models": config['model']['average_best_models'],
        "warmup_mode": config['model']['warmup_mode'],
        "warmup_initial_lr": config['model']['warmup_initial_lr'],
        "lr_warmup_epochs": config['model']['lr_warmup_epochs'],
        "initial_lr": config['model']['initial_lr'],
        "lr_mode": config['model']['lr_mode'],
        "cosine_final_lr_ratio": config['model']['cosine_final_lr_ratio'],
        "optimizer": config['model']['optimizer'],
        "optimizer_params": config['model']['optimizer_params'],
        "zero_weight_decay_on_bias_and_bn": config['model']['zero_weight_decay_on_bias_and_bn'],
        "ema": config['model']['ema'],
        "ema_params": config['model']['ema_params'],
        "max_epochs": config['model']['max_epochs'],
        "mixed_precision": mixed_precision,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=len(config['dataset']['classes']),
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=len(config['dataset']['classes']),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.5
                )
            )
        ],
        "metric_to_watch": config['model']['metric_to_watch'],
    }

    ### training ###
    trainer.train(
        model=model, 
        training_params=train_params, 
        train_loader=train_data, 
        valid_loader=val_data
    )

    ### evaluation ###
    best_model = models.get(
        config['model']['name'],
        num_classes=len(config['dataset']['classes']), 
        checkpoint_path=f"{CHECKPOINT_DIR}/{config['experiment']['name']}/ckpt_best.pth"
    )

    trainer.test(
        model=best_model,
        test_loader=test_data,
        test_metrics_list=DetectionMetrics_050(
            score_thres=0.1, 
            top_k_predictions=300, 
            num_cls=len(config['dataset']['classes']),
            normalize_targets=True, 
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7)                                                                                            
        )
    )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='default.yaml', help='model.yaml path')
    return parser.parse_known_args()[0] if known else parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

# python train.py --cfg default.yaml