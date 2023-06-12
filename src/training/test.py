import os
import yaml
import argparse
import cv2
import numpy as np
import torch
from super_gradients.training import models

def main(opt):
    ### load config ###
    config_file = os.path.join(".", "config", opt.cfg)
    with open(config_file, "r") as stream:
        config = yaml.safe_load(stream)
    CHECKPOINT_DIR = config['experiment']['checkpoint_dir']

    best_model = models.get(
            config['model']['name'],
            num_classes=len(config['dataset']['classes']), 
            checkpoint_path=f"{CHECKPOINT_DIR}/{config['experiment']['name']}/ckpt_best.pth"
        )
    
    # test
    image = cv2.imread('/content/smart_passage/src/spl_data/processed/test/images/0473_p_1_jpg.rf.886b6d12402598dbd0ad68cf8b80f1f8.jpg') 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = list(best_model.predict(image, conf=0.5))[0]
    results = np.concatenate((
                    result.prediction.bboxes_xyxy, 
                    result.prediction.confidence.reshape(-1, 1), 
                    result.prediction.labels.reshape(-1, 1)), axis=1)
    results = torch.Tensor(results)
    print(results.size())

    # unit test
    img_url = 'https://raw.githubusercontent.com/kalyco/yolo_detector/master/images/baggage_claim.jpg'
    best_model.predict(img_url).show()

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='default.yaml', help='model.yaml path')
    return parser.parse_known_args()[0] if known else parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)