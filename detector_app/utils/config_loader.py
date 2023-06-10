import os
import yaml

# loading configuration from config yaml
# config_file = os.path.join("config", "default.yml")
config_file = os.path.join("config", "yolo_retrain.yml")

with open(config_file, "r") as stream:
    config = yaml.safe_load(stream)