# Smart Passage Logic

This repo include the end to end proess from curating data to model deployment.
First, create a new environment with the packages

```
cd /smart_passage/src
pip install -r requirements.txt
```
## 1. Data Pulling and Updating

It's vital to note that no data should be allowed in code repository.
This step involves pushing the data source curated to Server Storage or Minio. 
```
cd smart_passage/src/pre-processing
python publish.py --origin {folder_to_be_uploaded} --bucket {bucket_name} --prefix {server_dir_name}
# python publish.py --origin "../data/landing" --bucket "smart-passage-logic" --prefix "/data/landing"
```

To pull the data to local from the server
```
cd smart_passage/src/pre-processing
python download.py --origin {local_destination} --bucket {bucket} --prefix {server_dir_name}
# python download.py --origin ".." --bucket "smart-passage-logic" --prefix "data/landing"
```

## 2. Pre-Processing
The pre-processing process involve:
- Copying the data from ```smart_passage/data/landing``` folder to ```smart_passage/data/processed```
- Unify data from different sources into one
- Altering the labels of opensource data to match model config

To run the pre-processing step
1. go to smart_passage/src/pre-processing/config/data_annotation.yaml to change the config yaml file
2. Edit the config file so that we can configure the original labels, ignore some classes if we are not interested at them
3. Run the preprocessing.py script to automate the conversion process
```
cd smart_passage/src/pre-processing
python preprocessing.py
```

## 3. Model Training
The model training starts with configuring the configuration file under ```smart_passage/scr/training/config```
You are free to create multiple .yaml file for different parameters or configuration
Once it's once, run the training script with the particular configuration file
```
cd smart_passage/src/training
python train.py --cfg {config file} # e.g. python train.py --cfg default.yaml
```
