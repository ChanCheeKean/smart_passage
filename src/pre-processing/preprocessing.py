import os
from glob import glob
import yaml
import shutil
from tqdm import tqdm

# loading configuration from config yaml
config_file = os.path.join(".", "config", "data_annotation.yaml")
with open(config_file, "r") as stream:
    pull_config = yaml.safe_load(stream)
replace_dict = pull_config['replace_label']
export_dir = os.path.join("../", "data", pull_config['destination'])

# to alter the label class in the data
data_dir = os.path.join("../", "data", pull_config['origin'])
data_list = list(replace_dict.keys())
class_dict = {'train': {}, 'valid': {}, 'test': {}}

def main():
    data_split_dir = ['train', 'valid', 'test']
    for _p in data_split_dir:

        # create directories if not exists
        export_p = os.path.join(export_dir, _p)
        if not os.path.exists(export_p):
            os.makedirs(os.path.join(export_dir, _p, "images"))
            os.makedirs(os.path.join(export_dir, _p, "labels"))

        # looping through each dataset and alter the label
        for _d in data_list:
            existing_data = [os.path.split(x)[-1] for x in glob(os.path.join(export_dir, "**"), recursive=True) if ".jpg" in x]

            # if label not there then just write empty file
            if os.path.exists(os.path.join(data_dir, _d, _p)):
                image_dir = os.path.join(data_dir, _d, _p, 'images')
                label_dir = os.path.join(data_dir, _d, _p, 'labels')
                r_dict = replace_dict[_d]

                # get all the image and label name in the directory
                names = [
                    os.path.split(x)[-1].rsplit('.',1)[0] 
                        for x in glob(os.path.join(label_dir, "*"), recursive=True) 
                            if ".txt" in x
                ]
                
                # move images and labels to final destionation
                for name in tqdm(names):
                    jpg_file = os.path.join(image_dir, f'{name}.jpg')
                    txt_file = os.path.join(label_dir, f'{name}.txt')

                    # to ensure image exists
                    if os.path.exists(jpg_file) and f'{name}.jpg' not in existing_data:
                        if os.path.exists(txt_file):
                            ### read original label ###
                            with open(txt_file) as f:
                                text = [t.split(" ") for t in f.readlines()]

                            ### skip this file if contains contaiminated labels ###
                            text_labels = [t[0] for t in text]
                            if replace_dict[_d].get('skip', None):
                                if any([int(t) in replace_dict[_d]['skip'] for t in text_labels]):
                                    # print(f"Skipping contaiminated file: {name}")
                                    continue

                            ### replace label ###
                            text_copy = text.copy()
                            for c, t in enumerate(text):
                                for key, value in r_dict.items():
                                    t[0] = text_copy[c][0].replace(str(key), str(value))

                                # add counter
                                if int(t[0]) in class_dict[_p]:
                                    class_dict[_p][int(t[0])] += 1
                                else:
                                    class_dict[_p][int(t[0])] = 0

                            ### save label output ###
                            text = [" ".join(t) for t in text]
                            with open(os.path.join(export_p, 'labels', f'{name}.txt'), 'w') as f:
                                if len(text) == 0:
                                    f.write("")
                                else:
                                    for t in text:
                                        f.write(f"{t}")
                        else:
                            print(f"Annotation not found for {name}, saving empty label")
                            with open(os.path.join(export_p, 'labels', f'{name}.txt'), 'w') as f:
                                f.write("")

                        ### save image ###
                        shutil.copy(jpg_file, os.path.join(export_p, 'images', f'{name}.jpg'))

                    else:
                        print(f"{name}.jpg not found or repeated")

    print(f"Total Data: {class_dict}")
                    
if __name__ == '__main__':
    main()