import os
import json
from glob import glob
import pandas as pd
import cv2
from dash import Input, Output, State, no_update, callback_context
from app import app

# initialize data path
processed_data_path = os.path.join(".", "data", "processed")

### libraries dropdown ###
@app.callback(
    [
        Output("validate_library_dropdown", "options"),
        Output("validate_library_dropdown", "value"),
        ],
    Input("url", "pathname"),
)
def return_dropdown_options(_):
    # get the list of subfolders
    dir_list = glob(os.path.join(processed_data_path, "*"), recursive=True)
    dir_list = [os.path.split(x)[-1] for x in dir_list]
    options = [{'label': x,'value' : x} for x in dir_list]
    return options, dir_list[0]

# generate result
@app.callback(
    Output("validate_dummy", "children"),
    Input("validate_but", "n_clicks"),
    Input("validate_library_dropdown", "value"),
)
def return_validate_result(n1, lib):
    if n1 == 0:
        return no_update

    # get the location of tehe ground truth
    img_data_path = os.path.join(processed_data_path, lib, 'images')
    label_data_path = os.path.join(processed_data_path, lib, 'labels')

    # extact all img files
    img_files = os.listdir(img_data_path)
    label_files = os.listdir(label_data_path)

    # check if file exists
    if label_files:
        sorted_names = sorted([f.split('.')[0] for f in label_files], reverse=True)
        pd_lis = []

        # loop each test file one by oe
        for name in sorted_names:
            img_pth = os.path.join(img_data_path, f"{name}.jpg")
            txt_pth = os.path.join(label_data_path, f"{name}.json")
            temp_dic = {}
            with open(txt_pth, 'r') as f:
                json_data = json.load(f)
            
            if isinstance(json_data, list):

                # predict data
                img = cv2.imread(img_pth)
                img = cv2.resize(img, (720, 1280), interpolation=cv2.INTER_AREA)
                temp_dic[f'img_shape'] = img.shape

                pred_data = []
                for data in pred_data:
                    if data.get('class', None) == '0':
                        zone = data.get('zone', None)
                        temp_dic[f'pred_human_zone_{zone}'] = 1

                    elif data.get('class', None) == '1':
                        pass

                # label data
                for data in json_data:
                    temp_dic['name'] = name
                    if data.get('class', None) == '0':
                        zone = data.get('zone', None)
                        temp_dic[f'human_zone_{zone}'] = 1
                    elif data.get('class', None) == '1':
                        pass

                pd_lis.append(temp_dic)
        
        df = pd.DataFrame(pd_lis).fillna(0)
        print(df)

    return no_update