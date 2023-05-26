import os, json, random
import cv2
from glob import glob
import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report
from dash import Input, Output, State, no_update, callback_context
from app import app
from utils.plot_helper import create_bar, create_pie
# initialize data path
processed_data_path = os.path.join(".", "data", "processed")
gate_xyxy = {
    'top': 400,
    'bottom': 620,
    'left': 80,
    'right': 1150,
    'left-safety': 280,
    'right-safety':800
}

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
    Output("validate_store", "children"),
    Input("validate_but", "n_clicks"),
    Input("validate_library_dropdown", "value"),
)
def return_validate_result(n1, lib):
    # if n1 == 0:
    #     return no_update
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

                pred_data = [[500, 200, 600, 500, 0]]
                for data in pred_data:
                    pred_class = data[-1]
                    pred_center = (data[0] + data[3]) / 2
                    if pred_center <= gate_xyxy['left-safety']:
                        zone = 0
                    elif pred_center <= gate_xyxy['right-safety']:
                        zone = 1
                    else:
                        zone = 2

                    if str(pred_class) == '0':
                        # temporary
                        zone = random.choice([0, 1, 2])
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
        
        column_names = [
            'name', 'human_zone_0', 'human_zone_1', 'human_zone_2',
            'pred_human_zone_0', 'pred_human_zone_1', 'pred_human_zone_2'
        ]
        df = pd.DataFrame(pd_lis).fillna(0)
        miss_col = set(column_names) - set(df.columns)
        for col in miss_col:
            df[col] = 0
        df = df[column_names]
        return df.to_json(orient='split')

    return no_update

# generate graph
@app.callback(
    [
        Output('validate_bar', 'figure'),
        Output('validate_pie', 'figure'),
        ],
    Input('validate_store', 'children')
)
def update_graph(data_json):
    if data_json:
        df = pd.read_json(data_json, orient='split')
        cols = ['human_zone_0', 'human_zone_1', 'human_zone_2']
        ground_truth = df[cols].values.tolist()
        predictions = df[['pred_' + col for col in cols]].values.tolist()
        report = classification_report(ground_truth, predictions, output_dict=True) 
        df_report = pd.DataFrame(report).transpose()

        # generate bar chart
        cols = ['precision', 'recall', 'f1-score']
        df_bar = df_report.iloc[[0, 1, 2]][cols].unstack().reset_index().copy()
        df_bar.columns = ['metric', 'zone', 'score']
        df_bar['zone'] = df_bar['zone'].astype(int)
        bar_fig = create_bar(df_bar, 'score', 'metric', 'zone', 'group', 'Score', 'Metric')

        # generate pie chart
        df_pie = df_report.iloc[[0, 1, 2]][['support']].unstack().reset_index().copy()
        df_pie.columns = ['metric', 'zone', 'score']
        df_pie.sort_values('zone', ascending=True, inplace=True)
        df_pie['zone'] = df_pie['zone'].astype(int)
        pie_fig = create_pie(df_pie['zone'].values, df_pie['score'].values)
        return bar_fig, pie_fig
    
    return no_update, no_update