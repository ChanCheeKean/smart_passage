import os, glob, time
from datetime import datetime
import json
from dash import html, Output, Input, callback_context
from app import app

@app.callback(
    [
        Output("pg1_indicator_normal", "color"),
        Output("pg1_indicator_warning", "color"),
        Output("left_human_led", "value"),
        Output("safety_human_led", "value"),
        Output("right_human_led", "value"),
        Output("left_object_led", "value"),
        Output("safety_object_led", "value"),
        Output("right_object_led", "value"),
        ],
    Input("pg1_interval_xs", "n_intervals"),
    Input("pg1_clear_warning_bt", "n_clicks"),
)
def update_output(n, n_click):
    ### if button clicked, clear warning light and images ###
    if callback_context.triggered_id == 'pg1_clear_warning_bt':
        if os.listdir("./static/img/"):
            files = glob.glob('./static/img/*')
            for f in files:
                os.remove(f)
        return '#808080', '#808080', 0, 0, 0, 0, 0, 0

    try:
        with open('./static/json/image_output.json', 'r') as f:
            json_data = json.load(f)
        # red warning  
        if (json_data['tailgate_flag'] == 1) | (json_data['antidir_flag'] == 1):
            json_flag = 1
        elif json_data['human_count'] > 0:
            json_flag = 0
        else:
            json_flag = -1

        h_left, h_safety, h_right, o_left, o_safety, o_right = \
            json_data['human_left'], json_data['human_safety'], json_data['human_right'], \
            json_data['object_left'], json_data['object_safety'], json_data['object_right']
            
    except Exception as e:
        json_flag = -1
        h_left, h_safety, h_right, o_left, o_safety, o_right = 0, 0, 0, 0, 0, 0
        # print(f"Load Json Failed, Error: {e}")
    
    if (json_flag == 1):
        return '#808080', 'red', h_left, h_safety, h_right, o_left, o_safety, o_right
    elif json_flag == 0:
        return '#00FF00', '#808080', h_left, h_safety, h_right, o_left, o_safety, o_right
    else:
        return '#808080', '#808080', h_left, h_safety, h_right, o_left, o_safety, o_right

@app.callback(
    Output("pg1_img_archive", "children"),
    Input("pg1_interval_1s", "n_intervals")
)
def update_output(n):
    files = os.listdir("./static/img/")
    if files:
        sort_files = sorted([f.split('.')[0] for f in files], reverse=True)
        child_list = []
        time.sleep(0.8)

        # display image in dashboard
        for f in  sort_files[:8]:
             child_list.append(
                 html.Div(
                     children=[
                         html.Span(datetime.fromtimestamp(int(f)).strftime('%Y-%m-%d %H:%M:%S'), className='mb-2 fs-4 text-info'),
                         html.Img(src=f'./static/img/{f}.jpg', alt='image', width=400, height=300)
                     ], 
                     className='my-3'
                 )
             )

        # delete extra image
        if len(sort_files) > 10:
            for f in sort_files[10:]:
                os.remove(f'./static/img/{f}.jpg')
        return child_list
    else:
        return None
 