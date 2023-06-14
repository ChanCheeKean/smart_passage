import os, glob, time
from datetime import datetime
import json
from dash import html, Output, Input, State, callback_context, no_update
import dash_bootstrap_components as dbc
from app import app

@app.callback(
    [
        Output("left_human_led", "value"),
        Output("safety_human_led", "value"),
        Output("right_human_led", "value"),
        Output("left_object_led", "value"),
        Output("safety_object_led", "value"),
        Output("right_object_led", "value"),
        Output("pg1_alert", "children"),
        Output("pg1_indicator_normal", "color"),
        Output("pg1_indicator_warning", "color"),
        ],
    Input("pg1_interval_xs", "n_intervals"),
    Input("pg1_clear_warning_bt", "n_clicks"),
    State("pg1_alert", "children")
)
def update_output(_int, _clicks, alert_state):

    # default scenario
    h_left, h_safety, h_right, o_left, o_safety, o_right = [0] * 6
    alert = html.Span("No Alert", className='text-secondary')
    ind_normal = '#808080'
    ind_warning = '#808080'

    # if button clicked, clear warning light and images #
    if callback_context.triggered_id == 'pg1_clear_warning_bt':
        if os.listdir("./static/img/"):
            files = glob.glob('./static/img/*')
            for f in files:
                os.remove(f)

    else:    
        # get info from json file #
        try:
            with open('./static/json/image_output.json', 'r') as f:
                json_data = json.load(f)

            # count indicator
            h_left, h_safety, h_right, o_left, o_safety, o_right = \
                json_data['human_left'], json_data['human_safety'], json_data['human_right'], \
                json_data['object_left'], json_data['object_safety'], json_data['object_right']
            
            # event flag
            tg_f, at_f, lt_f = json_data['tailgate_flag'], json_data['antidir_flag'], json_data['loiter_flag']
            
            # if any flag
            if any(tg_f, at_f, lt_f):
                ind_warning = '#ff0303'
                if isinstance(alert_state, list):
                    alert = no_update
                else:
                    if tg_f:
                        event = "Tailgate "
                    elif lt_f:
                        event = "Loitering "
                    else: 
                        event = "Wrong Direction "

                    alert = [
                        dbc.Row([
                            dbc.Col(dbc.Spinner(color="danger", type="grow", spinner_style={'width': "6vh", 'height': "6vh"}), width=2),
                            dbc.Col([
                                html.Div(event, className='text-danger fs-4'),   
                                html.Span("detected at ", className='text-white'), 
                                html.Span("Exit A: Gate 1", className='text-info')
                            ], 
                            width=10),
                        ])
                    ]
            else:
                if json_data['human_gate_count'] > 0:
                    ind_warning = '#00FF00'
                
        except Exception as e:
            # print(f"Load json Failed, Error: {e}")
            pass
            
    return h_left, h_safety, h_right, o_left, o_safety, o_right, alert, ind_normal, ind_warning

@app.callback(
    Output("pg1_img_archive", "children"),
    Input("pg1_interval_1s", "n_intervals")
)
def update_output(n):
    files = os.listdir("./static/img/")
    if files:
        sort_files = sorted([f.split('.')[0] for f in files if "jpg" in f], reverse=True)
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