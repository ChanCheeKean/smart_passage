from app import server, app
from dash import html, dcc, Output, Input, State, callback_context
import dash_bootstrap_components as dbc
import dash_daq as daq
from flask import Response
import cv2
import time
from utils import detect_helper
import jetson.inference
import os, glob
from datetime import datetime
import json

# global variable to make the alert stay until the button is clicked
warning_flag = 0

### clear all archive images ###
if os.listdir("./static/img/"):
    files = glob.glob('./static/img/*')
    for f in files:
        os.remove(f)

### embeded video ###
class VideoCamera(object):
    def __init__(self, path='/dev/video0'):
        self.video = cv2.VideoCapture(path)
        self.gate_xyxy = {
            'left': 150,
            'top': 400,
            'right': 480,
            'bottom': 480
        }
        self.net = jetson.inference.detectNet("ssd-inception-v2", threshold=0.6)
        _, self.ref_img = self.video.read()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        global warning_flag

        _, image = self.video.read()
        image = cv2.flip(image, 1)
        image, flag, info_dict = detect_helper.process_frame(
            image, self.net, self.gate_xyxy, self.ref_img, verbose=0
        )

        if flag == 1:
            cv2.imwrite(f"./static/img/{int(time.time())}.jpg", image)
            warning_flag = 1
        _, jpeg = cv2.imencode('.jpg', image)
        info_dict['flag'] = str(flag)
        return jpeg.tobytes(), info_dict

def gen(camera):
    while True:
        frame, info = camera.get_frame()
        
        with open('./static/image_output.json', 'w') as f:
            json.dump(info, f)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@server.route('/video_feed')
def video_feed():
    return Response(
        gen(VideoCamera()),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

video_container = html.Img(
    src="/video_feed", 
    id='main_video',
    style={'max-width': '100%', 'height' : '80vh', 'width' : '100%'},
)

text_container = html.P(
    id='main_text',
    className='text-info',
    )

### indicator ###
indicator_container = html.Div(
    children=[
        daq.Indicator(
            id='main_indicator_normal',
            value=True,
            height=100,
            color = '#808080',
            className='my-1 ps-2',
        ),

        daq.Indicator(
            id='main_indicator_warning',
            value=True,
            height=100,
            color='#808080',
            className='my-3 ps-2'
        ),

        dbc.Button(
            children="Clear Alert", 
            id='main_clear_warning_bt',
            outline=True, 
            disabled=True,
            color="danger", 
            size="lg", 
            className="ps-2 mt-4 mx-auto",
        ),
    ], 
    style={'margin-left': 'auto', 'margin-right': 'auto'}
)

### images container ###
archive_card = dbc.Card(
    children=[
        dbc.CardHeader(html.Span('Image Archive'), className='card_header', style={'height' : '2vh'}),
        dbc.CardBody(
            children=[html.Div(id='main_img_archive')], 
            style={'max-height' : '88vh', 'height' : '88vh', 'overflow-y' : 'scroll'}
        ),
    ],
    className='card'
)

### object count container ###
count_container = dbc.Row(
                      children=[
                          dbc.Col(
                              [
                                  html.H2("Left Zone", className='text-left px-3', style={'color':'#66ff99', 'font-size':'2rem'}),
                                  daq.LEDDisplay(
                                      id='left_human_led',
                                      label={'label': "Human", 'style': {'color' : '#63e5ff', 'font-size' : '1.2rem'}},
                                      labelPosition='bottom',
                                      backgroundColor='#323232',
                                      size=60,
                                      color='red',
                                      className='d-inline-block mx-2',
                                      value=0),


                                  daq.LEDDisplay(
                                      id='left_object_led',
                                      label={'label': "Object", 'style': {'color' : '#63e5ff', 'font-size' : '1.2rem'}},
                                      labelPosition='bottom',
                                      backgroundColor='#323232',
                                      size=60,
                                      className='d-inline-block mx-2',
                                      value=0),

                              ],
                              width=4,),


                          dbc.Col(
                              [
                                  html.H2("Safety Zone", className='text-left px-3', style={'color':'#66ff99', 'font-size':'2rem'}),

                                  daq.LEDDisplay(
                                      id='safety_human_led',
                                      label={'label': "Human", 'style': {'color' : '#63e5ff', 'font-size' : '1.2rem'}},
                                      labelPosition='bottom',
                                      backgroundColor='#323232',
                                      size=60,
                                      color='red',
                                      className='d-inline-block mx-2',
                                      value=0),

                                  daq.LEDDisplay(
                                      id='safety_object_led',
                                      label={'label': "Object", 'style': {'color' : '#63e5ff', 'font-size' : '1.2rem'}},
                                      labelPosition='bottom',
                                      backgroundColor='#323232',
                                      size=60,
                                      className='d-inline-block mx-2',
                                      value=0),

                              ],
                              width=4),

                          dbc.Col(
                              [
                                  html.H2("Right Zone", className='text-left px-3', style={'color':'#66ff99', 'font-size':'2rem'}),

                                  daq.LEDDisplay(
                                      id='right_human_led',
                                      label={'label': "Human", 'style': {'color' : '#63e5ff', 'font-size' : '1.2rem'}},
                                      labelPosition='bottom',
                                      backgroundColor='#323232',
                                      size=60,
                                      color='red',
                                      className='d-inline-block mx-2',
                                      value=0),

                                  daq.LEDDisplay(
                                      id='right_object_led',
                                      label={'label': "Object", 'style': {'color' : '#63e5ff', 'font-size' : '1.2rem'}},
                                      labelPosition='bottom',
                                      backgroundColor='#323232',

                                      size=60,
                                      className='d-inline-block mx-2',
                                      value=0),

                              ],
                              width=4),
                      ],
                      style={'height' : '20vh'},
                      className='pt-3'
                  )


### final layout ###
def make_layout():
    layout = html.Div(
        children = [
            dcc.Interval(id='main_interval_short', interval=0.1*1000, n_intervals=0),
            dcc.Interval(id='main_interval_1s', interval=1.5*1000, n_intervals=0),
            html.H1("TailGate Monitoring Dashboard", className='text-info text-center my-5 fs-1'),
            dbc.Row(
                children=[
                    dbc.Col(indicator_container, width=1),
                    dbc.Col([video_container, count_container], width=8),
                    dbc.Col(archive_card, width=3),
                    ],
                className='my-3 mx-2'),
        ]
    )
    return layout

@app.callback(
    [
        Output("main_indicator_normal", "color"),
        Output("main_indicator_warning", "color"),
        Output("main_clear_warning_bt", "outline"),
        Output("main_clear_warning_bt", "disabled"),
        Output("left_human_led", "value"),
        Output("safety_human_led", "value"),
        Output("right_human_led", "value"),
        Output("left_object_led", "value"),
        Output("safety_object_led", "value"),
        Output("right_object_led", "value"),
        ],
    Input("main_interval_short", "n_intervals"),
    Input("main_clear_warning_bt", "n_clicks"),
)
def update_output(n, n_click):
    global warning_flag

    ### if button clicked, clear warning light and images ###
    if callback_context.triggered_id == 'main_clear_warning_bt':
        warning_flag = 0
        if os.listdir("./static/img/"):
            files = glob.glob('./static/img/*')
            for f in files:
                os.remove(f)
        return '#808080', '#808080', True, True, 0, 0, 0, 0, 0, 0

    try:
        with open('./static/image_output.json', 'r') as f:
            json_data = json.load(f)

        json_flag = json_data['flag']
        h_left, h_safety, h_right, o_left, o_safety, o_right = \
            json_data['human_left'], json_data['human_safety'], json_data['human_right'], \
            json_data['object_left'], json_data['object_safety'], json_data['object_right']
            
    except:
        json_flag = -1
        h_left, h_safety, h_right, o_left, o_safety, o_right = 0, 0, 0, 0, 0, 0
    
    if (json_flag == '1') | (warning_flag):
        return '#808080', 'red', False, False, h_left, h_safety, h_right, o_left, o_safety, o_right
    elif json_flag == '0':
        return '#00FF00', '#808080', True, True, h_left, h_safety, h_right, o_left, o_safety, o_right
    else:
        return '#808080', '#808080', True, True, h_left, h_safety, h_right, o_left, o_safety, o_right

@app.callback(
    Output("main_img_archive", "children"),
    Input("main_interval_1s", "n_intervals")
)
def update_output(n):
    files = os.listdir("./static/img/")
    if files:
        sort_files = sorted([f.split('.')[0] for f in files], reverse=True)
        child_list = []
        time.sleep(0.5)

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

        return child_list

    else:
        return None
 
