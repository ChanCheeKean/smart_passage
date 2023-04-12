from app import server, app
from dash import html, dcc, Output, Input
import dash_bootstrap_components as dbc
import dash_daq as daq
from flask import Response
import cv2

### embeded video ###
class VideoCamera(object):
    def __init__(self, path=0):
        self.video = cv2.VideoCapture(path)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, image = self.video.read()
        image = cv2.flip(image, 1)
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        with open('./static/image_output.json', 'w') as f:
            f.write('-1')
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
    style={'max-width': '100%', 'height' : '75vh', 'width' : '100%'},
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
            height=80,
            className='my-1'
        ),

        daq.Indicator(
            id='main_indicator_warning',
            value=True,
            height=80,
            className='my-3'
        )
    ], 
    style={'margin-left': 'auto', 'margin-right': 'auto'}
)

### setting container ###
setting_card = dbc.Card(
    children=[
        dbc.CardHeader(html.Span('Setting'), className='card_header', style={'height' : '5vh'}),
        dbc.CardBody(html.Span('To Be Included', className='text-white'), style={'height' : '70vh'}),
    ],
    className='card',
)

### final layout ###
def make_layout():
    layout = html.Div(
        children = [
            dcc.Interval(id='main_interval', interval=0.1*1000, n_intervals=0),
            dbc.Row(
                children=[
                    dbc.Col(indicator_container, width=1),
                    dbc.Col(video_container, width=8),
                    dbc.Col(setting_card, width=3),
                    ],
                className='my-3 mx-2'),
        ]
    )
    return layout

@app.callback(
    [
        Output("main_indicator_normal", "color"),
        Output("main_indicator_warning", "color"),
        ],
    Input("main_interval", "n_intervals")
)
def update_output(n):
    with open('./static/image_output.json', 'r') as f:
        json_data = str(f.read())

    if json_data == '-1':
        return '#00FF00', '#808080'
    elif json_data == '1':
        return '#808080', 'red'
    else:
        return '#808080', '#808080'