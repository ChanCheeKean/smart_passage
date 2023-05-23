import cv2
from flask import Response
from dash import html
import dash_bootstrap_components as dbc
from app import server, app, video_loader
from components.callbacks import collect_cb
from components.style import general

### Video Camera Object
class VideoCamera(object):
    def __init__(self):
        self.video = video_loader

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
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@server.route('/video_feed')
def video_feed():
    return Response(
        gen(VideoCamera()),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

video_display = html.Img(
    src="/video_feed", 
    id='main_video',
    style={'max-width': '100%', 'height' : '80vh', 'width' : '100%'},
)

### button
delete_but = dbc.Button(
    children="Delete Last ✃", 
    id='delete_but',
    outline=False, 
    disabled=False,
    color="danger", 
    size="lg", 
    className="w-100",
    n_clicks=0
)

capture_but = dbc.Button(
    children="Capture ❂", 
    id='capture_but',
    outline=False, 
    disabled=False,
    color="primary", 
    size="lg", 
    className="w-100",
    n_clicks=0
)

input_butt = dbc.Row(
    children=[
        dbc.Col([delete_but], width=3),
        dbc.Col([capture_but], width=9),
        ],
    className='mt-2')

### card container for image
library_dropdown = dbc.Select(
    id="collect_library_dropdown",
    style=general.dropdown_selector,
    className='d-inline mx-1',
)

library_select = html.Div(
    children=[
        html.P("Export Library: ", className="fs-6 fw-bold d-inline"), 
        library_dropdown
    ],
    className='d-inline',
)

video_card = dbc.Card(
    children=[
        dbc.CardHeader(library_select),
        dbc.CardBody(
            children=[video_display, input_butt], 
        ),
    ],
)

### images container ###
archive_card = dbc.Card(
    children=[
        dbc.CardHeader(html.Span('Saved Images'), className='my-2'),
        dbc.CardBody(
            children=[html.Div(id='img_archive')], 
            style={'height' : '88vh', 'overflow-y' : 'scroll'}
        ),
    ],
)

### final layout
layout = html.Div(
        children = [
            dbc.Row(
                children=[
                    dbc.Col([video_card], width=9),
                    dbc.Col([archive_card], width=3),
                    ],
                className='my-1'),
        ]
    )
