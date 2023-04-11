from app import server
from dash import html, dcc
import dash_bootstrap_components as dbc
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
        return jpeg.tobytes(), 1

def gen(camera):
    while True:
        frame, output = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        yield output

@server.route('/video_feed')
def video_feed():
    return Response(
        gen(VideoCamera()),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

video_container = html.Img(
    src="/video_feed", 
    style={'height' : '50vh'},
)

### final layout ###
def make_layout():
    layout = html.Div(
        children = [
            dcc.Location(id="url"),
            dcc.Interval(id='clock_interval_30', interval=30*1000, n_intervals=0),
            dbc.Row(
                children=[
                    dbc.Col(video_container, width=4),
                    dbc.Col(html.Div('test'), width=8)
                    ],
                className='mt-2'),

        ]
    )
    return layout