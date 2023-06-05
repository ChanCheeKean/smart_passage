import cv2
from flask import Response
from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_daq as daq
from app import server, app
from utils.video_loader import ImageLoader
from utils.cv_helper import VitModelLoader, plot_gate_roi

### video streaming ###
def video_gen(camera, model):
    while True:
        ret, frame = camera.get_frame()
        if ret:
            image = cv2.resize(frame, camera.img_sz, interpolation=cv2.INTER_AREA)
            image, results = model.detect(image)

            if camera.save_video:
                camera.out_writter.write(image)

            if camera.plot_roi:
                plot_gate_roi(image)

            _, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@server.route('/video_feed')
def video_feed():
    return Response(
        video_gen(ImageLoader(), VitModelLoader()),
        mimetype='multipart/x-mixed-replace; boundary=frame')

video_container = html.Img(
    src="/video_feed", 
    id='pg1_video',
    style={'max-width': '100%', 'height' : '70vh', 'width' : '100%'},)


### indicator ###
indicator_container = html.Div(
    children=[
        daq.Indicator(
            id='pg1_indicator_normal',
            value=True,
            height=100,
            color = '#808080',
            className='my-1 ps-2',
        ),

        daq.Indicator(
            id='pg1_indicator_warning',
            value=True,
            height=100,
            color='#808080',
            className='my-3 ps-2'
        ),

        dbc.Button(
            children="Clear Alert", 
            id='pg1_clear_warning_bt',
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
        dbc.CardHeader(html.Span('Image Archive'), className='card_header', style={'height' : '5vh'}),
        dbc.CardBody(
            children=[html.Div(id='pg1_img_archive')], 
            style={'max-height' : '85vh', 'height' : '85vh', 'overflow-y' : 'scroll'}
        ),
    ],
    className='card'
)


### object count container ###
count_container = dbc.Row(
                      children=[
                          dbc.Col(
                              [
                                  html.H2("Left Zone", className='text-left px-5', style={'color':'#66ff99', 'font-size':'1.5rem'}),
                                  daq.LEDDisplay(
                                      id='left_human_led',
                                      label={'label': "Human", 'style': {'color' : '#63e5ff', 'font-size' : '1.5rem'}},
                                      labelPosition='bottom',
                                      backgroundColor='#323232',
                                      size=50,
                                      color='red',
                                      className='d-inline-block mx-2',
                                      value=0),
                                  daq.LEDDisplay(
                                      id='left_object_led',
                                      label={'label': "Object", 'style': {'color' : '#63e5ff', 'font-size' : '1.5rem'}},
                                      labelPosition='bottom',
                                      backgroundColor='#323232',
                                      size=50,
                                      className='d-inline-block mx-2',
                                      value=0),
                              ],
                              className='ps-5',
                              width=4,),

                          dbc.Col(
                              [
                                  html.H2("Safety Zone", className='text-left px-3', style={'color':'#66ff99', 'font-size':'1.5rem'}),
                                  daq.LEDDisplay(
                                      id='safety_human_led',
                                      label={'label': "Human", 'style': {'color' : '#63e5ff', 'font-size' : '1.5rem'}},
                                      labelPosition='bottom',
                                      backgroundColor='#323232',
                                      size=50,
                                      color='red',
                                      className='d-inline-block mx-2',
                                      value=0),
                                  daq.LEDDisplay(
                                      id='safety_object_led',
                                      label={'label': "Object", 'style': {'color' : '#63e5ff', 'font-size' : '1.5rem'}},
                                      labelPosition='bottom',
                                      backgroundColor='#323232',
                                      size=50,
                                      className='d-inline-block mx-2',
                                      value=0),
                              ],
                              className='ps-5',
                              width=4),

                          dbc.Col(
                              [
                                  html.H2("Right Zone", className='text-left px-4', style={'color':'#66ff99', 'font-size':'1.5rem'}),
                                  daq.LEDDisplay(
                                      id='right_human_led',
                                      label={'label': "Human", 'style': {'color' : '#63e5ff', 'font-size' : '1.5rem'}},
                                      labelPosition='bottom',
                                      backgroundColor='#323232',
                                      size=50,
                                      color='red',
                                      className='d-inline-block mx-2',
                                      value=0),
                                  daq.LEDDisplay(
                                      id='right_object_led',
                                      label={'label': "Object", 'style': {'color' : '#63e5ff', 'font-size' : '1.5rem'}},
                                      labelPosition='bottom',
                                      backgroundColor='#323232',
                                      size=50,
                                      className='d-inline-block mx-2',
                                      value=0),
                              ],
                              className='ps-5',
                              width=4),
                      ],
                      style={'height' : '20vh'},
                      className='pt-3'
                  )


### final layout ###
layout = html.Div(
    children = [
        dcc.Interval(id='pg1_interval_xs', interval=0.1*1000, n_intervals=0),
        dcc.Interval(id='pg1_interval_1', interval=1.8*1000, n_intervals=0),
        html.H2("SmartGate Passage", className='text-info text-center my-1 fs-1'),
        dbc.Row(
            children=[
                dbc.Col(indicator_container, width=1),
                dbc.Col([video_container, count_container], width=8),
                dbc.Col(archive_card, width=3),
                ],
            className='my-3 mx-2'),
    ]
)

# !ffmpeg -i ./static/test_video/saved_output.avi ./static/test_video/saved_outputmp4 -y
