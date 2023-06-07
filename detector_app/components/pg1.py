import os, glob
import cv2
import time, json
from flask import Response
from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_daq as daq
from app import server, app
from components import pg1_callback
from utils.video_loader import ImageLoader
from utils.cv_helper import (
    DeepSortTracker, 
    VitModelLoader, 
    plot_gate_roi, 
    plot_image, 
    tailgating_detection, 
    update_zone_info, 
    detect_dir,
    detect_loiter,
)

### clear images before start ###
if os.listdir("./static/img/"):
    files = glob.glob('./static/img/*')
    for f in files:
        os.remove(f)

### video streaming ###
def video_gen(camera, model, object_tracker):
    while True:
        ret, image = camera.get_frame()
        if ret:
            # pre-processing
            image = cv2.resize(image, camera.img_sz, interpolation=cv2.INTER_AREA)

            # detect and tracking
            results = model.detect(image)
            results = object_tracker.update(image, results)
            
            # update zone in the dictionaries
            info_dict = update_zone_info(results)

            # tailgate detection
            tailgate_flag, lis = tailgating_detection(results, camera.trigger_distance)
            info_dict['tailgate_flag'] = tailgate_flag
            info_dict['tailgate_lis'] = lis

            # anti detection
            anti_flag, camera.id_paid, camera.id_complete, lis = detect_dir(
                results, camera.id_paid, camera.id_complete, paid_zone='right')
            info_dict['antidir_flag'] = anti_flag
            info_dict['antidir_lis'] = lis

            # update passenger count
            info_dict['passenger_count'] = len(camera.id_complete)

            # loitering
            loiter_flag, lis = detect_loiter(results, camera.id_stay, camera.stay_limit)
            info_dict['loiter_flag'] = loiter_flag
            info_dict['loiter_lis'] = lis

            # plotting
            flag = any((anti_flag, tailgate_flag, loiter_flag))
            plot_image(image, results, camera.font_size, model.labels, camera.mm_per_pixel, flag)
            
            # plot roi area
            if camera.plot_roi:
                plot_gate_roi(image)
            
            # save video
            if camera.save_video:
                camera.out_writter.write(image)

            # log info
            with open('./static/json/image_output.json', 'w') as f:
                json.dump(info_dict, f)

            # capture image if violation
            if flag:
                cv2.imwrite(f"./static/img/{int(time.time())}.jpg", image)

            _, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@server.route('/video_feed')
def video_feed():
    return Response(
        video_gen(ImageLoader(), VitModelLoader(), DeepSortTracker()),
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
        dcc.Interval(id='pg1_interval_1s', interval=1.5*1000, n_intervals=0),
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
