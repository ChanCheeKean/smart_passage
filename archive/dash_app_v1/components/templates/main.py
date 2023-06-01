import cv2
from app import server, app
import jetson.inference
from flask import Response
from dash import html, dcc, Output, Input, State, callback_context
import dash_bootstrap_components as dbc
import dash_daq as daq
import os, glob
import json
import time
from pathlib import Path
from datetime import datetime
from utils import detect_helper
from utils.mqtt_helper import mqtt
# from trackers.multi_tracker_zoo import create_tracker
from utils.sort import Sort
import numpy as np
import threading
import time
# global variable to make the alert stay until the button is clicked
warning_flag = 0

# tracker config
# tracking_method = 'deepocsort'
#tracking_config = f"./trackers/{tracking_method}/configs/{tracking_method}.yaml"
# reid_weights = Path('osnet_x0_25_msmt17.pt')
#deep_tracker = create_tracker(tracking_method, tracking_config, reid_weights, 'cuda', False)
# if hasattr(deep_tracker, 'model'):
    #if hasattr(deep_tracker.model, 'warmup'):
       # deep_tracker.model.warmup()

### clear all archive images ###
if os.listdir("./static/img/"):
    files = glob.glob('./static/img/*')
    for f in files:
        os.remove(f)

### send first message to mqtt
# with open('./static/image_output.json', 'r') as f:
   # mqtt.publish("gate/camera_info", json.load(f))

### helper for video loader
class FreshestFrame(threading.Thread):
	def __init__(self, capture, name='FreshestFrame'):
		self.capture = capture
		assert self.capture.isOpened()

		# this lets the read() method block until there's a new frame
		self.cond = threading.Condition()

		# this allows us to stop the thread gracefully
		self.running = False

		# keeping the newest frame around
		self.frame = None

		# passing a sequence number allows read() to NOT block
		# if the currently available one is exactly the one you ask for
		self.latestnum = 0

		# this is just for demo purposes		
		self.callback = None
		
		super().__init__(name=name)
		self.start()

	def start(self):
		self.running = True
		super().start()

	def release(self, timeout=None):
		self.running = False
		self.join(timeout=timeout)
		self.capture.release()

	def run(self):
		counter = 0
		while self.running:
			# block for fresh frame
			(rv, img) = self.capture.read()
			assert rv
			counter += 1

			# publish the frame
			with self.cond: # lock the condition for this operation
				self.frame = img if rv else None
				self.latestnum = counter
				self.cond.notify_all()

			if self.callback:
				self.callback(img)

	def read(self, wait=True, seqnumber=None, timeout=None):
		# with no arguments (wait=True), it always blocks for a fresh frame
		# with wait=False it returns the current frame immediately (polling)
		# with a seqnumber, it blocks until that frame is available (or no wait at all)
		# with timeout argument, may return an earlier frame;
		#   may even be (0,None) if nothing received yet

		with self.cond:
			if wait:
				if seqnumber is None:
					seqnumber = self.latestnum + 1
				if seqnumber < 1:
					seqnumber = 1
				
				rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
				if not rv:
					return (self.latestnum, self.frame)

			return (self.latestnum, self.frame)

### embeded video ###
class VideoCamera(object):
    def __init__(self, path='/dev/video0'):

        # config 
        self.gate_xyxy = {
            'left': 40,
            'top': 400,
            'right': 1140,
            'bottom': 640,
            'left-safety': 300,
            'right-safety' : 780,
            'blocked_left_1': 150,
            'blocked_left_2': 340,
            'blocked_right_1': 780,
            'blocked_right_2': 1020,
            'blocked_top': 400,
        }
        self.gate_xyxy['center'] = (self.gate_xyxy['left'] + self.gate_xyxy['right']) / 2
        self.trigger_distance = 150

        # tracker config
        self.deep_tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.5)
        self.obj_tracker = Sort(max_age=1, min_hits=4, iou_threshold=0.5)
        self.id_store = []

        ### init image output
        self.detect_dict = {
                'human_count' : 0, 
                'human_left' : 0,
                'human_safety': 0,
                'human_right': 0,
                'object_left' : 0,
                'object_safety': 0,
                'object_right': 0,
                'tailgate_flag': -1,
                'oversize_flag': -1,
                'antidir_flag': -1
            }

        # model, substractor and video init
        self.net = jetson.inference.detectNet("ssd-inception-v2", threshold=0.6)
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
                    history=1, varThreshold=2000, detectShadows=False)
        # self.video = cv2.VideoCapture(path)    
        self.cnt = 0
        self.video = cv2.VideoCapture('rtsp://service:Thales1$8o8@192.168.100.108:554/live')
        self.video = FreshestFrame(self.video)
        _, self.ref_img = self.video.read()
        self.ref_img = cv2.resize(self.ref_img, (1280, 720), interpolation=cv2.INTER_AREA)
        
        with open('./static/image_output.json', 'w') as f:
            json.dump(self.detect_dict, f)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        global warning_flag
        self.cnt, image = self.video.read(seqnumber=self.cnt+1)

        if not self.cnt:
            return self.cnt, None, None

        image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
        # image = cv2.flip(image, 1)
        image, info_dict, self.id_store = detect_helper.process_frame(
            image, self.detect_dict.copy(), self.net, self.gate_xyxy, self.trigger_distance, 
            self.deep_tracker, self.id_store, self.ref_img, self.bg_sub, self.obj_tracker, verbose=0
        )
        self.id_store = self.id_store[:100]

        if (info_dict['tailgate_flag'] == 1) | (info_dict['antidir_flag'] == 1):
            cv2.imwrite(f"./static/img/{int(time.time())}.jpg", image)
            warning_flag = 1
        _, jpeg = cv2.imencode('.jpg', image)
        return self.cnt, jpeg.tobytes(), info_dict

def gen(camera):
    while True:
        ret, frame, info = camera.get_frame()
        if not ret:
            continue

        with open('./static/image_output.json', 'r') as f:
            json_data = json.load(f)
            if info != json_data:
                mqtt.publish("gate/camera_info", info)
        
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
    style={'max-width': '100%', 'height' : '70vh', 'width' : '100%'},
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
        dbc.CardHeader(html.Span('Image Archive'), className='card_header', style={'height' : '5vh'}),
        dbc.CardBody(
            children=[html.Div(id='main_img_archive')], 
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
def make_layout():
    layout = html.Div(
        children = [
            dcc.Interval(id='main_interval_short', interval=0.1*1000, n_intervals=0),
            dcc.Interval(id='main_interval_1s', interval=1.8*1000, n_intervals=0),
            html.H2("TailGate Monitoring Dashboard", className='text-info text-center my-1 fs-1'),
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

        tg_flag = json_data['tailgate_flag']
        at_flag = json_data['antidir_flag']
        
        if (tg_flag == 1) | (at_flag == 1):
            json_flag = 1
        else:
            json_flag = tg_flag

        # print(at_flag, json_flag)

        h_left, h_safety, h_right, o_left, o_safety, o_right = \
            json_data['human_left'], json_data['human_safety'], json_data['human_right'], \
            json_data['object_left'], json_data['object_safety'], json_data['object_right']
            
    except Exception as e:
        json_flag = -1
        h_left, h_safety, h_right, o_left, o_safety, o_right = 0, 0, 0, 0, 0, 0
        # print(f"Load Json Failed, Error: {e}")
    
    if (json_flag == 1) | (warning_flag):
        return '#808080', 'red', False, False, h_left, h_safety, h_right, o_left, o_safety, o_right
    elif json_flag == 0:
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
 
