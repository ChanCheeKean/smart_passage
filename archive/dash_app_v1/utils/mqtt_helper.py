#!/usr/bin/python3
import paho.mqtt.client as mqtt
import time
import json
import threading
import time

def get_json_from_payload(payload):
    return json.loads(str(payload)[2:len(str(payload)) - 1])

class MQTTClient(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.stop_loop: bool = False
        self.client = mqtt.Client()
        self.thread_mqtt = threading.Thread()
        self.client.on_connect = self.on_connect
        self.client.on_subscribe = self.on_subscribe
        self.client.on_message = self.on_message

    def __del__(self):
        self.client.loop_stop()

    def stop(self) -> None:
        self.stop_loop = True

    def publish(self, topic, msg):
        data = json.dumps(msg)
        self.client.publish(topic, data)

    def connect(self, address="localhost", port=1883, username="user", password="password"):
        try:
            self.client.username_pw_set(username, password)
            self.client.connect(address, port, 60)
            self.address = address

            # print("Subscribing to topic test/topic")
            # self.client.subscribe("test/topic", 0)
            # print("Subscribing to topic savari/request/stop")
            # self.client.subscribe("savari/request/stop", 0)
            # print("Subscribing to topic transcity/device/passage/obstacle_control/obstacle_state")
            # self.client.subscribe("transcity/device/passage/obstacle_control/obstacle_state", 0)

        except Exception as e:
            print("Connection failed = " + (str(e)))
    
    def subscribe(self, topic, qos=0):
        return self.client.subscribe(topic, qos)

    def on_connect(self, client, userdata, flags, rc):
        print("on_connect = " + str(rc))

    def on_subscribe(self, client, obj, mid, granted_qos):
        print("Subscribed: " + str(mid) + " " + str(granted_qos))

    def on_message(self, client, obj, msg):
        payload = get_json_from_payload(msg.payload)
        print("Message received from {}, payload = {}".format(msg.topic, str(payload)))

    def run(self): 
        self.client.loop_start()
        while not self.stop_loop:
            time.sleep(1)
        self.client.loop_stop()
        print("run() - Stop Loop")


mqtt = MQTTClient()
# mqtt.client.username_pw_set(username='nano', password='yahboom')
mqtt.connect(address="192.168.96.94", port=1883)
mqtt.publish("gate/camera_info", {"Camera Info" : f"Connection Established at {int(time.time())}"})
# mqtt.subscribe("gate/camera_info")
mqtt.client.loop_start()
# while True:
    # time.sleep(1)






