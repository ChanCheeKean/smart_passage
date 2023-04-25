#!/usr/bin/python3
import paho.mqtt.client as mqtt
import time
import json
import threading

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

    def connect(self, address="localhost", port=1883, username="user", password="password")
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

    def on_connect(self, client, userdata, flags, rc):
        print("on_connect = " + str(rc))

    def on_subscribe(self, client, obj, mid, granted_qos):
        print("Subscribed: " + str(mid) + " " + str(granted_qos))

    def on_message(self, client, obj, msg):
        print("Message received from {}, payload = {}".format(msg.topic, str(msg.payload)))
        if msg.topic == "savari/request/stop":
            data = get_json_from_payload(msg.payload)
            print("on_message() - Stop requested; reason = " + str(data["reason"]))
            self.stop()
        if msg.topic == "transcity/device/passage/obstacle_control/obstacle_state":
            data = get_json_from_payload(msg.payload)

    def run(self):
        self.client.loop_start()
        while not self.stop_loop:
            time.sleep(1)
        self.client.loop_stop()
        print("run() - Stop Loop")

mqtt = MQTTClient()
mqtt.connect(address="192.168.96.94", port=22, username="mqtt", password="ThalesPass!")
mqtt.publish("test/topic", 'test')