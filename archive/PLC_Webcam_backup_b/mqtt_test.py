# pip install paho-mqtt

import paho.mqtt.client as mqtt
import json
import threading

def get_json_from_payload(payload):
    return json.loads(str(payload)[2:len(str(payload)) - 1])

class MQTTClient(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.stop_loop: bool = False
        self.client: mqtt.Client = mqtt.Client()
        self.thread_mqtt: threading.Thread = threading.Thread()
        self.client.on_connect = self.on_connect
        self.client.on_subscribe = self.on_subscribe
        self.client.on_message = self.on_message
        self.door_state_lock: threading.Lock = threading.Lock()
        self.address: str = ""

    def __del__(self):
        self.client.loop_stop()

    def stop(self) -> None:
        self.stop_loop = True

    def publish(self, topic, passenger_list) -> None:
        data = json.dumps(passenger_list)
        self.client.publish(topic, data)

    def connect(self, address: str = "localhost", port: int = 1883, username: str = "transcity_passage", password: str = "transcity_passage")\
            -> None:
        try:
            self.client.username_pw_set(username, password)
            self.client.will_set("savari/connection_status", '{"status":"UNEXPECTEDLY_DISCONNECTED"}', 0, False)
            self.client.connect(address, port, 60)
            self.address = address
        except Exception as e:
            print("Connection failed = " + (str(e)))

    def on_connect(self, client, userdata, flags, rc):
        print("on_connect = " + str(rc))
        self.client.publish("savari/connection_status", '{"status":"CONNECTED"}')
        print("Subscribing to topic savari/request/stop")
        self.client.subscribe("savari/request/stop", 0)
        print("Subscribing to topic transcity/device/passage/obstacle_control/obstacle_state")
        self.client.subscribe("transcity/device/passage/obstacle_control/obstacle_state", 0)

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
        self.client.loop_stop()
        print("run() - Stop Loop")

mqtt_client = mqtt.MQTTClient()
mqtt_client.connect()
mqtt_client.start()
