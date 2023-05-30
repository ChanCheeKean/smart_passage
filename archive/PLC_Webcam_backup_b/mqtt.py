#!/usr/bin/python3
from typing import List

import paho.mqtt.client as mqtt
import time
import json
from enum import Enum
import threading


class DoorStatus(Enum):
    NONE = 0
    OPENED_LEFT = 1
    CLOSED = 2
    OPENED_RIGHT = 3
    MOVING = 4


class MQTTStatus(Enum):
    DISCONNECTED = 0
    CONNECTED = 1
    RUNNING = 2


def get_json_from_payload(payload):
    return json.loads(str(payload)[2:len(str(payload)) - 1])


class MQTTClient(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.stop_loop: bool = False
        self.status: MQTTStatus = MQTTStatus.DISCONNECTED
        self.client: mqtt.Client = mqtt.Client()
        self.thread_mqtt: threading.Thread = threading.Thread()

        self.client.on_connect = self.on_connect
        self.client.on_subscribe = self.on_subscribe
        self.client.on_message = self.on_message

        self.door_state_lock: threading.Lock = threading.Lock()
        self.__last_door_state: DoorStatus = DoorStatus.NONE

        self.address: str = ""

    def __del__(self):
        self.client.loop_stop()

    def stop(self) -> None:
        self.stop_loop = True

    def publish(self, topic: str, passenger_list: List) -> None:
        data = json.dumps(passenger_list)
        self.client.publish(topic, data)

    def get_last_door_state(self) -> DoorStatus:
        with self.door_state_lock:
            return self.__last_door_state

    def connect(self, address: str = "localhost", port: int = 1883, username: str = "transcity_passage", password: str = "transcity_passage")\
            -> None:
        try:
            self.client.username_pw_set(username, password)
            self.client.will_set("savari/connection_status", '{"status":"UNEXPECTEDLY_DISCONNECTED"}', 0, False)
            self.client.connect(address, port, 60)
            self.address = address

            self.status = MQTTStatus.CONNECTED
            
        except Exception as e:
            print("Connection failed = " + (str(e)))
            self.status = MQTTStatus.DISCONNECTED

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
            door_state: DoorStatus = DoorStatus.NONE
            if (data["obstacle_state"] == "OBSTACLE_STATE_OPENING_TO_ENTRY" or
                    data["obstacle_state"] == "OBSTACLE_STATE_OPENING_TO_EXIT" or
                    data["obstacle_state"] == "OBSTACLE_STATE_CLOSING_FROM_ENTRY" or
                    data["obstacle_state"] == "OBSTACLE_STATE_CLOSING_FROM_EXIT" or
                    data["obstacle_state"] == "OBSTACLE_STATE_STOPPED" or
                    data["obstacle_state"] == "OBSTACLE_STATE_BLOCKED" or
                    data["obstacle_state"] == "OBSTACLE_STATE_FORCED"):
                door_state = DoorStatus.MOVING
            elif data["obstacle_state"] == "OBSTACLE_STATE_OPENED_ENTRY":
                door_state = DoorStatus.OPENED_LEFT
            elif data["obstacle_state"] == "OBSTACLE_STATE_OPENED_EXIT":
                door_state = DoorStatus.OPENED_RIGHT
            elif data["obstacle_state"] == "OBSTACLE_STATE_CLOSED":
                door_state = DoorStatus.CLOSED

            with self.door_state_lock:
                self.__last_door_state = door_state

    def run(self):
        if self.status == MQTTStatus.DISCONNECTED:
            print("Connection failed, please connect to broker before launch this thread")
        else:
            self.client.loop_start()
            self.status = MQTTStatus.RUNNING
            while not self.stop_loop:
                time.sleep(1)
            self.client.loop_stop()
            self.status = MQTTStatus.CONNECTED
            print("run() - Stop Loop")


if __name__ == '__main__':
    # Only to test the module
    mqtt = MQTTClient()
    mqtt.connect()
    mqtt.start()

    ''' For test
    for i in range(0, 120):
        if mqtt.is_alive():
            print("Door state = {}".format(str(mqtt.get_last_door_state())))
            time.sleep(1)
        else:
            break
            
    passenger1 = {"id": 1, "type": "adult", "zones": []}
    passenger2 = {"id": 2, "type": "object", "zones": [1]}
    passenger3 = {"id": 3, "type": "adult", "zones": [2, 3]}

    for i in range(0, 120):
        if mqtt.is_alive():
            passenger1["id"] += 1
            passenger2["id"] += 1
            passenger3["id"] += 1
            passenger_list = [passenger1, passenger2, passenger3]
            print("Publish passenger list = {}".format(passenger_list))
            mqtt.publish("savari/passenger/list", passenger_list)
            time.sleep(0.1)
        else:
            break
    '''

    mqtt.join()
