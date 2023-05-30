"""

Program for LDP webcam
Author : L.JI

TODO: Tracking threshold a revoir
TODO: int width
TODO: time for warmup
"""

import jetson.inference
import jetson.utils
import cv2
import numpy as np
import time
import json
from status import Status
from blob import Blob
# from blob import BlobType
from tracking import Tracker, BlobType
from typing import List
import acquisition

''' Coordonnée haut gauche et bas droit du rectangle ROI de la gate '''
g_gate_roi = []

''' Ligne vertical pour la zone de safety gauche '''
g_safety_left = None

''' Ligne vertical pour la zone de safety droite '''
g_safety_right = None

''' Ligne horizontal pour la hauteur des pieds '''
g_threshold_height_foot = None

''' Use for get all ROI, depend how many click, fill the right variable '''
g_nb_click = 0

''' Status of Passage => Define all presence '''
g_status = Status()

''' img croped for left zone '''
g_img_left_zone = None
g_img_safety_zone = None
g_img_right_zone = None
g_img_object_zone = None

''' List of all object detected in gate '''
g_list_object = []
g_list_human = []

''' Time associated with reference image '''
g_time_reference_image: float = -1

g_marge_enlarge_erase_zone = 50


def load_gate_roi(file_name: str = "roi_gate.json") -> bool:
    """
    Load gate ROI configuration from file_name
    :param file_name:
    :return: true if configuration is done, false if file is not found
    """
    global g_gate_roi, g_safety_left, g_safety_right, g_threshold_height_foot
    try:
        with open(file_name, "r") as f:
            data = json.load(f)
            # In case where box is not define in same order
            gate_box = data["gate_box"]
            x_min = min(gate_box[0][0], gate_box[1][0])
            x_max = max(gate_box[0][0], gate_box[1][0])
            y_min = min(gate_box[0][1], gate_box[1][1])
            y_max = max(gate_box[0][1], gate_box[1][1])
            g_gate_roi.append((x_min, y_min))
            g_gate_roi.append((x_max, y_max))
            g_safety_left = data["safety_left"]
            g_safety_right = data["safety_right"]
            g_threshold_height_foot = data["threshold_foot"]
        return True
    except Exception:
        print("File not found or not available")
        return False


def is_in_zone(zone_left, zone_right, line) -> bool:
    return zone_left < line < zone_right


def determine_presence_zone(blobs) -> None:
    global g_gate_roi, g_safety_left, g_safety_right
    # Detection simple
    # Zone 2
    for blob in blobs:
        if blob.in_gate:
            if is_in_zone(blob.left, blob.right, g_safety_right) and is_in_zone(blob.left, blob.right, g_safety_left):
                if blob.type == BlobType.HUMAN:
                    g_status.person_in_zone_safety += 1
                    g_status.person_in_zone_2 += 1
                    g_status.person_in_zone_1 += 1
                elif blob.type == BlobType.OBJECT:
                    g_status.object_in_zone_safety += 1
                    g_status.object_in_zone_2 += 1
                    g_status.object_in_zone_1 += 1
                blob.in_roi = True
            elif is_in_zone(blob.left, blob.right, g_safety_right):
                if blob.type == BlobType.HUMAN:
                    g_status.person_in_zone_safety += 1
                    g_status.person_in_zone_1 += 1
                if blob.type == BlobType.OBJECT:
                    g_status.object_in_zone_safety += 1
                    g_status.object_in_zone_1 += 1
                blob.in_roi = True
            elif is_in_zone(blob.left, blob.right, g_safety_left):
                if blob.type == BlobType.HUMAN:
                    g_status.person_in_zone_safety += 1
                    g_status.person_in_zone_2 += 1
                if blob.type == BlobType.OBJECT:
                    g_status.object_in_zone_safety += 1
                    g_status.object_in_zone_2 += 1
                blob.in_roi = True
            elif is_in_zone(g_gate_roi[0][0], g_safety_left, blob.left) or is_in_zone(g_gate_roi[0][0], g_safety_left,
                                                                                      blob.right):
                if blob.type == BlobType.HUMAN:
                    g_status.person_in_zone_2 += 1
                if blob.type == BlobType.OBJECT:
                    g_status.object_in_zone_2 += 1
                blob.in_roi = True
            elif is_in_zone(g_safety_left, g_safety_right, blob.left) or is_in_zone(g_safety_left, g_safety_right,
                                                                                    blob.right):
                if blob.type == BlobType.HUMAN:
                    g_status.person_in_zone_safety += 1
                if blob.type == BlobType.OBJECT:
                    g_status.object_in_zone_safety += 1
                blob.in_roi = True
            elif is_in_zone(g_safety_right, g_gate_roi[1][0], blob.left) or is_in_zone(g_safety_right, g_gate_roi[1][0],
                                                                                       blob.right):
                if blob.type == BlobType.HUMAN:
                    g_status.person_in_zone_1 += 1
                if blob.type == BlobType.OBJECT:
                    g_status.object_in_zone_1 += 1
                blob.in_roi = True


def click_and_select_roi(event, x, y, flags, param):
    global g_threshold_height_foot, g_safety_left, g_safety_right, g_gate_roi, g_nb_click
    # Set entire gate ROI
    if event == cv2.EVENT_LBUTTONUP:
        x = int(x)
        y = int(y)
        # Set left top position of gate
        if g_nb_click == 0:
            g_gate_roi = [(x, y)]
        # Set right bottom position of gate
        elif g_nb_click == 1:
            g_gate_roi.append((x, y))

        # Set left safety zone
        elif g_nb_click == 2:
            g_safety_left = x

        # Set right safety zone
        elif g_nb_click == 3:
            g_safety_right = x

        # Set threshold height for distinguish inside and outside object of gate
        elif g_nb_click == 4:
            g_threshold_height_foot = y

        else:
            # Never happen
            pass

        g_nb_click += 1


def save_roi_file(json_file_name: str = "roi_gate.json"):
    """
    Save g_gate_roi, g_safety_left, g_safety_right, g_threshold_height_foot in a json file
    :param json_file_name:
    :return:
    """
    global g_gate_roi, g_safety_left, g_safety_right, g_threshold_height_foot
    data = {"gate_box": g_gate_roi, "safety_left": g_safety_left, "safety_right": g_safety_right,
            "threshold_foot": g_threshold_height_foot}
    with open(json_file_name, "w") as fileName:
        json.dump(data, fileName)


def set_roi(img: np.ndarray):
    global g_nb_click
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_select_roi)
    cv2.imshow("image", img)

    while g_nb_click < 5:
        cv2.waitKey(1)

    save_roi_file()


def print_status_in_image(img: np.ndarray, width: int, height: int) -> None:
    global g_status
    # Display status in image, prepare display part
    zone1_presence_position = (int(width) - 320, 50)
    zone1_person_position = (int(width) - 320, 80)
    zone1_object_position = (int(width) - 320, 110)
    color_presence = None
    if g_status.is_presence_zone_1():
        color_presence = (0, 255, 0)
    else:
        color_presence = (255, 0, 0)
    cv2.putText(img, "Presence zone right", zone1_presence_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color_presence, 3)
    cv2.putText(img, "Person = {}".format(g_status.person_in_zone_1), zone1_person_position,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(img, "Object = {}".format(g_status.object_in_zone_1), zone1_object_position,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    zoneSafety_presence_position = (int(width / 2) - 200, 50)
    zoneSafety_person_position = (int(width / 2) - 200, 80)
    zoneSafety_object_position = (int(width / 2) - 200, 110)
    color_presence = None
    if g_status.is_presence_zone_safety():
        color_presence = (0, 255, 0)
    else:
        color_presence = (255, 0, 0)
    cv2.putText(img, "Presence zone safety", zoneSafety_presence_position, cv2.FONT_HERSHEY_SIMPLEX, 1,
                color_presence, 3)
    cv2.putText(img, "Person = {}".format(g_status.person_in_zone_safety), zoneSafety_person_position,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(img, "Object = {}".format(g_status.object_in_zone_safety), zoneSafety_object_position,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    zone2_presence_position = (0, 50)
    zone2_person_position = (0, 80)
    zone2_object_position = (0, 110)
    color_presence = None
    if g_status.is_presence_zone_2():
        color_presence = (0, 255, 0)
    else:
        color_presence = (255, 0, 0)
    cv2.putText(img, "Presence zone left", zone2_presence_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color_presence, 3)
    cv2.putText(img, "Person = {}".format(g_status.person_in_zone_2), zone2_person_position,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(img, "Object = {}".format(g_status.object_in_zone_2), zone2_object_position,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


def print_blobs_in_image(img, blobs):
    for _, blob in blobs.items():
        color = (0, 0, 0)
        if blob.type == BlobType.HUMAN:
            if blob.in_gate:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
        else:
            color = (0, 255, 255)
        cv2.rectangle(img,
                      (int(blob.left), int(blob.bottom)),
                      (int(blob.right), int(blob.top)),
                      color, 5)
        cv2.putText(img, "{}".format(blob.identification),
                    (int(blob.get_center()[0]), int(blob.get_center()[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 3)


if __name__ == '__main__':
    # net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.40)
    net = jetson.inference.detectNet("ssd-inception-v2", threshold=0.40)
    # camera = jetson.utils.gstCamera(1280, 720, "0")

    camera = acquisition.Camera()
    camera.start()
    width = camera.get_width()
    height = camera.get_height()

    display = jetson.utils.glDisplay()

    # Init tracker
    tracker_object = Tracker(200, [100, 199])
    tracker_human = Tracker(200, [1, 99])

    try:
        # Wait 2 seconds to warmup the camera
        time.sleep(2)

        # Load gate_roi if exist
        if not load_gate_roi():
            frame = camera.read()
            if frame is None:
                raise Exception("No image")
            set_roi(frame)
        '''
        else:
            answer = input("Do you want to set new ROI configuration ? (y/n)")
            if answer == "y":
                print("Set new ROI configuration")
                img = cv2.cvtColor(img_cuda.astype(np.uint8), cv2.COLOR_RGBA2BGR)
                set_roi(img)
            elif answer == "n":
                print("Load previous ROI configuration")
            else:
                print("Exit program : Do not understand your choice")
                exit(0)
        '''

        while display.IsOpen():
            bench_start_loop = time.time()
            # Reset value
            g_status.reset()
            g_list_object.clear()
            g_list_human.clear()
            new_reference_img_condition = False

            bench_start_capture = time.time()
            # Capture image and store in variable for specific use
            # imgFromCamera, width, height = camera.CaptureRGBA(zeroCopy=1)
            frame = camera.read()
            if frame is None:
                continue
            bench_end_capture = time.time()

            img_for_utils = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

            bench_start_convert = time.time()
            img_for_cuda = jetson.utils.cudaFromNumpy(img_for_utils)
            bench_end_convert = time.time()

            # Detection part
            bench_start_detect = time.time()
            detections = net.Detect(img_for_cuda, width, height, 'none')
            bench_end_detect = time.time()

            for detection in detections:
                # ClassID == 1 => human
                if detection.ClassID == 1:
                    blob = Blob()
                    blob.bounding_box = (int(detection.Left), int(detection.Top), int(detection.Right - detection.Left),
                                         int(detection.Bottom - detection.Top))
                    blob.top = int(detection.Top)
                    blob.bottom = int(detection.Bottom)
                    blob.left = int(detection.Left)
                    blob.right = int(detection.Right)

                    blob.type = BlobType.HUMAN
                    blob.in_gate = (detection.Bottom > g_threshold_height_foot)

                    g_list_human.append(blob)

            bench_start_find_object = time.time()
            # img_object_gray = cv2.cvtColor(img_from_cuda.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
            img_object_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # See if it is necessary or just increase threshold is enough
            img_object_gray = cv2.GaussianBlur(img_object_gray, (5, 5), 0)

            if g_img_object_zone is not None:
                img_diff = cv2.absdiff(img_object_gray, g_img_object_zone)
                _, img_threshold = cv2.threshold(img_diff, 60, 255, cv2.THRESH_BINARY)

                # Each 2 minutes
                if time.time() - g_time_reference_image > 120:
                    # Define if we can take actual frame as reference for next iteration, depend on how many pixel diff
                    # Cropped image to get non zero only in gate roi
                    img_diff_gate_roi = img_threshold[g_gate_roi[0][1]:g_gate_roi[1][1],
                                        g_gate_roi[0][0]:g_gate_roi[1][0]]
                    count_diff = cv2.countNonZero(img_diff_gate_roi)
                    img_diff_gate_roi_size = len(img_diff_gate_roi) * len(img_diff_gate_roi[0])
                    ratio_diff_in_gate_roi = count_diff * 100 / img_diff_gate_roi_size
                    new_reference_img_condition = (ratio_diff_in_gate_roi < 0.1)
                    g_time_reference_image = time.time()
                    cv2.putText(frame, "{}".format(count_diff), (int(width / 2), int(height - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255), 3)
                    cv2.putText(frame, "{0:.2f}%".format(ratio_diff_in_gate_roi), (int(width / 2), int(height - 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                # Use find contour to get bounding box of blob.py not in reference image
                for blob in g_list_human:
                    # Enlarge the area to erase entirely human
                    x_min = blob.left - g_marge_enlarge_erase_zone
                    if x_min < 0:
                        x_min = 0
                    x_max = blob.right + g_marge_enlarge_erase_zone
                    if x_max > width:
                        x_max = width
                    y_min = blob.top - g_marge_enlarge_erase_zone
                    if y_min < 0:
                        y_min = 0
                    y_max = blob.bottom + g_marge_enlarge_erase_zone
                    if y_max > height:
                        y_max = height
                    img_threshold[y_min:y_max, x_min:x_max] = 0

                img_for_contour_gate_roi = np.zeros((height, width), dtype=np.uint8)
                img_for_contour_gate_roi[g_gate_roi[0][1]:g_gate_roi[1][1], g_gate_roi[0][0]:g_gate_roi[1][0]] = \
                    img_threshold[g_gate_roi[0][1]:g_gate_roi[1][1], g_gate_roi[0][0]:g_gate_roi[1][0]]

                # Dilate to make one blob when blobs are closed
                structElement5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                img_for_contour_gate_roi = cv2.erode(img_for_contour_gate_roi, structElement5x5)
                # TODO: Analyse if good practice
                for i in range(0, 2):
                    img_for_contour_gate_roi = cv2.dilate(img_for_contour_gate_roi, structElement5x5)
                    img_for_contour_gate_roi = cv2.dilate(img_for_contour_gate_roi, structElement5x5)

                contours, _ = cv2.findContours(img_for_contour_gate_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for i in range(0, len(contours)):
                    (x, y, w, h) = cv2.boundingRect(contours[i])
                    if w < 20 or h < 20:
                        print("Object ignored : w = {}; h = {}".format(w, h))
                        continue
                    if y + h < g_threshold_height_foot:
                        print("Object ignored : do not touch ground")
                        continue
                    blob_object = Blob()
                    blob_object.type = BlobType.OBJECT
                    blob_object.bounding_box = (int(x), int(y), int(w), int(h))
                    blob_object.right = int(x + w)
                    blob_object.left = int(x)
                    blob_object.top = int(y)
                    blob_object.bottom = int(y + h)
                    blob_object.in_gate = True

                    g_list_object.append(blob_object)
            bench_end_find_object = time.time()

            if g_img_object_zone is None or new_reference_img_condition:
                g_img_object_zone = img_object_gray.copy()
                g_time_reference_image = time.time()
                print("A new reference image taken !!!!!")

            bench_start_determine_presence = time.time()
            # determine presence in each zone
            determine_presence_zone(g_list_object)
            determine_presence_zone(g_list_human)
            bench_end_determine_presence = time.time()

            # Add tracking
            # TODO: REFACTO UGLY
            list_human_roi = []
            for human in g_list_human:
                #if human.in_gate and human.in_roi:
                list_human_roi.append(human)

            bench_start_tracking = time.time()
            tracker_object.update(g_list_object)
            list_object = tracker_object.elements
            tracker_human.update(list_human_roi)
            list_human = tracker_human.elements
            bench_end_tracking = time.time()

            # display blob in img
            bench_start_print = time.time()
            print_blobs_in_image(frame, list_object)
            print_blobs_in_image(frame, list_human)
            # display status in img
            print_status_in_image(frame, width, height)

            # Prepare img for object detection
            # Draw all line in img
            cv2.rectangle(frame, (g_gate_roi[0][0], g_gate_roi[0][1]), (g_gate_roi[1][0], g_gate_roi[1][1]),
                          (255, 0, 0), 3)
            cv2.line(frame, (g_safety_left, 0), (g_safety_left, height), (0, 255, 0), 3)
            cv2.line(frame, (g_safety_right, 0), (g_safety_right, height), (0, 255, 0), 3)
            cv2.line(frame, (0, g_threshold_height_foot), (width, g_threshold_height_foot), (0, 0, 255), 3)
            bench_end_print = time.time()

            bench_start_render = time.time()
            img_for_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            imgToDisplay = jetson.utils.cudaFromNumpy(img_for_display)
            display.RenderOnce(imgToDisplay, width, height)
            display.SetTitle("Object Detection")
            bench_end_render = time.time()

            bench_end_loop = time.time()

            # Display all timer
            print("############## Bench Time ##############")
            print("Entire Loop = {}".format(bench_end_loop - bench_start_loop))
            print("## INIT")
            print("Capture = {}".format(bench_end_capture - bench_start_capture))
            print("Convert = {}".format(bench_end_convert - bench_start_convert))
            print("## Detection & Tracking")
            print("Detect = {}".format(bench_end_detect - bench_start_detect))
            print("Find object = {}".format(bench_end_find_object - bench_start_find_object))
            print("Tracking = {}".format(bench_end_tracking - bench_start_tracking))
            print("Determine Presence = {}".format(bench_end_determine_presence - bench_start_determine_presence))
            print("## Visualisation")
            print("Print in image = {}".format(bench_end_print - bench_start_print))
            print("Render = {}".format(bench_end_render - bench_start_render))
            print("############## END Bench Time ##############")

        cv2.destroyAllWindows()

    except Exception as e:
        print(str(e))

    finally:
        # Clean resources
        camera.stop()

