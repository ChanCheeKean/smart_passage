'''
@author : Leo JI
@brief : Help to configure all ROI. Take ROI of Gate and safety zone and threshold of height

1 click : left top corner of gate
2 click : right bottom corner of gate
3 click : left safety
4 click : right safety
5 click : threshold height for foot
'''

import cv2
import Gstream
import json

'''
Coordonnée haut gauche et bas droit du rectangle ROI de la gate
'''
g_gate_roi = []

'''
ligne vertical pour la zone de safety gauche
'''
g_safety_left = None

'''
ligne vertical pour la zone de safety droite
'''
g_safety_right = None

'''
ligne horizontal pour la hauteur des pieds
'''
g_threshold_height_foot = None
''' Define line which is not horizontal '''
g_threshold_height_foot_line = []

'''
Use for get all ROI, depend how many click, fill the right variable
'''
g_nb_click = 0


def click_and_select_roi(event, x, y, flags, param):
    global g_threshold_height_foot, g_safety_left, g_safety_right, g_gate_roi, g_nb_click
    # Set entire gate ROI

    if event == cv2.EVENT_LBUTTONUP:
        x = int(x)
        y = int(y)
        #Set left top position of gate
        if g_nb_click == 0:
            g_gate_roi = [(x, y)]
        #Set right bottom position of gate
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

        elif g_nb_click == 5 or g_nb_click == 6:
            g_threshold_height_foot_line.append((x, y))

        else:
            # Never happen
            pass

        g_nb_click += 1


def set_gate_ROI(img) -> None:
    '''
    Set gate ROI manually
    '''

    global g_nb_click
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_select_roi)
    cv2.imshow("image", img)

    while g_nb_click < 7:
        cv2.waitKey(1)


def save_roi_file(jsonFileName: str = "roi_gate.json"):
    global g_gate_roi, g_safety_left, g_safety_right, g_threshold_height_foot
    data = {}
    data["gate_box"] = g_gate_roi
    data["safety_left"] = g_safety_left
    data["safety_right"] = g_safety_right
    data["threshold_foot"] = g_threshold_height_foot
    data["threshold_foot_2_points"] = g_threshold_height_foot_line
    with open(jsonFileName, "w") as fileName:
        json.dump(data, fileName)


if __name__ == '__main__':
    cap = cv2.VideoCapture(Gstream.gstreamer_pipeline(),cv2.CAP_GSTREAMER)
    ret, frame = cap.read()
    set_gate_ROI(frame)

    print("gate ROI coord = {}".format(g_gate_roi))
    print("safety left = {}".format(g_safety_left))
    print("safety right = {}".format(g_safety_right))
    print("threshold foot = {}".format(g_threshold_height_foot))

    cv2.rectangle(frame, (g_gate_roi[0][0], g_gate_roi[0][1]), (g_gate_roi[1][0], g_gate_roi[1][1]), (255, 0, 0), 3)
    cv2.line(frame, (g_safety_left, 0), (g_safety_left, 700), (0, 255, 0), 3)
    cv2.line(frame, (g_safety_right, 0), (g_safety_right, 700), (0, 255, 0), 3)
    #cv2.line(frame, (0, g_threshold_height_foot), (1200, g_threshold_height_foot), (0, 0, 255), 3)
    cv2.line(frame, (g_threshold_height_foot_line[0][0], g_threshold_height_foot_line[0][1]),
             (g_threshold_height_foot_line[1][0], g_threshold_height_foot_line[1][1]),
             (0, 0, 255), 3)

    cv2.imshow("All ROI set", frame)

    cv2.waitKey(0)

    save_roi_file()

    cv2.destroyAllWindows()
