import jetson.inference
import jetson.utils
import time
import cv2
import numpy as np


def render(img, detections, gate_xyxy, trigger_distance, bench_start_loop):
    
    for detection in detections:
        if (detection.ClassID == 1):
            if (is_inside_gate(detection, gate_xyxy)):
                cv2.rectangle( 	
                    img,
                    (int(detection.Left), int(detection.Top)),
                    (int(detection.Right), int(detection.Bottom)),
                    (0, 255, 0), 
                    2
                )
        # elif (detection.ClassID != 5) and (detection.ClassID != 1):
            # cv2.rectangle( 	
                # img,
                # (int(detection.Left), int(detection.Top)),
                # (int(detection.Right), int(detection.Bottom)),
                # (255, 255, 0), 
                # 2
            # )
   

    # cv2.rectangle(
        # img,
        # (gate_xyxy['left'], gate_xyxy['top']),
        # (gate_xyxy['right'], gate_xyxy['bottom']),
        # (255, 0, 0), 
        # 2
    # )

    # gate_x_mean = (gate_xyxy['right'] + gate_xyxy['left']) // 2
    # cv2.line(
    #     img,
    #     (gate_x_mean - trigger_distance//2, gate_xyxy['top']),
    #     (gate_x_mean + trigger_distance//2, gate_xyxy['top']),
    #     (0, 0, 255), 
    #     2
    # )

    # cv2.putText(
    #      img,
    #      f'FPS = {int(1/(time.time()-bench_start_loop))}',
    #      (5, 30),
    #      cv2.FONT_HERSHEY_SIMPLEX,
    #      1,
    #      (0, 255, 0),
    #      2
    #  )



def is_inside_gate(detection, gate_xyxy):
    return (detection.Left < gate_xyxy['right']) and (detection.Right > gate_xyxy['left']) and (detection.Bottom > gate_xyxy['top']) and (detection.Top < gate_xyxy['bottom'])

def determine_presence_zone(detections, gate_xyxy, detect_dict):

    for detection in detections:
        if (detection.ClassID == 1) and (is_inside_gate(detection, gate_xyxy)):
            center = (detection.Left + detection.Right) // 2
            detect_dict['human_count'] += 1

            if center < gate_xyxy['left']:
                detect_dict['human_left'] += 1
            elif (center > gate_xyxy['left']) and (center < gate_xyxy['right']):
                detect_dict['human_safety'] += 1
            elif center > gate_xyxy['right']:
                detect_dict['human_right'] += 1


def tailgating_detection(detections, gate_xyxy, trigger_distance, img):
    human_detection_in_gate = []
    for detection in detections:
        if detection.ClassID == 1:
            if (is_inside_gate(detection, gate_xyxy)):
              human_detection_in_gate.append(detection)

    for i, human_i in enumerate(human_detection_in_gate):
        for _, human_j in enumerate(human_detection_in_gate[:i]+human_detection_in_gate[(i+1):]):
            if human_i.Left < human_j.Left: # human i is on the left
                if human_j.Left - human_i.Right < trigger_distance:
                    # red box for tailgaters
                    cv2.rectangle(
                        img,
                        (int(human_i.Left), int(human_i.Top)),
                        (int(human_i.Right), int(human_i.Bottom)),
                        (0, 0, 255), 
                        3
                    )
                    cv2.rectangle(
                        img,
                        (int(human_j.Left), int(human_j.Top)),
                        (int(human_j.Right), int(human_j.Bottom)),
                        (0, 0, 255), 
                        3
                    )
                    return True
                
    return False

def draw_object(detections, ori_frame, reference_image, gate_xyxy, detect_dict):

    
    frame = ori_frame.copy()
    height, width, _ = frame.shape
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    # reference_image = cv2.GaussianBlur(reference_image, (5, 5), 0)

    img_object_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # img_object_gray = cv2.GaussianBlur(img_object_gray, (5, 5), 0)
    img_diff = cv2.absdiff(img_object_gray, reference_image, )

    _, img_threshold = cv2.threshold(img_diff, 100, 255, cv2.THRESH_BINARY)
    human_det = [det for det in detections if det.ClassID == 1]

    g_marge_enlarge_erase_zone = 20
    for det in human_det:
        # Enlarge the area to erase entirely human
        x_min = max(0, int(det.Left - g_marge_enlarge_erase_zone))
        x_max = min(width, int(det.Right + g_marge_enlarge_erase_zone))
        y_min = max(0, int(det.Top - g_marge_enlarge_erase_zone))
        y_max = min(height, int(det.Bottom + g_marge_enlarge_erase_zone))
        img_threshold[y_min:y_max, x_min:x_max] = 0

    structElement5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_for_contour = img_threshold

    for i in range(2):
        img_for_contour = cv2.erode(img_for_contour, structElement5x5)

    for i in range(6):
         img_for_contour = cv2.dilate(img_for_contour, structElement5x5)
    
    contours, _ = cv2.findContours(img_for_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        if cv2.contourArea(contours[i]) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contours[i])
        if (x < gate_xyxy['right'] + 50) and ((x+w) > gate_xyxy['left'] - 30) and ((y+h) > gate_xyxy['top']) and (y < gate_xyxy['bottom']):
            cv2.rectangle(ori_frame, (int(x), int(y)), (int(x+w), int(y+h)), (255, 255, 0), 2)

            center = x + w / 2
            if center < gate_xyxy['left']:
                detect_dict['object_left'] += 1
            elif (center > gate_xyxy['left']) and (center < gate_xyxy['right']):
                detect_dict['object_safety'] += 1
            elif center > gate_xyxy['right']:
                detect_dict['object_right'] += 1


def process_frame(frame, net, gate_xyxy, ref_img, verbose=0):
    trigger_distance = 150
    # record count
    detect_dict = {
        'human_count' : 0, 
        'human_left' : 0,
        'human_safety': 0,
        'human_right': 0,
        'object_left' : 0,
        'object_safety': 0,
        'object_right': 0,
    }

    try:
        bench_start_loop = time.time()
        
        # Convert numpy image to cuda
        height, width, _ = frame.shape
        img_for_cuda = jetson.utils.cudaFromNumpy(frame)

        # Object detection
        detections = net.Detect(img_for_cuda, width, height, 'none')

        # detect non human object
        # draw_object(detections, frame, ref_img, gate_xyxy, detect_dict)

        # Render detection
        render(frame, detections, gate_xyxy, trigger_distance, bench_start_loop)
        
        # zone count
        determine_presence_zone(detections, gate_xyxy, detect_dict)

        # Tailgating logic
        warning_flag = tailgating_detection(detections, gate_xyxy, trigger_distance, frame)
        
        # Output
        # cv2.imshow('output', frame)
        # c = cv2.waitKey(1)
        # if c == ord('q'):
        #     break
        # elif c == ord('6'):
        #     gate_xyxy['left'] += 5
        # elif c == ord('4'):
        #     gate_xyxy['left'] -= 5
        # elif c == ord('3'):
        #     gate_xyxy['right'] += 5
        # elif c == ord('1'):
        #     gate_xyxy['right'] -= 5
        # elif c == ord('2'):
        #     gate_xyxy['top'] += 5
        # elif c == ord('5'):
        #     gate_xyxy['top'] -= 5
        # elif c == ord('9'):
        #     trigger_distance += 5
        # elif c == ord('7'):
        #     trigger_distance -= 5


        # Terminal display
        if verbose > 0:
            print("############## Bench Time ##############")
            print(f"Entire Loop = {(time.time() - bench_start_loop):.3f} ms")
            print(f"Tailgating detected : {warning_flag}")
            print("############ END Bench Time ############")

        if (warning_flag):
            warning_flag = 1
        elif detect_dict['human_count'] > 0:
            warning_flag = 0
        else:
            warning_flag = -1
        return frame, warning_flag, detect_dict
    
    except Exception as e:
        print(e)
        return frame, -1, detect_dict

