import cv2
import jetson.inference
import jetson.utils
import time
import numpy as np

def render(img, gate_xyxy, trigger_distance, bench_start_loop):
    '''Draw the zone border in the image'''
    cv2.rectangle(
        img,
        (gate_xyxy['left'], gate_xyxy['top']),
        (gate_xyxy['right'], gate_xyxy['bottom']),
        (255, 0, 0), 
        2
    )

    cv2.rectangle(
        img,
        (gate_xyxy['left-safety'], gate_xyxy['top']),
        (gate_xyxy['right-safety'], gate_xyxy['bottom']),
        (255, 0, 255), 
        2
    )
    
    # for gate gantry
    cv2.rectangle(
        img,
        (gate_xyxy['blocked_left_1'], gate_xyxy['blocked_top']),
        (gate_xyxy['blocked_left_2'], gate_xyxy['bottom']),
        (255, 255, 255), 
        2
    )

    # for gate gantry
    cv2.rectangle(
        img,
        (gate_xyxy['blocked_right_1'], gate_xyxy['blocked_top']),
        (gate_xyxy['blocked_right_2'], gate_xyxy['bottom']),
        (255, 255, 255), 
        2
    )


   #  gate_x_mean = (gate_xyxy['right'] + gate_xyxy['left']) // 2
    # cv2.line(
        # img,
        # (gate_x_mean - trigger_distance//2, gate_xyxy['top']),
        #(gate_x_mean + trigger_distance//2, gate_xyxy['top']),
        # (0, 0, 255), 
        # 2
    # )
    
def is_inside_gate(detection, gate_xyxy, in_safety=True):
    '''Qualifier Criteria: Check if object is in the region of Interest
    Special limit if the person stand in the paid or unpaid zone as the bonding box is blocked by the gate
    '''
        
    center = (detection[0] + detection[2]) / 2
    top = gate_xyxy['top']
    in_safe = (center < gate_xyxy['right-safety']) and (center > gate_xyxy['left-safety']) 
    
    if ((center > gate_xyxy['blocked_right_1']) and (center < gate_xyxy['blocked_right_2'])) | ((center > gate_xyxy['blocked_left_1']) and (center < gate_xyxy['blocked_left_2'])):
        top = gate_xyxy['blocked_top']

    in_roi = (center < gate_xyxy['right']) and (center > gate_xyxy['left']) and (detection[3] > top) and (detection[3] < gate_xyxy['bottom'])

    if in_safety:
        return in_roi and in_safe
    else:
        return in_roi

def determine_zone(center, gate_xyxy):
    if center < gate_xyxy['left-safety']:
        return 'left'
    elif center > gate_xyxy['right-safety']:
        return 'right'
    else:
        return 'safety'

def determine_human_presence(frame, detections, gate_xyxy, object_count, check_dir=False, paid_pos='left', id_store=[]):
    '''draw boxes of detected human and check if the direction is correct'''

    antidir_flag = False
    for det in detections:
        if (is_inside_gate(det, gate_xyxy, in_safety=False)):
            center = (det[0] + det[2]) / 2
            # human_id = det[-1]
            human_id = 0
            zone = determine_zone(center, gate_xyxy)
            object_count.append({'type': 'human', 'id': human_id, 'zone' : zone})

            # to check entry direction
            color = (0, 255, 0)
            if check_dir:
                if paid_pos == 'left':
                    cond = center < gate_xyxy['center']
                else:
                    cond = center > gate_xyxy['center']
                if cond:
                    if human_id not in id_store:
                        id_store.append(human_id)
                elif (zone == 'safety'):
                    # if id in list then remove it then proceed, else raise alarm
                    if human_id not in id_store:
                        color = (0, 0, 255)
                        antidir_flag = True

            cv2.rectangle(
                frame, 
                (int(det[0]), int(det[1])),
                (int(det[2]), int(det[3])),
                color, 
                2)
            
            # cv2.putText(
                # frame, f'ID: {det[-1]}', (int(det[0]), int(det[1] + 50)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            
    return object_count, antidir_flag, id_store

def NMS(boxes, overlapThresh=0.4):
    '''Non Max Suppression to remove overlapped boxes'''

    # Return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We add 1, because the pixel at the start as well as at the end counts
    # The indices of all boxes at start. We will redundant indices one by one.
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):
        # Create temporary indices  
        temp_indices = indices[indices!=i]
        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])
        # Find out the width and the height of the intersection box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index  
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]
    return indices

def determine_object_presence(
    frame, ref_img, detections, gate_xyxy, object_count, bg_sub, obj_tracker, mask_human=True, show_contour_area=False):
    
    _ = bg_sub.apply(ref_img)
    bgMask = bg_sub.apply(frame)
    oversize_flag = False
    
    if mask_human: 
        # human_det = [det for det in detections if det[5] == 1]
        reduce_size = 100
        for det in detections:  
            x1 = det[0]
            x2 = det[2] 
            # if (x2 - x1) >= (reduce_size * 2):       
                # x1 = x1 + reduce_size
                # x2 = x2 - reduce_size  
            bgMask[int(det[1]) : int(det[3]), int(det[0]) : int(det[2])] = 0
    
    # structElement = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    # for _ in range(1):
         # bgMask = cv2.erode(bgMask, structElement)

    structElement = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    for _ in range(5):
         bgMask = cv2.dilate(bgMask, structElement)

    # gMask = cv2.GaussianBlur(bgMask, (5, 5), 0)
    bgMask = cv2.Canny(bgMask, 30, 90)
    contours, _ = cv2.findContours(bgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i in range(0, len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[i])
        size = cv2.contourArea(contours[i])
        area = (w * h) / 100
        if (w < 50) | (h < 50) | ((y + h) < gate_xyxy['top']) :
            continue
        zone = determine_zone(x + w / 2, gate_xyxy)
        object_count.append(
            {'type': 'object', 'zone': zone, 'bonding_box': [x, y, x+w, y+h], 'area': area, 'contourArea': size}
        )

    # use nms to remove overlap object
    # if len(object_count) > 0:
        # boxes = np.array([ob['bonding_box'] for ob in object_count])
        # ind = NMS(boxes, overlapThresh=0.3)
        # object_count = [object_count[i] for i in ind]

    # for ob in object_count:
        # (x, y, x2, y2) = ob['bonding_box']
        # area, c_size = ob['area'], ob['contourArea']
        # color = (255, 255, 0)
        # cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), color, 2)
        # cv2.putText(frame, f'Size: {area}', (int(x), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        
    '''
    # TODO: to show oversize object    
    obj_lis = []    
    for ob in object_count:
        (x, y, x2, y2) = ob['bonding_box']
        area, c_size = ob['area'], ob['contourArea']
        if (area > 400):
            obj_lis.append([x, y, x2, y2, 0.8])
    if len(obj_lis) > 0:
        output = obj_tracker.update(np.array(obj_lis))
        # oversize_flag = True
    else:
        output = obj_tracker.update(np.empty((0, 5)))
    
    # display box and size
    for out in output:
        x, y, x2, y2 = out[:4]
        color = (255, 0, 255)
        cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f'Size: {area}', (int(x), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
        # color = (255, 0, 0)
        # if (area > 250) and (c_size > 10) and (x < gate_xyxy['right'] - 60) and (x2 > gate_xyxy['left'] + 60) and ((y > 320) | (x2 < 630)) and (x > 10):
            # # to avoid undetected human before entry mistaken as object
            # # if ((y > 320) | (x2 < 630)) and (x > 10):
            # color = (255, 0, 255)
            # # oversize_flag = True
    '''
         
    return bgMask, object_count, oversize_flag

def tailgating_detection(img, detections, gate_xyxy, trigger_distance):
    human_detection_in_gate = []
    for det in detections:
        if (is_inside_gate(det, gate_xyxy)):
            human_detection_in_gate.append(det)

    for i, human_i in enumerate(human_detection_in_gate):
        for _, human_j in enumerate(human_detection_in_gate[:i] + human_detection_in_gate[(i + 1):]):
            if human_i[0] < human_j[0]: # human i is on the left
                if human_j[0] - human_i[2] < trigger_distance:
                    # red box for tailgaters
                    cv2.rectangle(
                        img,
                        (int(human_i[0]), int(human_i[1])),
                        (int(human_i[2]), int(human_i[3])),
                        (0, 0, 255), 
                        3
                    )
                    cv2.rectangle(
                        img,
                        (int(human_j[0]), int(human_j[1])),
                        (int(human_j[2]), int(human_j[3])),
                        (0, 0, 255), 
                        3
                    )
                    return True
    return False

def update_object_dict(detect_dict, warning_flag, oversize_flag, antidir_flag, object_count_list):
    '''update object count in each zone'''

    for _object in object_count_list:
        if _object['type'] == 'human':

            detect_dict['human_count'] += 1
            detect_dict[f'human_{_object["zone"]}'] += 1

        elif _object['type'] == 'object':
            detect_dict[f'object_{_object["zone"]}'] = 1

    detect_dict['oversize_flag'] = 1 if oversize_flag else -1
    detect_dict['antidir_flag'] = 1 if antidir_flag else -1

    if warning_flag:
        detect_dict['tailgate_flag'] = 1 
    elif detect_dict['human_count'] > 0:
        detect_dict['tailgate_flag'] = 0
    else: 
        detect_dict['tailgate_flag'] = -1
    return detect_dict

def track_id(frame, detections, deep_tracker):
    # detections = [{
    #     'Area': 95097.0546875, 
    #     'Bottom': 479.4737548828125, 
    #     'Center': (104.98439025878906, 244.78831481933594), 
    #     'ClassID': 1, 'Confidence': 0.7254542112350464, 
    #     'Height': 469.37091064453125, 
    #     'Left': 3.6817169189453125, 
    #     'ROI': (3.6817169189453125, 10.102858543395996, 206.2870635986328, 479.4737548828125), 
    #     'Right': 206.2870635986328, 
    #     'Top': 10.102858543395996, 
    #     'Width': 202.6053466796875
    #     }]

    processed_det = []
    for det in detections:
        processed_det.append([
            det.Left,
            det.Top,
            det.Right,
            det.Bottom,
            det.Confidence,
            # det.ClassID,
            # det.Center,
            # det.Width,
            # det.Height,
            # det.Area,
            # det.ROI,
        ])
    
    if len(processed_det) > 0:
        processed_det = np.array(processed_det)
        output = deep_tracker.update(processed_det)
    else:
        output = deep_tracker.update(np.empty((0, 5)))
        # output = processed_det
    return processed_det, output

def process_frame(
    frame, detect_dict, net, gate_xyxy, trigger_distance, deep_tracker, id_store_list, ref_img, bg_sub, obj_tracker, verbose=0):
    # print(frame.shape) # 720h, 1280w

    # record count
    object_count_list = []
    try:
        bench_start_loop = time.time()
        tailgate_flag, oversize_flag, antidir_flag = (False, False, False)

        # Convert numpy image to cuda
        height, width, _ = frame.shape
        img_for_cuda = jetson.utils.cudaFromNumpy(frame)

        # Object detection
        detections = net.Detect(img_for_cuda, width, height, 'none')
        detections = [det for det in detections if det.ClassID == 1]

        # deepsort tracking
        detections, detections_id = track_id(frame, detections, deep_tracker)
        # print(detections)

        # detect non human object
        bgMask, object_count_list, oversize_flag = determine_object_presence(
            frame, ref_img, detections, gate_xyxy, object_count_list, bg_sub, obj_tracker, mask_human=True)

        # Render detection
        # render(frame, gate_xyxy, trigger_distance, bench_start_loop)
        
        # zone count
        object_count_list, antidir_flag, id_store_list = determine_human_presence(frame, detections_id, gate_xyxy, object_count_list, False, id_store_list)

        # Tailgating logic
        tailgate_flag = tailgating_detection(frame, detections, gate_xyxy, trigger_distance)

        # update the localization dict
        detect_dict = update_object_dict(detect_dict, tailgate_flag, oversize_flag, antidir_flag, object_count_list)
        
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
            print("############ END Bench Time ############")
        return frame, detect_dict, id_store_list
    
    except Exception as e:
        print('Error:', e)
        return frame, detect_dict, id_store_list

