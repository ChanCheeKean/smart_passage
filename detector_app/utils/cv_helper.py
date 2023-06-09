import cv2
import torch
import time
from pathlib import Path
from utils.config_loader import config
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.utils.plotting import Annotator, colors
# from yolov5.models.common import DetectMultiBackend
# from yolov5.utils.general import non_max_suppression, scale_boxes
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.utils.ops import non_max_suppression, scale_boxes
from trackers.multi_tracker_zoo import create_tracker

### gate roi ###
gate_xyxy = config['gate_roi']

def determine_zone(det):
    '''check the zone of the passanger or object according to the config roi'''
    center = (det[0] + det[2]) / 2
    if center < gate_xyxy['left-safety']:
        return 'left'
    elif center > gate_xyxy['right-safety']:
        return 'right'
    else:
        return 'safety'
    
def check_roi(box, in_safety=False):
    '''Check if object is in the region of Interest and also within size limit
    in_safety: to check if the object is within the safety zone
    '''
    center = (box[0] + box[2]) / 2
    width = box[2] - box[0]
    height = box[3] - box[1]

    in_safe = (center < gate_xyxy['right-safety']) and (center > gate_xyxy['left-safety']) 
    in_roi = (center < gate_xyxy['right']) and (center > gate_xyxy['left']) and (box[1] < gate_xyxy['top']) and (box[3] > gate_xyxy['bottom'])

    # to remove overly big object
    in_size = ((width * height) < 100000) | (box[5] == 0)

    if in_safety:
        return in_roi and in_safe and in_size
    else:
        return in_roi and in_size

def plot_gate_roi(img):
    '''Draw zone border in image'''
    # gate roi
    cv2.rectangle(
        img,
        (gate_xyxy['left'], gate_xyxy['top']),
        (gate_xyxy['right'], gate_xyxy['bottom']),
        (255, 0, 0), 
        2
    )

    # safety zone
    cv2.rectangle(
        img,
        (gate_xyxy['left-safety'], gate_xyxy['top']),
        (gate_xyxy['right-safety'], gate_xyxy['bottom']),
        (255, 0, 255), 
        2
    )

    # test
    # top, left = 280, 140
    # width, height = 850, 350
    # cv2.rectangle(
    #     img,
    #     (left, top),
    #     (left + width, top + height),
    #     (255, 0, 255), 
    #     2
    # )

### tailgate detection ###
def tailgating_detection(detections, trigger_distance):
    '''return tailgate flag if >1 passengers in safety zone, depends on trigger distance'''
    human_detection_in_gate = []
    detections = [det for det in detections if det[5].item() == 0]
    lis = []

    for det in detections:
        if (check_roi(det, in_safety=True)):
            human_detection_in_gate.append(det)

    for i, human_i in enumerate(human_detection_in_gate):
        for _, human_j in enumerate(human_detection_in_gate[:i] + human_detection_in_gate[(i + 1):]):
            if human_i[0] < human_j[0]: # human i is on the left
                if human_j[0] - human_i[2] < trigger_distance:
                    lis = [human_j[5].item(), human_i[5].item()]
                    return 1, lis
    return 0, lis

### anti direction ###
def detect_dir(detections, id_paid, id_complete, paid_zone='right'):
    '''return anti direction flag and update recorded passenger id'''
    anti_flag = 0
    lis = []
    detections = [det for det in detections if det[5].item() == 0]
    for det in detections:
        zone = determine_zone(det)
        obj_id = det[6].item()
        if zone == paid_zone:
            if obj_id not in id_paid:
                id_paid.append(obj_id)
        elif (zone != paid_zone) & (zone != 'safety'):
            if (obj_id in id_paid) and (obj_id not in id_complete):
                id_complete.append(obj_id)
            elif obj_id not in id_paid:
                anti_flag = 1
                lis.append(obj_id)

    return anti_flag, id_paid, id_complete, lis

### loitering ###
def detect_loiter(detections, id_stay, stay_limit=5):
    loiter_flag = 0
    detections = [det for det in detections if det[5].item() == 0]
    lis = []

    for det in detections:
        obj_id = det[6].item()
        if obj_id not in id_stay:
            id_stay[obj_id] = int(time.time())
        else:
            stay_time = int(time.time()) - id_stay.get(obj_id, 0)
            if stay_time > stay_limit:
                loiter_flag = 1
                lis.append(obj_id)
    
    return loiter_flag, lis

### update metadata ###
def update_zone_info(dets):
    '''determine zone and combine the info to be displayed in dashboard'''
    info_dict = {
        'human_gate_count' : 0, 
        'human_left' : 0,
        'human_safety': 0,
        'human_right': 0,
        'object_left' : 0,
        'object_safety': 0,
        'object_right': 0,
        'tailgate_flag': 0,
        'antidir_flag': 0,
        'loiter_flag': 0,
        'tailgate_list': [],
        'anti_list': [],
        'loiter_list': [],
        'passenger_count' : 0, 
    }

    for det in dets:
        obj_cls = det[5].item()
        zone = determine_zone(det)
        if obj_cls == 0:
            info_dict['human_gate_count'] += 1
            info_dict[f'human_{zone}'] += 1
        else:
            info_dict[f'object_{zone}'] = 1
    return info_dict

### non max suppression ###
def box_iou(boxes1, boxes2):
    '''return the iou area of 2 boxes'''
    def _upcast(t):
        '''protects from numerical overflows in multiplications by upcasting to the equivalent higher type'''
        if t.is_floating_point():
            return t if t.dtype in (torch.float32, torch.float64) else t.float()
        else:
            return t if t.dtype in (torch.int32, torch.int64) else t.int()
    def box_area(boxes):
        '''computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates'''
        boxes = _upcast(boxes)
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union

### plotting function ###
def plot_image(img, detections, font_size, labels, mm_per_pixel, flag=False, plot_cat=False):
    '''plot detection boxes in the image'''
    annotator = Annotator(img, line_width=font_size, example=labels)
    det_res = [det for det in detections if det[5].item() not in [1, 2]]
    det_res_cat = [det for det in detections if det[5].item() in [1, 2]]

    for _, res in enumerate(det_res):
        bbox = [round(i, 2) for i in res[:4].tolist()]
        conf = round(res[4].item(), 3)
        obj_cls = res[5].item()

        object_id = res[6].item() if obj_cls == 0 \
            else f"H:{int((res[3] - res[1]) * mm_per_pixel)}mm W:{int((res[2] - res[0]) * mm_per_pixel)}mm"

        label_text =  labels[int(obj_cls)]
        if (flag) and (obj_cls == 0):
            color = (0, 0, 100)
        elif obj_cls == 0:
            color = (0, 100, 0) 
        else:
            color = (100, 100, 100)
        annotator.box_label(bbox, f'{object_id} {label_text} {conf}', color=color)

    # to plot eldery and kid
    if plot_cat:
        for _, res in enumerate(det_res_cat):
            bbox = [round(i, 2) for i in res[:4].tolist()]
            conf = round(res[4].item(), 3)
            obj_cls = res[5].item()
            object_id = res[6].item() 
            label_text =  labels[int(obj_cls)]
            color = (133, 3, 109)
            annotator.box_label(bbox, f'{object_id} {label_text} {conf}', color=color)

### object tracker ###
class DeepSortTracker():
    def __init__(self):
        # init tracker and warmup
        tracker_config = config['object_tracker']
        tracking_method = tracker_config['tracking_method']
        tracking_config = tracker_config['tracking_config']
        reid_weights = Path(tracker_config['reid_weights'])
        self.device = select_device(config['model']['device'])
        self.deep_tracker = create_tracker(tracking_method, tracking_config, reid_weights, self.device, False)

        if hasattr(self.deep_tracker, 'model'):
            if hasattr(self.deep_tracker.model, 'warmup'):
                self.deep_tracker.model.warmup()

    def update(self, image, det):
        with torch.no_grad():
            outputs = torch.Tensor(self.deep_tracker.update(det.cpu(), image))
            outputs[:, [-3, -1]] = outputs[:, [-1, -3]]
        return outputs

### OWL-Vit Model ###
class VitModelLoader():
    def __init__(self):
        '''
        OWL-Vit model to detect and plot the image'''
        model_config = config['model']
        self.img_sz = config['video']['img_sz']
        self.conf_thres = model_config['conf_thres']
        self.iou_thres = model_config['iou_thres']
        self.model_name = model_config['model_name']
        self.labels = model_config['classes']
        self.device = select_device(model_config['device'])
        
        # load owl-vit model and queries
        self.processor = OwlViTProcessor.from_pretrained(self.model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(self.model_name).to(self.device)
        self.query = ['photo of a ' + str(label) for label in self.labels]

    def detect(self, image):
        '''torch.Tensor([[xt, yt, xb, yb, conf, cls], [xt, yt, xb, yb, conf, cls], ...])'''
        # pre-processing and inference
        inputs = self.processor(text=[self.query], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.Tensor([image.shape[0:2]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self.conf_thres)[0]
        results = torch.cat(
            tensors=(results["boxes"], results["scores"].view(-1, 1), results["labels"].view(-1, 1)),
            dim=1)
            
        # only include tensor in roi, >confidence level
        lis = []
        for ind, res in enumerate(results):
            obj_cls = res[5].item()
            threshold = self.conf_thres + 0.1 if obj_cls == 0 else self.conf_thres
            conf = round(res[4].item(), 3)
            if (check_roi(res, in_safety=False)) and (conf > threshold):
                lis.append(ind)
        results = results[lis, :]

        # apply non-max suppression
        lis = []
        if self.iou_thres < 1.0:
            # split class for human and object
            # conditions = [results[:, 5] == 0, results[:, 5] != 0]
            conditions = [results[:, 5] != 100]
            for i in range(len(conditions)):
                res = results[(conditions[i]) & (results[:, 4] > 0)]
                for i in torch.argsort(-res[:, 4]):
                    ious = box_iou(res[i, :4].unsqueeze(0), res[:, :4])[0][0]
                    # Mask self-IoU.
                    ious[i] = -1.0 
                    res[:, 4][ious > self.iou_thres] = 0.0
                lis.append(res)
        results = torch.cat(lis, dim=0)
        results = results[results[:, 4] > 0]
        return results

### Yolo Model ###
class YoloLoader():
    '''yolo model to detect and plot the image'''
    def __init__(self):
        
        model_config = config['yolo']
        self.img_sz = config['video']['img_sz']
        self.device = model_config['device']
        self.font_size = model_config['font_size']
        self.classes = model_config['classes']
        self.conf_thres = model_config['conf_thres']
        self.iou_thres = model_config['iou_thres']
        self.max_det = model_config['max_det']
        self.agnostic_nms = model_config['agnostic_nms']
        self.model_name = model_config['model_name']
        self.device = select_device(model_config['device'])

        # load model, use FP16 half-precision inference, use OpenCV DNN for ONNX inference 
        # self.model = DetectMultiBackend(
        #     self.model_name, device=self.device, dnn=self.dnn, data='"./yolov5/data/Objects365.yaml"', fp16=self.half)
        self.model = AutoBackend(self.model_name, device=self.device, dnn=False, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.half = False

    def detect(self, image):
        # processing image before feeding to model
        im = torch.from_numpy(image).to(self.device)
        im = (im.half() if self.half else im.float()) / 255.0
        im = torch.unsqueeze(im, 0)
        im = torch.permute(im, (0, 3, 1, 2))

        # predictions and non max suppression
        preds = self.model(im, augment=False, visualize=False)
        results = non_max_suppression(
            preds, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        # Process detections
        annotator = Annotator(image, line_width=self.font_size, example=str(self.names))
        for _, det in enumerate(results):
            if det is not None and len(det): 
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], image.shape).round()
                for _, (output) in enumerate(det):
                    bbox = output[0:4]
                    conf = output[4]
                    obj_cls = output[5]
                    annotator.box_label(
                        bbox, f'{self.names[int(obj_cls)]} {conf:.2f}', color=colors(int(obj_cls), True))
        return image


