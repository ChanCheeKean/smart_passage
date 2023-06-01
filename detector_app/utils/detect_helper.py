import cv2
import torch
from utils.config_loader import config
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.utils.plotting import Annotator, colors
# from yolov5.models.common import DetectMultiBackend
# from yolov5.utils.general import non_max_suppression, scale_boxes
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.utils.ops import non_max_suppression, scale_boxes

### non max suppression
def box_iou(boxes1, boxes2):
    def _upcast(t):
        # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
        if t.is_floating_point():
            return t if t.dtype in (torch.float32, torch.float64) else t.float()
        else:
            return t if t.dtype in (torch.int32, torch.int64) else t.int()
    def box_area(boxes):
        """
        Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.
        """
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

class VitModelLoader():
    def __init__(self):
        '''OWL-Vit model to detect and plot the image'''
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
        image = cv2.resize(image, self.img_sz, interpolation=cv2.INTER_AREA)
        inputs = self.processor(text=[self.query], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.Tensor([image.shape[0:2]])
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self.conf_thres)[0]
        annotator = Annotator(image, line_width=2, example=self.labels)
        results = torch.cat(
            tensors=(results["boxes"], results["scores"].view(-1, 1), results["labels"].view(-1, 1)),
            dim=1)
        boxes, scores, labels = results[:, :4], results[:, 4], results[:, 5]

        # for human and object
        colors = [(0, 0, 255), (255, 255, 0)]
        label_cls = ['human', 'object']
        cond = [labels==0, labels!=0]
        threshold = [self.conf_thres + 0.1, self.conf_thres]

        for c in range(2):
            target_boxes = boxes[cond[c]]
            target_labels = labels[cond[c]]
            target_scores = scores[cond[c]]

            # Apply non-maximum suppression (NMS)
            if self.iou_thres < 1.0:
                for i in torch.argsort(-target_scores):
                    if not target_scores[i]:
                        continue
                    ious = box_iou(target_boxes[i, :].unsqueeze(0), target_boxes)[0][0]
                    ious[i] = -1.0  # Mask self-IoU.
                    target_scores[ious > self.iou_thres] = 0.0

            # plotting
            for score, label, box in zip(target_scores, target_labels, target_boxes):
                if score > threshold[c]:
                    bbox = [round(i, 2) for i in box.tolist()]
                    conf = round(score.item(), 3)
                    label_text = label_cls[c]
                    label_text =  self.labels[int(label)]
                    annotator.box_label(bbox, f'{label_text} {conf:.2f}', color=colors[c])
        return image

class YoloLoader():
    '''yolo model to detect and plot the image'''
    def __init__(self):
        
        model_config = config['yolo']
        self.img_sz = config['video']['img_sz']
        self.device = model_config['device']
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
        image = cv2.resize(image, self.img_sz, interpolation=cv2.INTER_AREA)
        im = torch.from_numpy(image).to(self.device)
        im = (im.half() if self.half else im.float()) / 255.0
        im = torch.unsqueeze(im, 0)
        im = torch.permute(im, (0, 3, 1, 2))

        # predictions and non max suppression
        preds = self.model(im, augment=False, visualize=False)
        results = non_max_suppression(
            preds, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        # Process detections
        annotator = Annotator(image, line_width=2, example=str(self.names))
        for _, det in enumerate(results):
            if det is not None and len(det): 
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], image.shape).round()
                for _, (output) in enumerate(det):
                    bbox = output[0:4]
                    conf = output[4]
                    cls = output[5]
                    annotator.box_label(
                        bbox, f'{self.names[int(cls)]} {conf:.2f}', color=colors(int(cls), True))
        return image
