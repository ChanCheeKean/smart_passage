import cv2
import matplotlib.pyplot as plt
from utils.video_loader import ImageLoader
from utils.cv_helper import (
    DeepSortTracker, 
    YoloLoader, 
    plot_gate_roi, 
    plot_image, 
    tailgating_detection, 
    update_zone_info, 
    detect_dir,
    detect_loiter,
)

camera = ImageLoader()
model = YoloLoader()
object_tracker = DeepSortTracker()

while True:
    ret, image = camera.get_frame()
    image = cv2.resize(image, camera.img_sz, interpolation=cv2.INTER_AREA)
    results = model.detect(image)
    print(results.size())
    results = object_tracker.update(image, results)
    plt.imshow(image)
    plt.show()
