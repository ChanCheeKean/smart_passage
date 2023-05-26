# import cv2
# gate_xyxy = {
#     'top': 400,
#     'bottom': 620,
#     'left': 80,
#     'right': 1150,
#     'left-safety': 280,
#     'right-safety':800
# }

# img_pth = './data/processed/batch_1/images/1684918616.jpg'
# img = cv2.imread(img_pth)
# img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)

# cv2.rectangle(
#     img,
#     (gate_xyxy['left'], gate_xyxy['top']),
#     (gate_xyxy['right'], gate_xyxy['bottom']),
#     (255, 0, 0), 
#     2
# )

# cv2.rectangle(
#     img,
#     (gate_xyxy['left-safety'], gate_xyxy['top']),
#     (gate_xyxy['right-safety'], gate_xyxy['bottom']),
#     (255, 255, 0), 
#     2
# )

# import matplotlib.pyplot as plt
# imgplot = plt.imshow(img)
# plt.show()

temp = {'0': {'precision': 0.2, 'recall': 0.3, 'f1-score': 0.24, 'support': 10}, '1': {'precision': 0.3333333333333333, 'recall': 0.3333333333333333, 'f1-score': 0.3333333333333333, 'support': 12}, '2': {'precision': 0.3333333333333333, 'recall': 0.3076923076923077, 'f1-score': 0.32, 'support': 13}, 'micro avg': {'precision': 0.28205128205128205, 'recall': 0.3142857142857143, 'f1-score': 0.29729729729729726, 'support': 35}, 'macro avg': {'precision': 0.2888888888888889, 'recall': 0.31367521367521367, 'f1-score': 0.29777777777777775, 'support': 35}, 'weighted avg': {'precision': 0.2952380952380952, 'recall': 0.3142857142857143, 'f1-score': 0.3017142857142857, 'support': 35}, 'samples avg': {'precision': 0.28205128205128205, 
'recall': 0.28205128205128205, 'f1-score': 0.28205128205128205, 'support': 35}}

print(temp['weighted avg'].keys())
