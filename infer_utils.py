from keras_cv import bounding_box

import cv2

class_names =  ['QR code', 'bag', 'bag with straw', 'basket of food', 'beer mug', 'blue QR code', 'bottle', 'bowl of food', 'chopstick', 'cup', 'fork', 'glass', 'jug', 'knife', 'menu', 'mobile', 'napkin', 'pan of food', 'people', 'pepper', 'plate of food', 'salt', 'soda can', 'spoon', 'table', 'tray of food', 'vase with flower', 'waiter', 'wine glass']


color_mapping = {
    "red": (255, 0, 0),
    "orange": (255, 128, 0),
    "yellow": (255, 255, 0),
    "lime_green": (128, 255, 0),
    "green": (0, 255, 0),
    "teal": (0, 255, 128),
    "cyan": (0, 255, 255),
    "sky_blue": (0, 128, 255),
    "blue": (0, 0, 255),
    "purple": (128, 0, 255),
    "magenta": (255, 0, 255),
    "pink": (255, 0, 128),
    "vermilion": (255, 64, 0),
    "amber": (255, 192, 0),
    "turquoise": (128, 255, 192),
    "gray": (64, 64, 64),
    "silver": (128, 128, 128),
    "light_gray": (192, 192, 192),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "crimson": (255, 0, 64),
    "rose": (255, 128, 192),
    "spring_green": (128, 255, 64),
    "aquamarine": (0, 255, 192),
    "ruby": (192, 0, 64),
    "copper": (192, 128, 64),
    "olive": (192, 192, 0),
    "chartreuse": (128, 192, 0),
    "forest_green": (64, 192, 0)
}

def decode_detection(output):
    output = bounding_box.to_ragged(output)
    boxes = output['boxes'].numpy()[0]
    classes = output['classes'].numpy()[0]
    scores = output['confidence'].numpy()[0]
    return boxes, classes, scores

def plot_boxes(image, boxes, classes, scores, threshold, class_names):
    # The boxes are sorted as per decreasing confidence score.
    for i, box in enumerate(boxes):
        if scores[i] >= threshold:
            class_name = class_names[int(classes[i])]
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color=color_mapping[class_name], 
                thickness=2,
                lineType=cv2.LINE_AA
            )
            cv2.putText(
                image,
                text=class_name,
                org=((int(box[0]), int(box[1]-5))),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=color_mapping[class_name],
                thickness=2,
                lineType=cv2.LINE_AA
            )
    return image