import cv2
import matplotlib.pyplot as plt

def show_input_boxes(image_source, input_boxes, color=(0, 255, 0), thickness=2):
    img = image_source.copy()

    for x1, y1, x2, y2 in input_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    plt.imshow(img)
    plt.axis("off")
    plt.show()
