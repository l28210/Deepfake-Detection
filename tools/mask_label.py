import cv2
import numpy as np
import matplotlib.pyplot as plt

drawing = False
mask = None
current_pts = []

def draw_callback(event, x, y, flags, param):
    global drawing, current_pts, mask
    brush_thickness = 45  # 设置笔刷粗细

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_pts = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_pts.append((x, y))
        for i in range(len(current_pts) - 1):
            cv2.line(param['img'], current_pts[i], current_pts[i + 1], (0, 255, 0), brush_thickness)
        cv2.imshow("Image", param['img'])

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if len(current_pts) > 1:
            for i in range(len(current_pts) - 1):
                cv2.line(param['img'], current_pts[i], current_pts[i + 1], (0, 255, 0), brush_thickness)
                cv2.line(mask, current_pts[i], current_pts[i + 1], 1, brush_thickness)
            current_pts = []

def annotate_image(image_path):
    global mask

    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image.")
        return None

    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_callback, {'img': img})

    print("Draw on the image with mouse. Press 'q' to quit and save mask.")
    while True:
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return mask


mask_result = annotate_image("/home/l/test_self/deepfake_detect/data/archive/test/fake/0A266M95TD.jpg")

if mask_result is not None:
    print("Mask shape:", mask_result.shape)
    print("Non-zero pixels (i.e., annotated):", np.count_nonzero(mask_result))
    # 保存 mask（可选）
    cv2.imwrite("mask.png", mask_result * 255)
