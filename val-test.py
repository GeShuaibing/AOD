from ultralytics import YOLO
import cv2 as cv
path = r'R-C6.jpg'

# iou_type: WIoUv3  # IoU , CIoU , GIoU , EIoU ,FEIoU , WIoUv1 , WIoUv2 , WIoUv3
if __name__ == '__main__':
    img = cv.imread(path)
    model = YOLO('yolov8s.pt')
    # results = model.val(data='ultralytics/cfg/datasets/aoddata.yaml', batch=1, imgsz=640, name="val-EMA-L",
    #                       )
    res = model(img)
    img = res[0].plot()
    cv.imwrite("rc-6.jpg",img)