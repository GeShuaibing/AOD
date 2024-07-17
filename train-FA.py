from ultralytics import YOLO
import torch
# iou_type: WIoUv3  # IoU , CIoU , GIoU , EIoU ,FEIoU , WIoUv1 , WIoUv2 , WIoUv3
if __name__ == '__main__':
    model = YOLO("yolov8s-cpca.yaml")
    # # results = model.train(data='ultralytics/cfg/datasets/aoddata.yaml', epochs=200, batch=16, imgsz=640, name="MPA"
    # #                       )
    print(model)
    # print(torch.ones(2))