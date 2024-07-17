from ultralytics import YOLO
import torch
# iou_type: WIoUv3  # IoU , CIoU , GIoU , EIoU ,FEIoU , WIoUv1 , WIoUv2 , WIoUv3
if __name__ == '__main__':
    model = YOLO("yolov8n-starnet.yaml")
    # results = model.train(data='ultralytics/cfg/datasets/aoddata.yaml', epochs=200, batch=16, imgsz=640, name="Civi",
    #                       iou_type='WIoUv3')
    print(model)
    # a = torch.empty(1)
    # b = torch.empty([2,3])
    # print(b*a)
