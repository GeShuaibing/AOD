import cv2
from ultralytics import YOLO
import AbDetection
# Load the YOLOv8 model
model = YOLO(r"yolov8m.pt")
# print('111')
# Open the video file
vedio = r"G:\ABODA-master\ABODA-master\test.mp4"
# vedio = r"result.avi"
# vedio = r'video3_Trim.mp4'
cap = cv2.VideoCapture(vedio)
out = cv2.VideoWriter('result2.avi',cv2.VideoWriter_fourcc('I','4','2','0'),20,(int(cap.get(3)),int(cap.get(4))))
# Loop through the video frames
last = []
n = 0
while (True):
    # current = []
    # Read a frame from the video
    success, frame = cap.read()
    # Run YOLOv8 inference on the frame
    results = model.track(frame, persist=True)
    # print(results)
    # Visualize the results on the frame
    # if results[0].boxes.is_track:
    #     pass
    # else:
    #     continue
    annotated_frame = results[0].plot()
    text = f"Frame:{n}"
    cv2.putText(annotated_frame, text, (25, 25), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 1)
    current = []
    if results[0].boxes.is_track:

        label =results[0].boxes.cls.tolist()
        box = results[0].boxes.xyxy.tolist()
        ids = results[0].boxes.id.tolist()
    # print(box)
        current = AbDetection.curFb_list(label,box,results[0].names,ids)
        last = AbDetection.abdetect(last,current)
        annotated_frame = AbDetection.draw_lost(last,annotated_frame)

    # Display the annotated frame
    out.write(annotated_frame)
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    # Break the loop if 'q' is pressed
    k = cv2.waitKey(10) & 0xff
    if k == 27:  # esc
        break
    n = n+1

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()