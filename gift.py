import cv2
import imageio
import numpy as np
import PIL
# 捕获视频
# cap = cv2.VideoCapture('your_video.mp4')
wel1 = cv2.imread('welcome1.jpg')
wel2 = cv2.imread('welcome2.jpg')
frame1 = cv2.cvtColor(wel1, cv2.COLOR_BGR2RGB)
frame2 = cv2.cvtColor(wel2, cv2.COLOR_BGR2RGB)
# 初始化帧列表
frames = []
for i in range(5):
    frames.append(frame1)
for i in range(5):
    frames.append(frame2)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#         # OpenCV读取的图像是BGR格式的，但我们需要将其转换为RGB格式以便后续处理
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frames.append(frame)
#
# cap.release()

# 使用imageio创建GIF动画
imageio.mimsave('output.gif', frames, fps=1)  # fps参数表示每秒的帧数