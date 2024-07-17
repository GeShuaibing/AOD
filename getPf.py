import os
import cv2
import shutil
import time

# 全局变量
VIDEO_PATH = r'G:\ABODA-master\ABODA-master\v10_Trim.mp4'  # 视频地址
EXTRACT_FOLDER = r'E:\BaiduNetdiskDownload\shoutibao\img7'  # 存放帧图片的位置
EXTRACT_FREQUENCY = 5  # 帧提取频率
video = cv2.VideoCapture(VIDEO_PATH)
index = 1
while True:
    print(0)
    _, frame = video.read()
    save_path = r"{}\s10trim{}.jpg".format(EXTRACT_FOLDER,index)
    if frame is None:
        break
    if index % EXTRACT_FREQUENCY == 0:
        cv2.imwrite(save_path, frame)
    index += 1  # 保存图片数＋1
video.release()
# 主操作
# def extract_frames(video_path, dst_folder, index):
#     # 实例化视频对象
#     video = cv2.VideoCapture(video_path)
#     frame_count = 0
#
#     # 循环遍历视频中的所有帧
#     while True:
#         # 逐帧读取
#         _, frame = video.read()
#         if frame is None:
#             break
#         # 按照设置的频率保存图片
#         if frame_count % EXTRACT_FREQUENCY == 0:
#             # 设置保存文件名
#             save_path = "{}/{:>03d}.jpg".format(dst_folder, index)
#             # 保存图片
#             cv2.imwrite(save_path, frame)
#             index += 1  # 保存图片数＋1
#         frame_count += 1  # 读取视频帧数＋1
#
#     # 视频总帧数
#     print(f'the number of frames: {frame_count}')
#     # 打印出所提取图片的总数
#     print("Totally save {:d} imgs".format(index - 1))
#
#     # 计算FPS 方法一 get()
#     (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')  # Find OpenCV version
#     # (major_ver, minor_ver, subminor_ver) = (4, 5, 4)
#     if int(major_ver) < 3:
#         fps = video.get(cv2.cv.CV_CAP_PROP_FPS)  # 获取当前版本opencv的FPS
#         print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
#     else:
#         fps = video.get(cv2.CAP_PROP_FPS)  # 获取当前版本opencv的FPS
#         print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
#
#     # 计算FPS 方法二 手动计算 总帧数 / 总时间
#     # new_vid = cv2.VideoCapture(video_path)
#     # start = time.time()  # 开始时间
#     # c = 0
#     # while True:
#     #     _, frames = new_vid.read()
#     #     if frames is None:
#     #         break
#     #     c += 1
#     # end = time.time()  # 结束时间
#     # fps2 = c / (end - start)  # 总帧数 / 总时间
#     # print(f'frames:{c}')
#     # print(f'seconds:{end - start}')
#     # print("Frames per second using frames / seconds : {0}".format(fps2))
#     # new_vid.release()
#
#     # 最后释放掉实例化的视频
#     video.release()

#
# def main():
#     # 递归删除之前存放帧图片的文件夹，并新建一个
#     try:
#         shutil.rmtree(EXTRACT_FOLDER)
#     except OSError:
#         pass
#     os.mkdir(EXTRACT_FOLDER)
#     # 抽取帧图片，并保存到指定路径
#     extract_frames(VIDEO_PATH, EXTRACT_FOLDER, 1)
#
#
# if __name__ == '__main__':
#     main()
