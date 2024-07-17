import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

#
# x_ = np.linspace(-5,5,1000)
# # print(x_)
# y_l1 = np.abs(x_)
# y_l2 = np.power(x_,2)
# y_smothl1 = np.where(np.abs(x_)<1,0.5*x_**2,np.abs(x_)-0.5)
# plt.plot(x_,y_l1,label="L1")
# plt.plot(x_,y_l2,label="L2")
# plt.plot(x_,y_smothl1,label="Smooth L1")
# plt.xlabel("f(x)-y")
# plt.ylabel("Ln Loss")
# plt.ylim(0,5)
# plt.legend()
# plt.show()


# 读入图片
src = cv.imread(r"G:\img-AOD\bag\0bag_img3.jpg")

# 调用cv.putText()添加文字
text = f"Frame:{1}"
text2 = 'exist abd'
AddText = src.copy()
cv.putText(AddText, text, (25, 25), cv.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 1)
cv.putText(AddText, text2, (25, 50), cv.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 1)
# 将原图片和添加文字后的图片拼接起来
# res = np.hstack([src, AddText])

# 显示拼接后的图片
cv.imshow('text', AddText)
cv.waitKey()
cv.destroyAllWindows()