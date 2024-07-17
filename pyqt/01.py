import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
'''
使用 PyQt 创建一个简单的 GUI 应用程序涉及以下步骤 -
从 PyQt5 包中导入 QtCore、QtGui 和 QtWidgets 模块。
创建一个 QApplication 类的应用程序对象。
'''
# def window():
#    app = QApplication(sys.argv)
#    w = QWidget()  # 创建顶级窗口
#    b = QLabel(w)  # 添加 QLabel 对象
#    b.setText("Hello World!")  # 将标签的标题设置为“hello world”。
#    w.setGeometry(800,500,200,200)  # 通过 setGeometry() 方法定义窗口的大小和位置
#    b.move(50,20)
#    w.setWindowTitle("PyQt5")
#    w.show()
#    sys.exit(app.exec_())   # 通过以下方式进入应用程序的主循环app.exec_()方法。
class window(QWidget):
   def __init__(self, parent = None):
      super(window, self).__init__(parent)
      self.resize(200,100)
      self.setWindowTitle("PyQt5")
      self.label = QLabel(self)
      self.label.setText("Hello World")
      font = QFont()
      font.setFamily("Arial")
      font.setPointSize(16)
      self.label.setFont(font)
      self.label.move(50,20)
def main():
   app = QApplication(sys.argv)
   ex = window()
   ex.show()
   sys.exit(app.exec_())
if __name__ == '__main__':
   # window()
   main()