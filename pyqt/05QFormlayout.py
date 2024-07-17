import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


def window():
    app = QApplication(sys.argv)
    win = QWidget()

    l1 = QLabel("Name")
    nm = QLineEdit()   # 单行文本输入框

    l2 = QLabel("Address")
    add1 = QLineEdit() # 单行文本输入框
    add2 = QLineEdit() # 单行文本输入框
    fbox = QFormLayout()
    fbox.addRow(l1, nm)  # name：......
    vbox = QVBoxLayout()

    vbox.addWidget(add1) #
    vbox.addWidget(add2)
    fbox.addRow(l2, vbox)
    hbox = QHBoxLayout()

    r1 = QRadioButton("Male")  # 带有文本标签的可选按钮
    r2 = QRadioButton("Female")
    hbox.addWidget(r1)
    hbox.addWidget(r2)
    hbox.addStretch()
    fbox.addRow(QLabel("sex"), hbox)
    fbox.addRow(QPushButton("Submit"), QPushButton("Cancel"))

    win.setLayout(fbox)

    win.setWindowTitle("PyQt")
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    window()
