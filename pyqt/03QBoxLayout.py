import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


def window():
    app = QApplication(sys.argv)
    win = QWidget()

    b1 = QPushButton("Button1")
    b2 = QPushButton("Button2")

    vbox = QVBoxLayout()
    vbox.addWidget(b1)
    vbox.addStretch()
    vbox.addWidget(b2)
    hbox = QHBoxLayout()

    b3 = QPushButton("Button3")
    b4 = QPushButton("Button4")
    hbox.addWidget(b3)
    hbox.addStretch() # 可拉伸的空白空间
    hbox.addWidget(b4)

    vbox.addStretch()
    vbox.addLayout(hbox)
    win.setLayout(vbox)

    win.setWindowTitle("PyQt")
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    window()
