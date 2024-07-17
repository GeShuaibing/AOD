import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class tooldemo(QMainWindow):
    def __init__(self, parent=None):
        super(tooldemo, self).__init__(parent)
        layout = QVBoxLayout()
        tb = self.addToolBar("File")

        new = QAction(QIcon("R-C6.jpg"), "new", self)
        tb.addAction(new)

        open = QAction(QIcon("R-C6.jpg"), "open", self)
        tb.addAction(open)
        save = QAction(QIcon("R-C6.jpg"), "save", self)
        tb.addAction(save)
        tb.actionTriggered[QAction].connect(self.toolbtnpressed)
        self.setLayout(layout)
        self.setWindowTitle("toolbar demo")

    def toolbtnpressed(self, a):
        print("pressed tool button is", a.text())


def main():
    app = QApplication(sys.argv)

    ex = tooldemo()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
