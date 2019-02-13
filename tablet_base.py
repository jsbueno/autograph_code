
import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGraphicsPixmapItem
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint

SIZE = 800, 600

class Test:
    def __init__(self):
        window = self.window = QLabel()
        window.setGeometry(0, 0, *SIZE)
        window.setWindowTitle("Tablet Scribling")
        window.paintEvent = self.paintEvent
        window.tabletEvent = self.tabletEvent
        window.keyPressEvent = self.keyPressEvent
        window.show()
        self.text = "fnord"
        window.setMouseTracking(True)
        self.points = []
        self.last_point = None
        self.pen = QPen(Qt.black, 1, Qt.SolidLine)
        self.image = QImage(*SIZE, QImage.Format_ARGB32)
        self.image.fill(QColor(255, 255, 255))
        window.setPixmap(QPixmap.fromImage(self.image))


    def paintEvent(self, event):
        # print(event, dir(event))
        qp = QPainter()
        qp.begin(self.image)
        qp.setRenderHints(QPainter.Antialiasing, True)
        qp.setPen(self.pen)
        # self.drawText(event, qp)
        qp.setPen(Qt.black)
        self.draw_blurbs(qp)
        # self.window.setPixmap(QPixmap.fromImage(self.image))
        # self.window.pixmap().fill(Qt.red)
        qp.end()
        qp = QPainter()
        qp.begin(self.window)
        qp.drawImage(QPoint(0,0), self.image)
        qp.end()


    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Delete:
            self.image.fill(QColor(255, 255, 255))
            self.window.update()
        elif key == Qt.Key_Return:
            self.image.save("image.png")

    #def mouseMoveEvent(self, event):
        #print(event, event.pos())
        #event.accept()
        #self.points.append((event.x(), event.y()))
        #self.window.update()

    def tabletEvent(self, event):
        print(event, event.pos(), event.pressure())
        event.accept()
        self.points.append((event.x(), event.y(), event.pressure()))
        self.window.update()

    def draw_blurbs(self, qp):
        if not self.last_point:
            if self.points:
                if self.points[0][2]:
                    self.last_point = self.points[0]
                else:
                    self.points[:] = []
            return
        lp = self.last_point
        for p in self.points:
            if p[2] != lp[2]:
                # pen = QtGui.QPen(QtCore.Qt.black, 2, QtCore.Qt.SolidLine)
                self.pen.setWidth(p[2] * 60)
                qp.setPen(self.pen)

            qp.drawLine(lp[0], lp[1], p[0], p[1])

            if p[2]:
                self.last_point = p
            else:
                self.last_point = None
        self.points[:] = []


def main():

    app = QApplication(sys.argv)
    w = Test()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
