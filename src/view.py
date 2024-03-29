from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog, QMainWindow
import sys
import constant

class Ui_MainWindow(object):
    def setupUi(self, Dialog):
        # 设置字体
        self.font = QtGui.QFont()
        self.font.setFamily("./Verdana")
        self.font.setPointSize(10)

        Dialog.setObjectName("Dialog")
        Dialog.resize(1024, 768)

        Dialog.setFont(self.font)
        Dialog.setLayoutDirection(QtCore.Qt.RightToLeft)
        Dialog.setAutoFillBackground(True)
        Dialog.setStyleSheet("")

        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(150, 30, 601, 111))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        self.label.setPalette(palette)

        self.label.setFont(self.font)
        self.label.setCursor(QtGui.QCursor(QtCore.Qt.UpArrowCursor))
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(110, 600, 141, 81))
        self.label_2.setObjectName("label_2")

        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(120, 200, 551, 401))
        self.graphicsView.setObjectName("graphicsView")

        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(120, 200, 551, 401))
        self.label_4.setObjectName("label_4")
        self.label_4.setScaledContents(True)  # label自适应图片大小
        self.label_4.setAlignment(QtCore.Qt.AlignLeft)  # 将图像向左对齐

        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(250, 620, 421, 41))
        self.lineEdit.setObjectName("lineEdit")

        self.lineEdit_2 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_2.setGeometry(QtCore.QRect(220, 140, 441, 41))
        self.lineEdit_2.setObjectName("lineEdit_2")

        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(720, 440, 111, 51))
        self.set_button_style(self.pushButton)
        self.pushButton.setObjectName("pushButton")

        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(720, 510, 111, 51))
        self.set_button_style(self.pushButton_2)
        self.pushButton_2.setObjectName("pushButton_2")

        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(120, 140, 111, 41))
        self.label_3.setObjectName("label_3")

        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(570, 680, 121, 16))

        self.label_5.setFont(self.font)
        self.label_5.setObjectName("label_5")

        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(720, 20, 141, 191))
        self.label_6.setObjectName("label_6")

        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(720, 370, 111, 51))
        self.pushButton_3.setMinimumSize(QtCore.QSize(111, 51))

        self.pushButton_3.setFont(self.font)
        self.pushButton_3.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.pushButton_3.setStyleSheet(
            "Button { background-color: rgb(255, 255, 255); border-radius: 3px; color: rgb(255, 255, 255); } QPushButton:hover { background-color: rgb(85, 170, 127); }")
        self.pushButton_3.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_3.setObjectName("pushButton_3")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)



        self.pushButton.setGeometry(QtCore.QRect(
            constant.marginOfButton, Dialog.height() - constant.button_bottom_margin - constant.heightOfButton, constant.widthOfButton, constant.heightOfButton))
        self.pushButton_2.setGeometry(QtCore.QRect(
            constant.marginOfButton + constant.widthOfButton + constant.marginOfButton, Dialog.height() - constant.button_bottom_margin - constant.heightOfButton,
            constant.widthOfButton, constant.heightOfButton))
        self.pushButton_3.setGeometry(QtCore.QRect(
            constant.marginOfButton + (constant.widthOfButton + constant.marginOfButton) * 2, Dialog.height() - constant.button_bottom_margin - constant.heightOfButton,
            constant.widthOfButton, constant.heightOfButton))

    def set_button_style(self, button):
        button.setFont(self.font)
        button.setLayoutDirection(QtCore.Qt.RightToLeft)
        button.setStyleSheet(
            "Button { background-color: rgb(255, 255, 255); border-radius: 3px; color: rgb(255, 255, 255); } QPushButton:hover { background-color: rgb(85, 170, 127); }")
        button.setIconSize(QtCore.QSize(30, 30))

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog",
                                      "<html><head/><body><p><span style=\" font-size:28pt; color:#428663;\">基于TensorFlow的垃圾分类系统</span></p></body></html>"))
        self.label_2.setText(_translate("Dialog",
                                        "<html><head/><body><p><span style=\" font-size:16pt;\">垃圾类型：</span></p></body></html>"))
        # self.label_4.setText(_translate("Dialog", "待测垃圾"))
        self.pushButton.setText(_translate("Dialog", "开始识别"))
        self.pushButton_2.setText(_translate("Dialog", "上传图片"))
        self.label_3.setText(_translate("Dialog",
                                        "<html><head/><body><p><span style=\" font-size:12pt;\">图片路径：</span></p></body></html>"))
        self.label_5.setText(_translate("Dialog",
                                        "<html><head/><body><p><span style=\" font-size:9pt; color:#306148;\">备用</span></p></body></html>"))
        self.label_6.setText(
            _translate("Dialog", "<html><head/><body><p><img src=\"./pic/垃圾桶.jpg\"/></p></body></html>"))
        self.pushButton_3.setText(_translate("Dialog", "摄像头模式"))


class ImageRecognitionApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(ImageRecognitionApp, self).__init__()
        self.setupUi(self)

        # 将按钮点击事件连接到相应的函数
        self.pushButton_2.clicked.connect(self.upload_image)
        self.pushButton.clicked.connect(self.start_recognition)

        # 设置默认图片路径
        image_path = "./pic/垃圾桶.jpg"
        self.load_image(image_path)

    def load_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.label_4.setPixmap(pixmap)

    def upload_image(self):
        # 打开文件对话框以获取图片文件路径
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, "选择图片文件", "", "图片文件 (*.png *.jpg *.bmp *.jpeg)")
        if image_path:
            # 加载所选图片
            self.load_image(image_path)
            # 在行编辑器中显示图片路径
            self.lineEdit.setText(image_path)

    def start_recognition(self):
        # 在这里添加启动图像识别的代码
        pass


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ImageRecognitionApp()
    window.show()
    sys.exit(app.exec())