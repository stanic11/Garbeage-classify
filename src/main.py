import sys
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox
from view import Ui_MainWindow
from classify import Model, generate_result
import numpy as np
import constant
from tensorflow.keras.preprocessing import image
import cv2


class Window(QtWidgets.QWidget, Ui_MainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.pathChoose)  # 上传图片
        self.pushButton.clicked.connect(self.Recognition)  # 开始识别
        self.pushButton_3.clicked.connect(self.cameraMode)  # 摄像头模式
        self.test_path = './dataset-resized'

    def pathChoose(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open file dialog", self.test_path, "图片(*.jpg)")
        print(file_name)
        self.test_path = file_name
        self.lineEdit_2.setText(self.test_path)  # 显示路径
        self.label_4.setPixmap(QtGui.QPixmap(self.test_path))  # 显示待测图片

        # 清空不相关内容
        self.lineEdit.clear()

    def Recognition(self):
        path = self.lineEdit_2.text()  # 存获取的地址
        garbage_type = ''
        if path == "":
            QMessageBox.warning(self, "警告", "请插入图片", QMessageBox.Yes, QMessageBox.Yes)
            return
        else:
            try:
                model = Model()
                model1 = model.loadModel()
                print('model loaded')
                img_path = path
                img = image.load_img(img_path, target_size=(300, 300))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                result = model1.predict(img)
                print(generate_result(result))
                result0 = generate_result(result)
                garbage = result0
                # 对结果进行分类
                if garbage in constant.DryWasteMap:
                    garbage_type = "干垃圾"
                elif garbage in constant.recyclableWasteMap:
                    garbage_type = "可回收垃圾"
                self.lineEdit.setText(garbage_type)
                QMessageBox.information(self, "提醒", "成功识别！该垃圾为：" + garbage, QMessageBox.Yes, QMessageBox.Yes)
            except Exception as e:
                QMessageBox.warning(self, "错误", f"发生错误：{str(e)}", QMessageBox.Yes, QMessageBox.Yes)

    def cameraMode(self):
        garbage_type = ''
        try:
            model = Model()
            model1 = model.loadModel()
            print('模型加载成功')

            cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()
                # 将OpenCV帧转换为QImage
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_image = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

                # 调整图像大小以适应label_4的尺寸
                q_image = q_image.scaled(self.label_4.width(), self.label_4.height(), QtCore.Qt.KeepAspectRatio)

                # 将QImage设置为label_4的Pixmap
                self.label_4.setPixmap(QtGui.QPixmap.fromImage(q_image))

                img = cv2.resize(frame, (300, 300))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                result = model1.predict(img)
                print(generate_result(result))
                result0 = generate_result(result)
                garbage = result0
                # 对结果进行分类
                if garbage in constant.DryWasteMap:
                    garbage_type = "干垃圾"
                elif garbage in constant.recyclableWasteMap:
                    garbage_type = "可回收垃圾"
                # 更新界面上的文本框
                self.lineEdit.setText(garbage_type)
                # 检测按键事件，按下 ESC 键退出摄像头
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # 按下 ESC 键退出摄像头
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            QMessageBox.warning(self, "错误", f"发生错误：{str(e)}", QMessageBox.Yes, QMessageBox.Yes)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = Window()
    main.setWindowTitle("基于TensorFlow的垃圾分类系统")
    main.show()
    sys.exit(app.exec())