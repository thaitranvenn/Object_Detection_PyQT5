from PyQt5 import QtCore, QtGui, QtWidgets


class RegisterForm(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 300)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("register.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form.setWindowIcon(icon)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(90, 10, 241, 61))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.layoutWidget = QtWidgets.QWidget(Form)
        self.layoutWidget.setGeometry(QtCore.QRect(120, 80, 181, 24))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.editUsername = QtWidgets.QLineEdit(self.layoutWidget)
        self.editUsername.setObjectName("editUsername")
        self.horizontalLayout.addWidget(self.editUsername)
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(30, 30, 54, 51))
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.layoutWidget_3 = QtWidgets.QWidget(Form)
        self.layoutWidget_3.setGeometry(QtCore.QRect(120, 130, 181, 24))
        self.layoutWidget_3.setObjectName("layoutWidget_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget_3)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.layoutWidget_3)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.editPassword = QtWidgets.QLineEdit(self.layoutWidget_3)
        self.editPassword.setObjectName("editPassword")
        self.horizontalLayout_2.addWidget(self.editPassword)
        self.pushButtonRegister = QtWidgets.QPushButton(Form)
        self.pushButtonRegister.setGeometry(QtCore.QRect(150, 180, 93, 28))
        self.pushButtonRegister.setObjectName("pushButtonRegister")
        self.pushButtonCancel = QtWidgets.QPushButton(Form)
        self.pushButtonCancel.setGeometry(QtCore.QRect(300, 250, 93, 28))
        self.pushButtonCancel.setObjectName("pushButtonCancel")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Register an account"))
        self.label.setText(_translate("Form", "Register an account"))
        self.label_2.setText(_translate("Form", "Login      "))
        self.label_3.setText(_translate("Form", "Password"))
        self.pushButtonRegister.setText(_translate("Form", "Register"))
        self.pushButtonCancel.setText(_translate("Form", "Cancel"))
