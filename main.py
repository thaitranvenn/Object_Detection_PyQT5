import sys
from PyQt5.QtWidgets import *
from utils.id import getId, saveId
from lib.share import shareInfo
from login_ui import LoginForm
from register_ui import RegisterForm
from GUI import Ui_MainWindow

# Class này là giao diện login
class windowLogin(QMainWindow):
    def __init__(self, parent = None):
        super(windowLogin, self).__init__(parent)
        self.loginUI = LoginForm()
        self.loginUI.setupUi(self)
        self.initSlots()
        self.hidenPassword()

    # The password edit box is hidden
    def hidenPassword(self):
        self.loginUI.editPassword.setEchoMode(QLineEdit.Password)

    def initSlots(self):
        # Using login button and enter key
        self.loginUI.buttonLogin.clicked.connect(self.signIn)
        self.loginUI.editPassword.returnPressed.connect(self.signIn)
        self.loginUI.buttonRegister.clicked.connect(self.createID)

    # Skip to register UI
    def createID(self):
        shareInfo.createWin = windowRegister()
        shareInfo.createWin.show()

    # This function is to login using login button or press enter
    def signIn(self):
        print("You pressed sign in")
        # Lấy tài khoản mật khẩu từ giao diện đăng nhập
        username = self.loginUI.editUsername.text().strip()
        password = self.loginUI.editPassword.text().strip()

        # Get information login and password for Login UI
        USER_PWD = getId()
        #print(USER_PWD)

        if username not in USER_PWD.keys():
            replay = QMessageBox.warning(self, "Failed to login!", "The login or password is incorrect!", QMessageBox.Ok)
        else:
            # Skip to main UI
            if USER_PWD.get(username) == password:
                print("Skipping to main window")
                shareInfo.mainWin = Ui_MainWindow()
                shareInfo.mainWin.show()
                self.close()

# Register GUI
class windowRegister(QDialog):
    def __init__(self, parent = None):
        super(windowRegister, self).__init__(parent)
        self.registerUI = RegisterForm()
        self.registerUI.setupUi(self)
        self.initSlots()

    # Click to register or cancel the register
    def initSlots(self):
        self.registerUI.pushButtonRegister.clicked.connect(self.newAccount)
        self.registerUI.pushButtonCancel.clicked.connect(self.cancel)

    # Create new account
    def newAccount(self):
        print("Create new account")
        USER_PWD = getId()
        #print(USER_PWD)
        newUsername = self.registerUI.editUsername.text().strip()
        newPassword = self.registerUI.editPassword.text().strip()
        # Check username is empty
        if newUsername == "":
            replay = QMessageBox.warning(self, "Abort", "Username is empty!", QMessageBox.Ok)
        else:
            # Check username is exist
            if newUsername in USER_PWD.keys():
                replay = QMessageBox.warning(self, "Abort", "Username is exist!", QMessageBox.Ok)
            else:
                # Check password is empty
                if newPassword == "":
                    replay = QMessageBox.warning(self, "Abort", "Password is empty!", QMessageBox.Ok)
                else:
                    # Successful created account
                    print("Successful!")
                    saveId(newUsername, newPassword)
                    replay = QMessageBox.warning(self, "Abort", "Successful!", QMessageBox.Ok)
                    self.close()

    def cancel(self):
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Chọn giao diện đăng nhập làm giao diện chính
    shareInfo.loginWin = windowLogin()
    shareInfo.loginWin.show()
    sys.exit(app.exec_())