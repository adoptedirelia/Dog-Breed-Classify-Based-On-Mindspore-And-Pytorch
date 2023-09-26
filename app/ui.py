import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget, QFileDialog, QDialog, QVBoxLayout, QHBoxLayout, QStyleFactory, QMenuBar, QAction, QMessageBox
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt

import myui

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    
    ui = myui.Ui_MainWindow()
    ui.setupUi(window)
    window.setStyleSheet(
            "QMainWindow { background-color: #f7f7f7; }"
            "QLabel { background-color: white; border: 2px solid #ddd; border-radius: 12px; padding: 30px; }"
        )
    window.show()
    sys.exit(app.exec_())
