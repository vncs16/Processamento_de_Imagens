# src/main.py
import sys
from PySide6.QtWidgets import QApplication # Alterado de PyQt5
from .gui import ImageProcessingApp

def main():
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()