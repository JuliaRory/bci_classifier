import sys
from PyQt5.QtWidgets import QApplication


from ui.main_window import MainWindow

folder = r"R:\projects_Agency_EBCI\Agency_EBCI\models\test_classifier.json"
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    
    sys.exit(app.exec_())