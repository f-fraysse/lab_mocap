"""
Lab MoCap GUI Application
Main entry point for the graphical user interface
"""
import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Lab MoCap")
    app.setOrganizationName("UniSA Biomechanics Lab")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
