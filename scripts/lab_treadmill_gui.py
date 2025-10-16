"""
Lab Treadmill GUI Application
Main entry point for the treadmill analysis graphical user interface
"""
import sys
from PyQt5.QtWidgets import QApplication
from gui.treadmill_main_window import TreadmillMainWindow


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Lab MoCap - Treadmill Analysis")
    app.setOrganizationName("UniSA Biomechanics Lab")
    
    # Create and show main window
    window = TreadmillMainWindow()
    window.show()
    
    # Run application event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
