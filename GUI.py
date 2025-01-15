import sys
import os
import json
import logging
import pandas as pd
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, pyqtSlot, QRectF, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton,
    QLabel, QFileDialog, QLineEdit, QComboBox, QSpinBox, QProgressBar,
    QMessageBox, QTableWidget, QTableWidgetItem, QDialog,
    QHeaderView, QTreeWidget, QTreeWidgetItem, QSizePolicy,
    QPlainTextEdit, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QSplashScreen
)
import threading

from Preparation import load_and_prepare_image, rot_img
from Processing import seg_proc, gini_coefficient
from Evaluation import analyze_and_visualize_results, analyze_and_visualize_results_2d, extract_and_save_data, plot_fft

from Calculations import apply_size_correction, round_values
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar


# Logging configuration
def setup_logger(name="MyAppLogger", level=logging.INFO, log_file=None, formatter=None):
    """
    Set up a logger with console and optional file handlers.

    :param name: Name of the logger.
    :param level: Logging level (e.g., logging.INFO, logging.DEBUG).
    :param log_file: Optional path to a log file.
    :param formatter: Optional custom formatter string.
    :return: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():  # Prevent duplicate handlers
        logger.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            formatter or "%(asctime)s [%(levelname)s] %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)

    return logger

class MultiOutput(QObject):
    """
    A class that redirects standard output (stdout) to the GUI,
    while optionally also being displayed in the console.
    """
    new_text = pyqtSignal(str)

    def __init__(self, enable_console_output=True):
        """
        Initialises the MultiOutput class.
        param enable_console_output: If true, the output is still displayed in the console.
        """
        super().__init__()
        self.original_stdout = sys.__stdout__
        self.lock = threading.Lock()  # Thread-Sicherheit

    def write(self, text):
        """
        Writes text to the GUI and optionally to the console.
        """
        if self.original_stdout:
            self.original_stdout.write(text)
            self.original_stdout.flush()
        self.new_text.emit(text)

    def flush(self):
        """
        Empties the buffers for stdout.
        """
        if self.original_stdout:
            self.original_stdout.flush()

class ConsoleOutputDialog(QDialog):
    """
    A dialog to display console output in a GUI with options to scroll, clear and close.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Console Output")
        #Layout Setup
        layout = QVBoxLayout(self)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)
        self.setLayout(layout)
        self.resize(400, 200)

    @pyqtSlot(str)
    def append_text(self, text):
        """
        Slot to append text to the QPlainTextEdit widget.
        """
        self.text_edit.appendPlainText(text)

class MultiImageInfoDialog(QDialog):
    """
    A dialog to input metadata (magn, width, height) for each file in a given list. 
    """
    def __init__(self, files_list, parent=None, default_values=None):
        """
        Initialize the dialog.

        :param files_list: List of filenames to display.
        :param parent: Parent widget (optional).
        :param default_values: Default values for metadata (optional).
        """
        super().__init__(parent)
        self.setWindowTitle("Enter Image Info for Each File")
        self.files_list = files_list
        self.file_info = {}
        self.default_values = default_values or {"magnification": "1.0", "width_um": "100.0", "height_um": "100.0"}

        self.table = QTableWidget(len(files_list), 4)
        self.table.setHorizontalHeaderLabels(["Filename", "Magnification", "Width (µm)", "Height (µm)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)

        # Fill the table with filenames and default metadata
        for row, fname in enumerate(files_list):
            item_fname = QTableWidgetItem(fname)
            item_fname.setFlags(item_fname.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(row, 0, item_fname)

            # Default
            self.table.setItem(row, 1, QTableWidgetItem(self.default_values["magnification"]))
            self.table.setItem(row, 2, QTableWidgetItem(self.default_values["width_um"]))
            self.table.setItem(row, 3, QTableWidgetItem(self.default_values["height_um"]))
            
            # Add action buttons
        btn_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.on_ok)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.resize(500, 300)
        
    def reset_to_defaults(self):
        """
        Reset all metadata entries to their default values.
        """
        for row in range(self.table.rowCount()):
            self.table.setItem(row, 1, QTableWidgetItem(self.default_values["magnification"]))
            self.table.setItem(row, 2, QTableWidgetItem(self.default_values["width_um"]))
            self.table.setItem(row, 3, QTableWidgetItem(self.default_values["height_um"]))

        
    def on_ok(self):
        """
        Validate and save the entered metadata for each file.
        """
        for row in range(self.table.rowCount()):
            fname_item = self.table.item(row, 0)
            mag_item   = self.table.item(row, 1)
            w_item     = self.table.item(row, 2)
            h_item     = self.table.item(row, 3)

            fname = fname_item.text()
            try:
                mag = float(mag_item.text())
            except ValueError:
                mag = float(self.default_values["magnification"])
                mag_item.setBackground(Qt.red)
            try:
                w_um = float(w_item.text())
            except ValueError:
                w_um = float(self.default_values["width_um"])
                w_item.setBackground(Qt.red)

            try:
                h_um = float(h_item.text())
            except ValueError:
                h_um = float(self.default_values["height_um"])
                h_item.setBackground(Qt.red)

            self.file_info[fname] = {
                "magnification": mag,
                "width_um": w_um,
                "height_um": h_um
            }

        # Check for errors and close the dialog
        if all(value.get("magnification") != Qt.red for value in self.file_info.values()):
            self.accept()

    def get_file_info(self):
        """
        Return the collected metadata for all files.
        """
        return self.file_info

class PlotWindow(QWidget):
    """
    A window to display a Matplotlib figure with an optional toolbar.
    """

    def __init__(self, figure, image_name, label, parent=None, show_toolbar=True):
        """
        Initialize the PlotWindow.

        :param figure: Matplotlib figure to display.
        :param image_name: Name of the image associated with the plot.
        :param label: Label or title for the plot window.
        :param parent: Parent widget (optional).
        :param show_toolbar: Whether to display the Matplotlib navigation toolbar.
        """
        super().__init__(parent)
        self.setWindowTitle(f"{label} - {image_name}")

        # Minimum size for the window
        self.setMinimumSize(500, 300)

        # Main layout
        layout = QVBoxLayout(self)

        # Matplotlib canvas
        self.canvas = FigureCanvas(figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        layout.addWidget(self.canvas)

        # Optional toolbar
        if show_toolbar:
            self.toolbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(self.toolbar)

        self.setLayout(layout)

class DataBrowser(QDialog):
    """
    A dialog to manage and display hierarchical data using QTreeWidget.
    """

    def __init__(self, parent=None):
        """
        Initialize the DataBrowser dialog.
        
        :param parent: Parent widget (optional).
        """
        super().__init__(parent)
        self.setWindowTitle("Data Browser")
        self.resize(300, 300)
        
        # Ensure the DataBrowser stays on top
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        # Main layout
        layout = QVBoxLayout(self)

        # Tree widget for displaying hierarchical data
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Image", "Plot Label"])  # Optional header for clarity
        layout.addWidget(self.tree)


        # Dictionary to store plot windows
        self.plot_windows = {}
        self.tree.itemChanged.connect(self.on_item_changed)

        self.setLayout(layout)

    def showEvent(self, event):
        """
        Ensure the dialog gets focus when shown.
        """
        super().showEvent(event)
        self.raise_()
        self.activateWindow()

    def add_plots_for_image(self, image_name, figures, label):
        """
        Add plots for a specific image under a top-level item in the tree.
    
        :param image_name: The name of the image (top-level item).
        :param figures: List of Matplotlib figures to associate with the image.
        :param label: Label describing the plot type.
        """
        top_item = self.find_top_item(image_name)
        if not top_item:
            top_item = QTreeWidgetItem([image_name, ""])
            top_item.setFlags(top_item.flags() | Qt.ItemIsUserCheckable)
            top_item.setCheckState(0, Qt.Checked)  # Default to checked
            self.tree.addTopLevelItem(top_item)
    
        for idx, fig in enumerate(figures):
            figure_name = f"{label} {idx+1}" if len(figures) > 1 else label
            child_item = QTreeWidgetItem([figure_name, "Plot"])
            child_item.setFlags(child_item.flags() | Qt.ItemIsUserCheckable)
            child_item.setCheckState(0, Qt.Checked)  # Default to checked
            top_item.addChild(child_item)
    
            # Create and store the PlotWindow instance
            pw = PlotWindow(fig, image_name, figure_name)
            key = (image_name, figure_name)
            self.plot_windows[key] = pw
    
            # Show the plot window if checked
            if child_item.checkState(0) == Qt.Checked:
                pw.show()
    
        # Expand the top-level item for better visibility
        self.tree.expandItem(top_item)


    def find_top_item(self, image_name):
        """
        Find the top-level item in the tree for a given image name.

        :param image_name: The name of the image to search for.
        :return: QTreeWidgetItem if found, None otherwise.
        """
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.text(0) == image_name:
                return item
        return None

    def on_item_changed(self, item, column):
        """
        Handle changes in the tree's items (e.g., checkbox toggling).
        
        :param item: The QTreeWidgetItem that was changed.
        :param column: The column that was changed (unused here).
        """
        parent = item.parent()
        if parent is None:
            # Top-level item changed
            image_name = item.text(0)
            checked = (item.checkState(0) == Qt.Checked)
            for c in range(item.childCount()):
                child = item.child(c)
                child.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
        else:
            # Child item changed
            image_name = parent.text(0)
            figure_label = item.text(0)
            checked = (item.checkState(0) == Qt.Checked)

            key = (image_name, figure_label)
            if key in self.plot_windows:
                if checked:
                    self.plot_windows[key].show()
                else:
                    self.plot_windows[key].hide()

class ResultsTableDialog(QDialog):
    """
    Displays the contents of a Pandas DataFrame in a QTableWidget.
    Provides a button to export the data as a CSV file.
    """

    def __init__(self, df_c, parent=None):
        """
        Initialize the ResultsTableDialog.
        
        :param df_c: Pandas DataFrame to display.
        :param parent: Parent widget (optional).
        """
        super().__init__(parent)
        self.setWindowTitle("Results Table")
        self.df_c = df_c  # Pandas DataFrame

        layout = QVBoxLayout(self)

        # Table widget to display data
        self.table = QTableWidget()
        layout.addWidget(self.table)

        # Buttons for export and close
        btn_layout = QHBoxLayout()
        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self.export_csv)
        btn_layout.addWidget(export_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.resize(600, 400)

        # Load data into the table
        self.load_df_into_table()

    def load_df_into_table(self):
        """
        Populate the QTableWidget with data from the DataFrame.
        """
        if self.df_c is None or self.df_c.empty:
            QMessageBox.warning(self, "No Data", "The provided DataFrame is empty.")
            return

        row_count = len(self.df_c.index)
        col_count = len(self.df_c.columns)
        self.table.setRowCount(row_count)
        self.table.setColumnCount(col_count)

        # Set column headers
        self.table.setHorizontalHeaderLabels(self.df_c.columns.tolist())

        # Populate table with DataFrame values
        for r in range(row_count):
            for c in range(col_count):
                value = str(self.df_c.iloc[r, c])
                item = QTableWidgetItem(value)
                self.table.setItem(r, c, item)

        # Enable sorting
        self.table.setSortingEnabled(True)

        # Adjust column widths
        self.table.resizeColumnsToContents()

    def export_csv(self):
        """
        Open a QFileDialog to save the DataFrame as a CSV file.
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            try:
                self.df_c.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", f"CSV saved successfully to {file_path}.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving CSV: {e}")

class ResizableRectItem(QGraphicsRectItem):
    """
    A resizable and movable rectangle with handles for resizing.
    """

    def __init__(self, rect, parent=None):
        """
        Initialize the resizable rectangle.

        :param rect: Initial QRectF for the rectangle.
        :param parent: Parent QGraphicsItem (optional).
        """
        super().__init__(rect, parent)
        self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsRectItem.ItemIsFocusable, True)
        self.setPen(Qt.red)
        self.setBrush(Qt.transparent)

        # Handle size and positions
        self.handle_size = 10
        self.handles = {}
        self.update_handles()

        self.resizing = False
        self.current_handle = None

    def update_handles(self):
        """
        Update the positions of the resize handles based on the rectangle's geometry.
        """
        rect = self.rect()
        self.handles = {
            "top_left": QRectF(rect.topLeft().x() - self.handle_size / 2,
                               rect.topLeft().y() - self.handle_size / 2,
                               self.handle_size, self.handle_size),
            "bottom_right": QRectF(rect.bottomRight().x() - self.handle_size / 2,
                                   rect.bottomRight().y() - self.handle_size / 2,
                                   self.handle_size, self.handle_size)
        }

    def boundingRect(self):
        """
        Include handles in the bounding rectangle.
        """
        return self.rect().adjusted(-self.handle_size, -self.handle_size, self.handle_size, self.handle_size)

    def paint(self, painter, option, widget):
        """
        Draw the rectangle and its resize handles.
        """
        super().paint(painter, option, widget)
        painter.setBrush(Qt.blue)
        for rect in self.handles.values():
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        """
        Detect if the mouse press occurred on a resize handle.
        """
        for handle, rect in self.handles.items():
            if rect.contains(event.pos()):
                self.resizing = True
                self.current_handle = handle
                break
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """
        Resize the rectangle or move it based on mouse movement.
        """
        if self.resizing:
            new_rect = self.rect()
            if self.current_handle == "top_left":
                new_rect.setTopLeft(event.scenePos())
            elif self.current_handle == "bottom_right":
                new_rect.setBottomRight(event.scenePos())

            # Ensure the rectangle does not collapse
            if new_rect.width() > self.handle_size and new_rect.height() > self.handle_size:
                self.setRect(new_rect)
                self.update_handles()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """
        End resizing on mouse release.
        """
        self.resizing = False
        self.current_handle = None
        super().mouseReleaseEvent(event)

class RegionSelectorForImageDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Region of Interest")
        self.image_path = image_path
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        layout.addWidget(self.view)

        pixmap = QPixmap(self.image_path)
        self.scene.addPixmap(pixmap)
        self.rect_item = ResizableRectItem(QRectF(50, 50, 100, 100))
        self.scene.addItem(self.rect_item)

        button_layout = QHBoxLayout()
        self.process_whole_btn = QPushButton("Process Whole Image")
        self.process_whole_btn.clicked.connect(self.process_whole_image)
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept_selection)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(self.process_whole_btn)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

    def process_whole_image(self):
        self.selection = None  # None means process the whole image
        self.accept()

    def accept_selection(self):
        rect = self.rect_item.rect()
        x, y, width, height = int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())
        image_width, image_height = self.scene.sceneRect().width(), self.scene.sceneRect().height()
    
        print(f"Selected ROI: x={x}, y={y}, width={width}, height={height}")
        print(f"Image dimensions: width={image_width}, height={image_height}")
    
        if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
            QMessageBox.warning(self, "Invalid ROI", "Selected ROI is out of bounds.")
            return
    
        self.selection = (x, y, width, height)
        self.accept()

    def get_selection(self):
        """
        Returns the selected ROI or a flag indicating whole-image processing.
        """
        if self.selection is None:
            return None, None, None, None, True
        x, y, width, height = self.selection
        return x, y, width, height, False




##############################################
# ProcessingThread
##############################################
class ProcessingThread(QThread):
    update_signal = pyqtSignal(str)
    figure_signal = pyqtSignal(object, str, str)
    progress_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)
    dfc_ready = pyqtSignal(object)

    def __init__(
        self, directory_path, output_path, notch_Filter, segment_width, coordinates,
        correction_factor, Filter, decimals, rotation_enabled, file_info
    ):
        super().__init__()
        self.directory_path = directory_path
        self.output_path = output_path
        self.notch_Filter = notch_Filter
        self.segment_width = segment_width
        self.coordinates = coordinates
        self.correction_factor = correction_factor
        self.Filter = Filter
        self.decimals = decimals
        self.rotation_enabled = rotation_enabled
        self.file_info = file_info
        
    def set_roi_data(self, roi_data):
        """
        Sets ROI data for each image file.
        """
        self.roi_data = roi_data

    def run(self):
        """
        Core method to process files in the selected directory with per-image ROI or whole-image processing.
        """
        try:
            # Step 1: Get list of all image files in the directory
            all_files = [
                f for f in os.listdir(self.directory_path)
                if f.lower().endswith((".jpg", ".png", ".tif"))
            ]
        except Exception as e:
            self.error_signal.emit(f"Could not read directory: {str(e)}")
            return
    
        total = len(all_files)
        if total == 0:
            self.update_signal.emit("No matching files found.")
            return
    
        self.update_signal.emit(f"Start processing directory: {self.directory_path}")
    
        # Step 2: Process each file
        for i, filename in enumerate(all_files, start=1):
            image_path = os.path.join(self.directory_path, filename)
            self.update_signal.emit(f"Processing file: {filename}")
    
            try:
                # 1) Load image and prepare DataFrame
                image_data, df_c = load_and_prepare_image(image_path, self.Filter)
                if image_data is None or df_c is None:
                    raise ValueError(f"Invalid data for {filename}")
    
                self.update_signal.emit(f"Loaded image: {filename} with shape {image_data.shape}")
                
                

    
                # 2) Handle ROI or process the entire image
                roi = self.roi_data.get(filename, None) if hasattr(self, 'roi_data') else None

                if roi and len(roi) == 4:
                    x, y, width, height = roi
                    roi = [x, x + width, y, y + height]
                    selected_region = image_data[y:y + height, x:x + width]
                    self.update_signal.emit(f"Using ROI for {filename}: {roi}")
                else:
                    roi = [0, image_data.shape[1], 0, image_data.shape[0]]
                    selected_region = image_data
                    self.update_signal.emit(f"No ROI for {filename}. Processing whole image.")

    
                # 3) Add user-defined metadata
                default_info = {"magnification": 1.0, "width_um": 100.0, "height_um": 100.0}
                info = self.file_info.get(filename, default_info)
                df_c["magnification"] = info["magnification"]
                df_c["width_um"] = info["width_um"]
                df_c["height_um"] = info["height_um"]
    
                # 4) Apply correction factor if provided
                if self.correction_factor:
                    df_c = apply_size_correction(df_c, self.correction_factor)
    
                # 5) Rotate image if rotation is enabled
                if self.rotation_enabled:
                    selected_region = rot_img(selected_region)
                    if selected_region is None or not hasattr(selected_region, 'shape'):
                        raise ValueError(f"Rotation failed for {filename}")
                    height_px, width_px = selected_region.shape
                    df_c["width_px"] = width_px
                    df_c["height_px"] = height_px
                    self.update_signal.emit(f"Rotated {filename} -> {width_px}x{height_px} px")
    
                # 6) Perform FFT and emit FFT figure
                fft_figure = plot_fft(selected_region, df_c)
                if fft_figure is not None:
                    self.figure_signal.emit(fft_figure, filename, "FFT")
                else:
                    self.error_signal.emit(f"FFT plot for {filename} is None.")
                
                # 7) Process based on segment width (1D or 2D)
                df, df_c, region_used = seg_proc(selected_region, df_c, self.segment_width, self.notch_Filter, roi)
                if self.segment_width == 1:
                    try:
                        gini_coefficient(df, df_c, 1)
                        if self.decimals is not None:
                            df, df_c = round_values(self.decimals, df, df_c)
                        plot_data_1d = {"df": df, "df_c": df_c, "image_data": selected_region, "region": region_used}
                        self.figure_signal.emit(plot_data_1d, filename, "1D Analysis")
                    except Exception as e:
                        self.error_signal.emit(f"Error during 1D Analysis for {filename}: {e}")
                else:
                    try:
                        gini_coefficient(df, df_c, 2)
                        if self.decimals is not None:
                            df, df_c = round_values(self.decimals, df, df_c)
                        plot_data_2d = {"df": df, "df_c": df_c, "image_data": selected_region, "region": region_used}
                        self.figure_signal.emit(plot_data_2d, filename, "2D Analysis")
                    except Exception as e:
                        self.error_signal.emit(f"Error during 2D Analysis for {filename}: {e}")
    
                # Save processed data
                extract_and_save_data(self.output_path, df_c, self.notch_Filter, self.segment_width, df)
                self.dfc_ready.emit(df_c)
    
                # Update progress
                self.update_signal.emit(f"Processed: {filename}")
                progress_value = int((i / total) * 100)
                self.progress_signal.emit(progress_value)
    
            except Exception as e:
                self.error_signal.emit(f"Error while processing {filename}: {str(e)}")
    
        self.update_signal.emit(f"Processing completed for directory: {self.directory_path}")



##############################################
# ImageProcessingApp
##############################################
class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.directory_path = None
        self.output_path = None
        self.notch_Filter = 0
        self.segment_width = 1
        self.coordinates = [None, None, None, None]
        self.correction_factor = None
        self.Filter = 1
        self.decimals = 3
        self.rotation_enabled = True

        self.file_info = {}
        self.data_browser = None
        self.df_c_latest = None

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab Input
        self.tab_input = QWidget()
        tab_input_layout = QVBoxLayout(self.tab_input)

        file_layout = QHBoxLayout()
        file_label = QLabel("Select Input Directory:")
        self.file_input = QLineEdit()
        file_button = QPushButton("Browse")
        file_button.clicked.connect(self.open_directory_dialog)
        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(file_button)

        output_layout = QHBoxLayout()
        output_label = QLabel("Select Output Directory:")
        self.output_input = QLineEdit()
        output_button = QPushButton("Browse")
        output_button.clicked.connect(self.open_output_directory_dialog)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_input)
        output_layout.addWidget(output_button)

        self.btn_image_info = QPushButton("Set Image Info per File")
        self.btn_image_info.clicked.connect(self.set_image_info_for_files)
        db_btn = QPushButton("Open Data Browser")
        db_btn.clicked.connect(self.open_data_browser)

        tab_input_layout.addLayout(file_layout)
        tab_input_layout.addLayout(output_layout)
        tab_input_layout.addWidget(self.btn_image_info)
        tab_input_layout.addWidget(db_btn)

        self.tab_input.setLayout(tab_input_layout)
        self.tabs.addTab(self.tab_input, "Input")

        # Tab Parameters
        self.tab_params = QWidget()
        tab_params_layout = QVBoxLayout(self.tab_params)

        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter:")
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["None", "Mean + Hanning", "Mean", "Hanning"])
        self.filter_combo.setCurrentIndex(self.Filter)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_combo)

        segment_layout = QHBoxLayout()
        segment_label = QLabel("Segment Width:")
        self.segment_spinbox = QSpinBox()
        self.segment_spinbox.setMinimum(1)
        self.segment_spinbox.setValue(self.segment_width)
        segment_layout.addWidget(segment_label)
        segment_layout.addWidget(self.segment_spinbox)

        options_layout = QHBoxLayout()
        correction_label = QLabel("Correction Factor:")
        self.correction_input = QLineEdit()
        decimal_label = QLabel("Decimals:")
        self.decimal_spinbox = QSpinBox()
        self.decimal_spinbox.setMinimum(0)
        self.decimal_spinbox.setValue(self.decimals)

        rotation_checkbox = QPushButton("Enable Rotation")
        rotation_checkbox.setCheckable(True)
        rotation_checkbox.setChecked(self.rotation_enabled)
        rotation_checkbox.toggled.connect(self.toggle_rotation)

        options_layout.addWidget(correction_label)
        options_layout.addWidget(self.correction_input)
        options_layout.addWidget(decimal_label)
        options_layout.addWidget(self.decimal_spinbox)
        options_layout.addWidget(rotation_checkbox)

        tab_params_layout.addLayout(filter_layout)
        tab_params_layout.addLayout(segment_layout)
        tab_params_layout.addLayout(options_layout)
        self.tab_params.setLayout(tab_params_layout)
        self.tabs.addTab(self.tab_params, "Parameters")

        # Action-Buttons
        run_layout = QHBoxLayout()
        run_button = QPushButton("Run Processing")
        run_button.clicked.connect(self.run_processing)
        run_layout.addStretch()
        run_layout.addWidget(run_button)

        self.show_table_button = QPushButton("Show Results Table")
        self.show_table_button.setEnabled(False)
        self.show_table_button.clicked.connect(self.show_results_table)
        run_layout.addWidget(self.show_table_button)


        main_layout.addLayout(run_layout)

        # Progress bar
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        main_layout.addWidget(self.progressBar)

        self.setLayout(main_layout)
        self.setWindowTitle("Regularity")
        self.resize(300, 300)



    def open_data_browser(self):
        """
        Opens the Data Browser dialog, initializing it if necessary.
        """
        if not self.data_browser:
            self.data_browser = DataBrowser(self)
        self.data_browser.show()

    def open_directory_dialog(self):
        """
        Opens a directory selection dialog for input directory.
        """
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.file_input.setText(dir_path)

    def open_output_directory_dialog(self):
        """
        Opens a directory selection dialog for output directory.
        """
        output_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if output_path:
            self.output_input.setText(output_path)

    def toggle_rotation(self, checked):
        """
        Toggles the rotation feature on or off based on user input.
        """
        self.rotation_enabled = checked

    def set_image_info_for_files(self):
        """
        Allows the user to set image information for each file in the input directory.
        """
        directory = self.file_input.text()
        if not os.path.isdir(directory):
            QMessageBox.warning(self, "Warning", "Please select a valid input directory first.")
            return

        all_files = [
            f for f in os.listdir(directory)
            if f.lower().endswith((".jpg", ".png", ".tif"))
        ]
        if not all_files:
            QMessageBox.information(self, "Info", "No matching image files found in directory.")
            return

        dlg = MultiImageInfoDialog(all_files, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.file_info = dlg.get_file_info()
            QMessageBox.information(self, "Info", "Image info saved for each file.")
        else:
            QMessageBox.information(self, "Info", "Canceled setting image info.")

    def run_processing(self):
        """
        Gathers ROI information for each image and starts the processing thread.
        """
        # Step 1: Validate directories
        self.directory_path = self.file_input.text()
        self.output_path = self.output_input.text()
        self.Filter = self.filter_combo.currentIndex()
        self.segment_width = self.segment_spinbox.value()
        cf_text = self.correction_input.text()
        self.correction_factor = float(cf_text) if cf_text else None
        self.decimals = self.decimal_spinbox.value()
    
        if not os.path.isdir(self.directory_path) or not os.path.isdir(self.output_path):
            QMessageBox.warning(self, "Error", "Please select valid input/output directories.")
            return
    
        # Step 2: Gather image files
        all_files = [
            f for f in os.listdir(self.directory_path)
            if f.lower().endswith((".jpg", ".png", ".tif"))
        ]
    
        if not all_files:
            QMessageBox.warning(self, "Warning", "No valid image files found in the directory.")
            return
    
        # Step 3: Open ROI selection dialog for each file
        self.roi_data = {}
        for file in all_files:
            image_path = os.path.join(self.directory_path, file)
            dialog = RegionSelectorForImageDialog(image_path, self)
            if dialog.exec_() == QDialog.Accepted:
                x, y, width, height, process_whole = dialog.get_selection()
                if process_whole:
                    self.roi_data[file] = None  # None means process the whole image
                else:
                    self.roi_data[file] = (x, y, width, height)
    
        # Step 4: Start processing thread with ROI data
        self.start_processing_thread()
    
    def start_processing_thread(self):
        """
        Starts the ProcessingThread using the gathered ROI data.
        """
        if not self.roi_data:
            QMessageBox.warning(self, "Error", "No images were selected for processing.")
            return
    
        # Initialize and start the thread
        self.thread = ProcessingThread(
            directory_path=self.directory_path,
            output_path=self.output_path,
            notch_Filter=self.notch_Filter,
            segment_width=self.segment_width,
            coordinates=None,  # Per-image coordinates handled within ProcessingThread
            correction_factor=self.correction_factor,
            Filter=self.Filter,
            decimals=self.decimals,
            rotation_enabled=self.rotation_enabled,
            file_info=self.file_info
        )
        self.thread.set_roi_data(self.roi_data)  # Pass ROI data to the thread
        self.thread.update_signal.connect(self.update_result_display)
        self.thread.figure_signal.connect(self.process_plot_data)
        self.thread.progress_signal.connect(self.progressBar.setValue)
        self.thread.error_signal.connect(self.show_error_message)
        self.thread.dfc_ready.connect(self.store_df_c)
    
        self.thread.start()


    def store_df_c(self, df_c):
        """
            Stores each DataFrame (df_c) instance in a list for later use.
        """
        if not hasattr(self, 'all_df_c'):
            self.all_df_c = []
        self.all_df_c.append(df_c.copy())  # Ensure a copy is saved
        self.show_table_button.setEnabled(True)


    def show_results_table(self):
        """
        Combines all collected DataFrames and displays them in a ResultsTableDialog.
        """
        if not hasattr(self, 'all_df_c') or not self.all_df_c:
            QMessageBox.information(self, "Info", "Keine Daten in df_c vorhanden.")
            return
    
        combined_df = pd.concat(self.all_df_c, ignore_index=True)
        dlg = ResultsTableDialog(combined_df, parent=self)
        dlg.exec_()


    def update_result_display(self, message):#
        """
        Displays status messages in the console or GUI log.
        """
        print(message)

    def show_figures_in_data_browser(self, figures, image_name, label):
        """
        Displays the provided figures in the Data Browser.
        """
        if not self.data_browser:
            self.data_browser = DataBrowser(self)
        self.data_browser.add_plots_for_image(image_name, figures, label)
        self.data_browser.show()
        
    def open_region_selector(self):
        """
        Opens a dialog to select a region of interest (ROI) from the first image in the directory.
        """
        if not self.directory_path:
            QMessageBox.warning(self, "Warning", "Please run the processing for the whole data first.")
            return
    
        image_files = [f for f in os.listdir(self.directory_path) if f.lower().endswith((".jpg", ".png", ".tif"))]
        if not image_files:
            QMessageBox.warning(self, "Warning", "No image files found in the directory.")
            return
    
        first_image_path = os.path.join(self.directory_path, image_files[0])
        dialog = RegionSelectorForImageDialog(first_image_path, self)
        dialog.region_selected.connect(self.process_selected_region)
        dialog.exec_()

    def process_plot_data(self, plot_data, image_name, label):
        """
        Receives data and creates the Matplotlib figures in the main thread.
        Labels the plots according to their content.
        """
        
        try:
            print(f"Received plot data for {image_name} with label: {label}")
            
            if label == "FFT":
                self.show_figures_in_data_browser([plot_data], image_name, "FFT")
            
            elif label == "1D Analysis":
                figs_1d = analyze_and_visualize_results(
                    plot_data["df"], plot_data["df_c"], plot_data["image_data"], plot_data["region"]
                )
                for fig, lbl in zip(figs_1d, ["Processed Image", "1D-Y-Period", "1D-Y-Phase/Δ-Phase"]):
                    self.show_figures_in_data_browser([fig], image_name, lbl)
            
            elif label == "2D Analysis":
                figs_2d = analyze_and_visualize_results_2d(
                    plot_data["df"], plot_data["df_c"], plot_data["image_data"], plot_data["region"])
                
                for fig, lbl in zip(figs_2d, ["Processed Image", "2D-X/Y-Period", "2D X-Phase/Δ-Phase", "2D Y-Phase/Δ-Phase"]):
                    self.show_figures_in_data_browser([fig], image_name, lbl)
            else:
                print(f"Unknown label: {label}")
        
        except Exception as e:
            print(f"Error in process_plot_data for {image_name}: {e}")
            self.show_error_message(f"Error processing plot data: {e}")

    def process_selected_region(self, x, y, width, height):
        """
        Stores the selected region and displays a confirmation message.
        """
        # Speichere die ausgewählten Koordinaten
        self.coordinates = [x, y, width, height]
        print(f"Selected region: x={x}, y={y}, width={width}, height={height}")
        
        QMessageBox.information(
            self,
            "Region Selected",
            f"Selected Region:\n"
            f"x = {x}\n"
            f"y = {y}\n"
            f"width = {width}\n"
            f"height = {height}"
        )

    def rerun_analysis_with_new_region(self):
        """
        Reruns the analysis using the newly selected region of interest.
        """
        if not self.coordinates or not all(c is not None for c in self.coordinates):
            QMessageBox.warning(
                self,
                "Error",
                "Please select a valid region before rerunning the analysis."
            )
            return
    
        if not self.directory_path or not self.output_path:
            QMessageBox.warning(
                self,
                "Error",
                "Please ensure that both input and output directories are set."
            )
            return
    
        print("Rerunning analysis with selected region...")
        self.run_processing()  # Startet die Verarbeitung erneut

    def process_all_images(self):
        """
        Starts the ProcessingThread using the ROI data or whole-image options collected for all files.
        """
        if not self.roi_data:
            QMessageBox.warning(self, "Error", "No images were selected for processing.")
            return
    
        self.thread = ProcessingThread(
            directory_path=self.directory_path,
            output_path=self.output_path,
            notch_Filter=self.notch_Filter,
            segment_width=self.segment_width,
            coordinates=None,  # Per-image coordinates are handled within ProcessingThread
            correction_factor=self.correction_factor,
            Filter=self.Filter,
            decimals=self.decimals,
            rotation_enabled=self.rotation_enabled,
            file_info=self.file_info
        )
        self.thread.set_roi_data(self.roi_data)  # Pass ROI data to the thread
        self.thread.update_signal.connect(self.update_result_display)
        self.thread.figure_signal.connect(self.process_plot_data)
        self.thread.progress_signal.connect(self.progressBar.setValue)
        self.thread.error_signal.connect(self.show_error_message)
        self.thread.dfc_ready.connect(self.store_df_c)
    
        self.thread.start()


    def show_error_message(self, error_str):
        """
        Displays an error message dialog. 
        """
        QMessageBox.critical(self, "Error", error_str)


def main():
    try:
        # Step 1: Create a QApplication instance
        app = QApplication(sys.argv)

        # Step 2: Set up optional logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger("ImageProcessingApp")
        logger.info("Application started.")

        # Step 3: Set up ConsoleOutputDialog for logging console output
        console_output = MultiOutput()
        console_dialog = ConsoleOutputDialog()
        console_output.new_text.connect(console_dialog.append_text)
        console_dialog.show()

        # Redirect stdout and stderr to the console dialog
        sys.stdout = console_output
        #sys.stderr = console_output
        
        
        # # Step 4: Create and show the splash screen
        # splash_pix = QPixmap(r"C:\Users\Eric\Desktop\GUI 1_0\icon.bmp")  # Replace with your splash image path
        # splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
        # splash.setMask(splash_pix.mask())
        # splash.showMessage("...ReGuLaRiTy...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
        # splash.show()

        # # Perform any startup tasks
        # QTimer.singleShot(1000, lambda: splash.showMessage("Initializing UI...", Qt.AlignBottom | Qt.AlignCenter, Qt.white))
        # QTimer.singleShot(2000, lambda: splash.showMessage("Loading resources...", Qt.AlignBottom | Qt.AlignCenter, Qt.white))
        # QTimer.singleShot(3000, lambda: splash.showMessage("Starting application...", Qt.AlignBottom | Qt.AlignCenter, Qt.white))

        # Step 5: Create and display the main application window
        ex = ImageProcessingApp()
        #splash.finish(ex)
        ex.show()

        # Step 6: Start the Qt event loop
        logger.info("Launching application UI.")
        print("Regularity - Developed by Eric Rahner")
        sys.exit(app.exec_())
    
    except Exception as e:
        # Log critical errors and display an error message to the user
        logger = logging.getLogger("ImageProcessingApp")
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        QMessageBox.critical(None, "Critical Error", f"An error occurred: {e}")

    finally:
        # Restore stdout and stderr to their original states
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        logger.info("Application terminated.")

if __name__ == "__main__":
    main()
