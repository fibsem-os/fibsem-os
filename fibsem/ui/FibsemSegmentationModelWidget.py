
import napari
import napari.utils.notifications
from fibsem.ui.qt.threading import thread_worker
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont

from fibsem.detection.detection import DetectedFeatures
from fibsem.segmentation.model import SegmentationModel, load_model
from fibsem.ui import stylesheets
from fibsem.ui.utils import WheelBlocker

CHECKPOINT_PATH = "autolamella-mega-20240107.pt"
SEGMENT_ANYTHING_AVAIABLE = False
try:
    SEGMENT_ANYTHING_AVAIABLE = True
    from fibsem.segmentation.sam_model import SamModelWrapper
except ImportError:
    pass


AVAILABLE_MODELS = ["SegmentationModel"]

if SEGMENT_ANYTHING_AVAIABLE:
    AVAILABLE_MODELS.append("SegmentAnythingModel")
RECOMMENDED_SAM_CHECKPOINTS = ["facebook/sam-vit-base", "Zigeng/SlimSAM-uniform-50"]

class FibsemSegmentationModelWidget(QtWidgets.QDialog):
    continue_signal = pyqtSignal(DetectedFeatures)
    model_loaded = pyqtSignal()

    def __init__(
        self,
        model: SegmentationModel = None,
        parent=None,
    ):
        super().__init__(parent=parent)
        self._setup_ui()

        self.model = model
        self.model_type = None
        self.setup_connections()

    def _setup_ui(self):
        """Hand-built replacement for the former Qt Designer form."""
        self.wheel_blocker = WheelBlocker()
        layout = QtWidgets.QGridLayout(self)

        self.label_header_model = QtWidgets.QLabel("Segmentation Model")
        header_font = QFont()
        header_font.setBold(True)
        header_font.setWeight(75)
        self.label_header_model.setFont(header_font)
        layout.addWidget(self.label_header_model, 0, 0, 1, 2)

        self.label_model_type = QtWidgets.QLabel("Model")
        self.comboBox_model_type = QtWidgets.QComboBox()
        layout.addWidget(self.label_model_type, 1, 0)
        layout.addWidget(self.comboBox_model_type, 1, 1)

        self.label_checkpoint = QtWidgets.QLabel("Checkpoint")
        self.lineEdit_checkpoint = QtWidgets.QLineEdit()
        self.checkpoint_seg_button = QtWidgets.QToolButton()
        self.checkpoint_seg_button.setText("...")
        layout.addWidget(self.label_checkpoint, 2, 0)
        layout.addWidget(self.lineEdit_checkpoint, 2, 1)
        layout.addWidget(self.checkpoint_seg_button, 2, 2)

        self.pushButton_load_model = QtWidgets.QPushButton("Load Model")
        layout.addWidget(self.pushButton_load_model, 3, 0, 1, 2)

        spacer = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        layout.addItem(spacer, 4, 0, 1, 2)

        # block accidental scroll-to-change on the combobox
        self.comboBox_model_type.installEventFilter(self.wheel_blocker)

    def setup_connections(self):

        # model
        self.pushButton_load_model.clicked.connect(self.load_model)
        self.lineEdit_checkpoint.setText(CHECKPOINT_PATH)
        self.comboBox_model_type.addItems(AVAILABLE_MODELS)
        self.comboBox_model_type.currentIndexChanged.connect(self.update_model_type)

    def update_model_type(self):

        model_type = self.comboBox_model_type.currentText()

        if model_type == "SegmentationModel":
            self.lineEdit_checkpoint.setText(CHECKPOINT_PATH)
            self.lineEdit_checkpoint.setToolTip("Please use a pretrained model.")
            self.checkpoint_seg_button.setEnabled(True)
        elif model_type == "SegmentAnythingModel":

            self.lineEdit_checkpoint.setText(RECOMMENDED_SAM_CHECKPOINTS[0])
            self.lineEdit_checkpoint.setToolTip(f"""Any supported transformer SAM model from HuggingFace can be used. 
                                                \nRecommended: 
                                                \nLarge GPU: {RECOMMENDED_SAM_CHECKPOINTS[0]}
                                                \nSmall GPU / CPU: {RECOMMENDED_SAM_CHECKPOINTS[1]}""")
            self.checkpoint_seg_button.setEnabled(False)

    # thread this, as it can take a long time to download the models
    def load_model(self) -> SegmentationModel:

        model_type = self.comboBox_model_type.currentText()
        checkpoint = self.lineEdit_checkpoint.text()

        self.pushButton_load_model.setEnabled(False)
        self.pushButton_load_model.setText("Loading...")
        self.pushButton_load_model.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
        self.pushButton_load_model.setToolTip("Downloading model... check terminal for progress...")

        worker = self.load_model_worker(model_type=model_type, checkpoint=checkpoint)
        worker.finished.connect(self.load_model_finished)
        worker.start()


    @thread_worker
    def load_model_worker(self, model_type: str, checkpoint: str):
        
        if model_type == "SegmentationModel":
            self.model = load_model(checkpoint=checkpoint)

        if model_type == "SegmentAnythingModel":
            self.model = SamModelWrapper(checkpoint=checkpoint)

        self.model_type = model_type
        self.model.checkpoint = checkpoint

        return self.model

    def load_model_finished(self):

        self.pushButton_load_model.setEnabled(True)
        self.pushButton_load_model.setText("Load Model")
        self.pushButton_load_model.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
        self.pushButton_load_model.setToolTip("")

        if self.model is not None:
            print(f"Loaded: {self.model}, {self.model.device}, {self.model.checkpoint}")
            self.model_loaded.emit()

def main():

    viewer  = napari.Viewer()
    widget = FibsemSegmentationModelWidget()
    viewer.window.add_dock_widget(widget, 
                                  area="right", 
                                  name="Fibsem Segmentation Model")

    napari.run()
    


if __name__ == "__main__":
    main()
