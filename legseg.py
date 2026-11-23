import glob
import os
import shutil
import sys
import time
from dataclasses import dataclass, field

import imageio.v3 as imageio
import numpy as np
import pydicom
import torch
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QTreeView,
    QVBoxLayout,
    QWidget,
)
from PyQt6.uic import loadUi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# nnU-Net reads these at import time; only nnUNet_results is actually used.
os.environ["nnUNet_raw"] = os.path.join(BASE_DIR, "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = os.path.join(BASE_DIR, "nnUNet_preprocessed")
os.environ["nnUNet_results"] = os.path.join(BASE_DIR, "nnUNet_results")

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from model_registry import (
    build_colormap,
    discover_models,
    get_label_names,
    resolve_default_model,
)
from seg_io import export_case_images, get_reader_writer_cls, read_segmentation_array


class ProcessingDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Processing")
        self.setGeometry(500, 300, 200, 100)
        layout = QVBoxLayout()
        label = QLabel("Processing images... Please wait.")
        layout.addWidget(label)
        self.setLayout(layout)
        self.setModal(True)


@dataclass
class PatientRecord:
    """One patient's cached state: raw DICOM data plus any prediction already run for
    it, so switching the patient slider back to an already-processed patient doesn't
    require re-running prediction."""
    name: str
    scratch_index: int
    dicom_files1: list = field(default_factory=list)
    dicom_files2: list = field(default_factory=list)
    fat: list = field(default_factory=list)
    muscle: list = field(default_factory=list)
    rows: int = 0
    cols: int = 0
    slices: int = 0
    patient_info1: list = field(default_factory=list)
    patient_info2: list = field(default_factory=list)
    pixel_spacing1: tuple = None
    pixel_spacing2: tuple = None
    slice_spacing1: float = None
    slice_spacing2: float = None
    prediction_files: list = field(default_factory=list)
    segmentation_volume: object = None
    start_slice_used: int = 0
    end_slice_used: int = 0
    start_slice_value: int = 1
    end_slice_value: int = 1
    current_slice_value: int = 1


class PatientAssignmentDialog(QDialog):
    """Lets the user confirm/correct which auto-detected subfolder is Fat/Water for
    each patient selected via 'Load Patients...', since the sequence-folder layout
    varies between datasets and can't be trusted blindly."""

    BROWSE_TEXT = "Browse..."

    def __init__(self, parent, patients_info):
        super().__init__(parent)
        self.setWindowTitle("Confirm Fat/Water Folders")
        self.setGeometry(400, 200, 720, 420)

        outer_layout = QVBoxLayout()
        title = QLabel("<h3>Confirm Fat/Water sequence folders per patient:</h3>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer_layout.addWidget(title)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        grid = QGridLayout()
        scroll_widget.setLayout(grid)
        scroll_area.setWidget(scroll_widget)

        grid.addWidget(QLabel("<b>Patient</b>"), 0, 0)
        grid.addWidget(QLabel("<b>Fat</b>"), 0, 1)
        grid.addWidget(QLabel("<b>Water</b>"), 0, 2)

        self.rows = []
        for row, patient in enumerate(patients_info, start=1):
            grid.addWidget(QLabel(patient["name"]), row, 0)

            fat_combo = QComboBox()
            water_combo = QComboBox()
            options = [""] + patient["candidates"] + [self.BROWSE_TEXT]
            for combo, guess in ((fat_combo, patient["fat_guess"]), (water_combo, patient["water_guess"])):
                combo.addItems(options)
                if guess and guess in patient["candidates"]:
                    combo.setCurrentText(guess)
                combo.activated.connect(lambda _i, c=combo, p=patient: self._handle_browse(c, p))

            grid.addWidget(fat_combo, row, 1)
            grid.addWidget(water_combo, row, 2)

            self.rows.append({"patient": patient, "fat_combo": fat_combo, "water_combo": water_combo})

        outer_layout.addWidget(scroll_area)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        outer_layout.addLayout(button_layout)

        self.setLayout(outer_layout)

    def _handle_browse(self, combo, patient):
        if combo.currentText() != self.BROWSE_TEXT:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", patient["folder"])
        if folder:
            rel = os.path.relpath(folder, patient["folder"])
            if combo.findText(rel) < 0:
                combo.insertItem(combo.count() - 1, rel)
            combo.setCurrentText(rel)
        else:
            combo.setCurrentIndex(0)

    def get_assignments(self):
        """{patient_name: (fat_rel_path_or_None, water_rel_path_or_None)}."""
        assignments = {}
        for row in self.rows:
            fat = row["fat_combo"].currentText()
            water = row["water_combo"].currentText()
            fat = fat if fat and fat != self.BROWSE_TEXT else None
            water = water if water and water != self.BROWSE_TEXT else None
            assignments[row["patient"]["name"]] = (fat, water)
        return assignments


class LegSegWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        loadUi(os.path.join(BASE_DIR, "legseg.ui"), self)
        self.setWindowIcon(QIcon(os.path.join(BASE_DIR, "assets", "logo.ico")))
        self.setWindowTitle("LegSeg")

        self.dicom_files1 = []
        self.dicom_files2 = []
        self.prediction_files = []
        self.fat = []
        self.muscle = []
        self.current_index = 0
        self.start_slice_used = 0
        self.end_slice_used = 0
        self.rows = 0
        self.cols = 0
        self.slices = 0
        self.patient_info1 = []
        self.patient_info2 = []
        self.pixel_spacing1 = None
        self.pixel_spacing2 = None
        self.slice_spacing1 = None
        self.slice_spacing2 = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Multi-patient state: self.patients caches every loaded patient's data/results,
        # while the bare attributes above always mirror the *currently active* patient
        # (self.patients[self.current_patient_index]).
        self.patients = []
        self.current_patient_index = -1
        self._next_scratch_index = 0

        self.available_models = discover_models(os.environ["nnUNet_results"])
        self.selected_model = resolve_default_model(self.available_models)
        if self.selected_model is None:
            QMessageBox.warning(self, "No Model Found",
                "No trained nnU-Net model was found in the nnUNet_results folder.\n"
                "Prediction will be unavailable until a model is installed and selected via Run > Select Model...")

        self.current_palette_name = "Default"
        self.current_colormap = {}
        self.legend_swatches = {}
        self.stat_labels = {}
        self.is_3d_model = False
        self.reader_writer = None
        self.segmentation_volume = None
        self.rebuild_labels_ui()

        self._clear_scratch_trees()

        self.slice_slider.valueChanged.connect(self.update_display)
        self.fat_slider.valueChanged.connect(lambda: self.update_overlay(1))
        self.water_slider.valueChanged.connect(lambda: self.update_overlay(2))
        self.patient_slider.valueChanged.connect(self.switch_patient)
        self.load_patients_button.clicked.connect(self.load_patients)
        self.swap_button.clicked.connect(self.swap_datasets)
        self.run_button.clicked.connect(self.run_prediction)
        self.actionShowStatistics.triggered.connect(self.show_statistics)
        self.save_button.clicked.connect(self.save_segmentations)
        self.actionReset.triggered.connect(self.reset)
        self.actionLoadPatients.triggered.connect(self.load_patients)
        self.actionLoadSeg.triggered.connect(self.load_segmentation)
        self.actionExit.triggered.connect(self.close)
        self.actionSwap.triggered.connect(self.swap_datasets)
        self.actionSaveSeg.triggered.connect(self.save_segmentations)
        self.actionAbout.triggered.connect(self.about)
        self.actionDevice.triggered.connect(self.set_device)
        self.actionSelectModel.triggered.connect(self.select_model)
        self.actionDataset.triggered.connect(self.show_dataset_info)
        self.actionDefault.triggered.connect(lambda: self.set_colormap("Default"))
        self.actionContrast.triggered.connect(lambda: self.set_colormap("Contrast"))
        self.actionGrayscale.triggered.connect(lambda: self.set_colormap("Grayscale"))
        self.actionViridis.triggered.connect(lambda: self.set_colormap("Viridis"))
        self.actionPaired.triggered.connect(lambda: self.set_colormap("Paired"))
        self.actionDark2.triggered.connect(lambda: self.set_colormap("Dark2"))

        self.fat_image.installEventFilter(self)
        self.water_image.installEventFilter(self)
        self.patient_slider.installEventFilter(self)
        self.slice_slider.installEventFilter(self)

    def _compute_slice_spacing(self, dicom_files):
        """Physical spacing between slices, from consecutive ImagePositionPatient z-values."""
        if len(dicom_files) < 2:
            return None
        try:
            first = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
            second = pydicom.dcmread(dicom_files[1], stop_before_pixels=True)
            return abs(float(second.ImagePositionPatient[2]) - float(first.ImagePositionPatient[2]))
        except (AttributeError, IndexError, TypeError):
            return None

    def _clear_scratch_trees(self):
        """Wipe the whole imagesTs/predictionTs scratch trees (all patients' subfolders)."""
        for folder_name in ("imagesTs", "predictionTs"):
            folder = os.path.join(BASE_DIR, folder_name)
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder, exist_ok=True)

    def _patient_scratch_dirs(self, scratch_index=None):
        """(images_dir, prediction_dir) scratch subfolders for the given patient (the
        active one by default), so different patients' exports/predictions never
        collide and persist on disk across patient switches."""
        if scratch_index is None:
            scratch_index = self.patients[self.current_patient_index].scratch_index
        images_dir = os.path.join(BASE_DIR, "imagesTs", f"patient_{scratch_index:03d}")
        prediction_dir = os.path.join(BASE_DIR, "predictionTs", f"patient_{scratch_index:03d}")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(prediction_dir, exist_ok=True)
        return images_dir, prediction_dir

    def _load_dicom_series(self, folder_path):
        """Load every DICOM/IMA file directly inside folder_path. Returns
        (dicom_files, pixel_arrays, patient_info, pixel_spacing, slice_spacing, rows, cols),
        or None if the folder has no DICOM files."""
        dicom_files = sorted(
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith((".dcm", ".ima"))
        )
        if not dicom_files:
            return None

        pixel_arrays = []
        dicom_data = None
        for file in dicom_files:
            dicom_data = pydicom.dcmread(file)
            pixel_arrays.append(dicom_data.pixel_array.astype(np.float32))

        rows, cols = pixel_arrays[0].shape[0], pixel_arrays[0].shape[1]

        patient_info = [dicom_data.PatientName if 'PatientName' in dicom_data else "Unknown",
                    dicom_data.PatientID if 'PatientID' in dicom_data else "Unknown",
                    dicom_data.StudyDate if 'StudyDate' in dicom_data else "Unknown",
                    dicom_data.StudyDescription if 'StudyDescription' in dicom_data else "Unknown",
                    dicom_data.SeriesDescription if 'SeriesDescription' in dicom_data else "Unknown"]

        # Physical spacing is needed to export a correctly scaled volume for 3D models.
        pixel_spacing = tuple(float(v) for v in getattr(dicom_data, "PixelSpacing", [1.0, 1.0]))
        slice_spacing = self._compute_slice_spacing(dicom_files) or float(getattr(dicom_data, "SliceThickness", 1.0))

        return dicom_files, pixel_arrays, patient_info, pixel_spacing, slice_spacing, rows, cols

    def _sync_patient_slider_range(self):
        """Keep patient_slider/patient_spinbox's range and value in sync with the
        patient list, without re-triggering switch_patient."""
        count = max(len(self.patients), 1)
        value = self.current_patient_index + 1 if self.current_patient_index >= 0 else 1
        for widget in (self.patient_spinbox, self.patient_slider):
            widget.blockSignals(True)
            widget.setMaximum(count)
            widget.setValue(value)
            widget.blockSignals(False)
        self.patient_slider.setEnabled(len(self.patients) > 1)

    def _save_active_into_record(self, record):
        """Snapshot the active attributes (currently displayed patient) into record."""
        record.dicom_files1 = self.dicom_files1
        record.dicom_files2 = self.dicom_files2
        record.fat = self.fat
        record.muscle = self.muscle
        record.rows = self.rows
        record.cols = self.cols
        record.slices = self.slices
        record.patient_info1 = self.patient_info1
        record.patient_info2 = self.patient_info2
        record.pixel_spacing1 = self.pixel_spacing1
        record.pixel_spacing2 = self.pixel_spacing2
        record.slice_spacing1 = self.slice_spacing1
        record.slice_spacing2 = self.slice_spacing2
        record.prediction_files = self.prediction_files
        record.segmentation_volume = self.segmentation_volume
        record.start_slice_used = self.start_slice_used
        record.end_slice_used = self.end_slice_used
        record.start_slice_value = self.start_slice.value()
        record.end_slice_value = self.end_slice.value()
        record.current_slice_value = self.slice_slider.value()

    def _load_record_into_active(self, record):
        """Restore record into the active attributes and refresh the dependent UI."""
        self.dicom_files1 = record.dicom_files1
        self.dicom_files2 = record.dicom_files2
        self.fat = record.fat
        self.muscle = record.muscle
        self.rows = record.rows
        self.cols = record.cols
        self.slices = record.slices
        self.patient_info1 = record.patient_info1
        self.patient_info2 = record.patient_info2
        self.pixel_spacing1 = record.pixel_spacing1
        self.pixel_spacing2 = record.pixel_spacing2
        self.slice_spacing1 = record.slice_spacing1
        self.slice_spacing2 = record.slice_spacing2
        self.prediction_files = record.prediction_files
        self.segmentation_volume = record.segmentation_volume
        self.start_slice_used = record.start_slice_used
        self.end_slice_used = record.end_slice_used

        max_slice = max(record.slices, 1)
        self.slice_slider.setMaximum(max_slice)
        self.start_slice.setMaximum(max_slice)
        self.end_slice.setMaximum(max_slice)
        self.start_slice.setValue(min(record.start_slice_value or 1, max_slice))
        self.end_slice.setValue(min(record.end_slice_value or max_slice, max_slice))
        self.slice_slider.setValue(min(record.current_slice_value or 1, max_slice))

        self.patient_name_label.setText(record.name)
        self._sync_patient_slider_range()

        has_data = bool(record.dicom_files1 and record.dicom_files2)
        has_prediction = bool(record.prediction_files)
        self.run_button.setEnabled(has_data and self.selected_model is not None)
        self.actionShowStatistics.setEnabled(has_prediction)
        self.save_button.setEnabled(has_prediction)
        self.actionSaveSeg.setEnabled(has_prediction)

    def switch_patient(self, value):
        """Slot for patient_slider/patient_spinbox (1-based value)."""
        new_index = value - 1
        if new_index == self.current_patient_index or not (0 <= new_index < len(self.patients)):
            return
        if self.current_patient_index >= 0:
            self._save_active_into_record(self.patients[self.current_patient_index])
        self.current_patient_index = new_index
        self._load_record_into_active(self.patients[new_index])
        self.update_display()

    def _select_multiple_directories(self, title):
        """A directory picker that allows selecting several folders at once (PyQt's
        native directory dialog only allows one, so this uses a non-native dialog with
        extended selection enabled on its file-list views)."""
        dialog = QFileDialog(self, title)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        for view in dialog.findChildren((QListView, QTreeView)):
            view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.selectedFiles()
        return []

    def _discover_dicom_subfolders(self, parent_folder):
        """Every subfolder (at any depth) under parent_folder that directly contains at
        least one DICOM/IMA file, as a path relative to parent_folder."""
        candidates = []
        for dirpath, _dirnames, filenames in os.walk(parent_folder):
            if any(f.lower().endswith((".dcm", ".ima")) for f in filenames):
                candidates.append(os.path.relpath(dirpath, parent_folder))
        return sorted(candidates)

    def _guess_role(self, parent_folder, rel_path):
        """Best-effort guess of 'fat'/'water'/None for a candidate sequence folder,
        from its path and (as a fallback) its first DICOM file's SeriesDescription.
        Only used to pre-fill the confirm/correct dialog - never trusted blindly."""
        lowered = rel_path.lower()
        # Exact "F"/"W" path components (not a substring match - too many folder
        # names incidentally contain the letter f or w) are also treated as Fat/Water.
        components = lowered.replace("\\", "/").split("/")
        if "fat" in lowered or "dual_f" in lowered or "f" in components:
            return "fat"
        if "water" in lowered or "dual_w" in lowered or "w" in components:
            return "water"

        full_path = os.path.join(parent_folder, rel_path)
        dicom_names = sorted(f for f in os.listdir(full_path) if f.lower().endswith((".dcm", ".ima")))
        if not dicom_names:
            return None
        try:
            dicom_data = pydicom.dcmread(os.path.join(full_path, dicom_names[0]), stop_before_pixels=True)
            description = str(getattr(dicom_data, "SeriesDescription", "")).lower()
        except Exception:
            return None
        if "fat" in description:
            return "fat"
        if "water" in description:
            return "water"
        return None

    def load_patients(self):
        """Multi-select several patient parent folders, auto-detect their Fat/Water
        sequence subfolders, let the user confirm/correct the detection, then load
        every confirmed patient (cached independently, switchable via patient_slider)."""
        parent_folders = self._select_multiple_directories("Select Patient Folders")
        if not parent_folders:
            return

        patients_info = []
        for parent_folder in parent_folders:
            candidates = self._discover_dicom_subfolders(parent_folder)
            if not candidates:
                continue
            fat_guess = None
            water_guess = None
            for rel_path in candidates:
                role = self._guess_role(parent_folder, rel_path)
                if role == "fat" and fat_guess is None:
                    fat_guess = rel_path
                elif role == "water" and water_guess is None:
                    water_guess = rel_path
            patients_info.append({
                "name": os.path.basename(os.path.normpath(parent_folder)),
                "folder": parent_folder,
                "candidates": candidates,
                "fat_guess": fat_guess,
                "water_guess": water_guess,
            })

        if not patients_info:
            QMessageBox.warning(self, "No DICOM Data Found",
                "No DICOM (.dcm/.ima) files were found under the selected folder(s).")
            return

        dialog = PatientAssignmentDialog(self, patients_info)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        assignments = dialog.get_assignments()

        loaded_names = []
        skipped_names = []
        first_new_index = None

        for patient in patients_info:
            fat_rel, water_rel = assignments[patient["name"]]
            if not fat_rel or not water_rel or fat_rel == water_rel:
                skipped_names.append(patient["name"])
                continue

            fat_loaded = self._load_dicom_series(os.path.join(patient["folder"], fat_rel))
            water_loaded = self._load_dicom_series(os.path.join(patient["folder"], water_rel))
            if not fat_loaded or not water_loaded:
                skipped_names.append(patient["name"])
                continue

            record = PatientRecord(name=patient["name"], scratch_index=self._next_scratch_index)
            self._next_scratch_index += 1
            (record.dicom_files1, record.fat, record.patient_info1,
             record.pixel_spacing1, record.slice_spacing1, rows1, cols1) = fat_loaded
            (record.dicom_files2, record.muscle, record.patient_info2,
             record.pixel_spacing2, record.slice_spacing2, _rows2, _cols2) = water_loaded

            record.rows, record.cols = rows1, cols1
            record.slices = min(len(record.dicom_files1), len(record.dicom_files2))
            record.start_slice_value = 1
            record.end_slice_value = record.slices
            record.current_slice_value = 1

            if first_new_index is None:
                first_new_index = len(self.patients)
            self.patients.append(record)
            loaded_names.append(patient["name"])

        if not loaded_names:
            QMessageBox.warning(self, "No Patients Loaded",
                "No patient had both a Fat and a Water folder assigned.\nSkipped: " + ", ".join(skipped_names))
            return

        if self.current_patient_index >= 0:
            self._save_active_into_record(self.patients[self.current_patient_index])

        self.current_patient_index = first_new_index
        self._load_record_into_active(self.patients[self.current_patient_index])
        self.update_display()

        summary = f"Loaded {len(loaded_names)} patient(s)."
        if skipped_names:
            summary += f"\nSkipped (missing Fat/Water assignment): {', '.join(skipped_names)}"
        QMessageBox.information(self, "Load Patients Complete", summary)

    def get_segmentation_for_slice(self, index):
        """The 2D segmentation array for a given slice, regardless of the selected model's format."""
        if self.selected_model is None or self.current_patient_index == -1:
            return None

        if self.is_3d_model:
            if self.segmentation_volume is None:
                return None
            # The exported volume only covers the selected slice range, not the
            # whole loaded series, so re-base the absolute slice index onto it.
            volume_index = index - (self.start_slice_used - 1)
            if volume_index < 0 or volume_index >= self.segmentation_volume.shape[0]:
                return None
            return self.segmentation_volume[volume_index]

        _, prediction_dir = self._patient_scratch_dirs()
        prediction_filename = os.path.join(prediction_dir, f"dataset_{index + 1:04d}{self.selected_model.file_ending}")
        if not os.path.exists(prediction_filename):
            return None
        return read_segmentation_array(self.reader_writer, prediction_filename)[0]

    def update_display(self):
        """Show the current Fat/Water slice, its segmentation overlay and per-label statistics."""
        self.current_index = self.slice_slider.value() - 1

        self.display_dicom(self.fat_image, self.fat_title, self.dicom_files1, self.fat, self.current_index, "Fat")
        self.display_dicom(self.water_image, self.water_title, self.dicom_files2, self.muscle, self.current_index, "Water")

        segmentation = self.get_segmentation_for_slice(self.current_index)

        stats = {}
        if segmentation is not None and self.selected_model is not None:
            total = np.sum(segmentation > 0)
            if total > 0:
                for label_id in get_label_names(self.selected_model.labels):
                    stats[label_id] = np.sum(segmentation == label_id) / total

        self.update_statistics(stats)

        if self.prediction_files:
            self.update_overlay(1)
            self.update_overlay(2)

    def display_dicom(self, view: QGraphicsView, title_label: QLabel, dicom_files, images, index, title):
        """Display the DICOM image at the given index on the given axis."""
        scene = QGraphicsScene()

        if not dicom_files:
            view.setScene(scene)
            view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return

        pixel_array = self._normalize_to_uint8(images[index])

        qimg = QImage(pixel_array.data, self.cols, self.rows, pixel_array.strides[0], QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)

        pixmap_item = QGraphicsPixmapItem(pixmap)
        scene.addItem(pixmap_item)
        view.setScene(scene)
        view.fitInView(pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

        self.set_title(title_label, title, index + 1, len(dicom_files))

    @staticmethod
    def _normalize_to_uint8(pixel_array):
        low, high = np.min(pixel_array), np.max(pixel_array)
        if high == low:
            return np.zeros(pixel_array.shape, dtype=np.uint8)
        return ((pixel_array - low) / (high - low) * 255).astype(np.uint8)

    def resizeEvent(self, event):
        """Handle window resize events to adjust the view."""
        super().resizeEvent(event)
        for view in [self.fat_image, self.water_image]:
            if view.scene():
                view.fitInView(view.scene().itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

        if self.prediction_files:
            self.update_overlay(1)
            self.update_overlay(2)

    def update_overlay(self, dataset_num):
        if dataset_num == 1:
            view = self.fat_image
            slider = self.fat_slider
            images = self.fat
            title = "Fat"
            label = self.fat_title
        else:
            view = self.water_image
            slider = self.water_slider
            images = self.muscle
            title = "Water"
            label = self.water_title

        if not images:
            return

        segmentation = self.get_segmentation_for_slice(self.current_index)
        if segmentation is None:
            if self.prediction_files:
                self.set_title(label, title+" (Not Segmented)", self.current_index + 1, self.slices)
        else:
            self.change_alpha(segmentation, view, slider, images, self.current_index)

    def change_alpha(self, segmentation, view: QGraphicsView, slider: QSlider, images, index):
        """Blend the colored segmentation over the slice with the slider's opacity."""
        pixel_array = self._normalize_to_uint8(images[index])
        color_segmentation = self.apply_colormap(segmentation)

        alpha = slider.value() / 100.0
        alpha_mask = np.where(segmentation != 0, alpha, 0).astype(np.float32)[..., np.newaxis]
        rgb = np.stack([pixel_array] * 3, axis=-1).astype(np.float32)
        blended = (1 - alpha_mask) * rgb + alpha_mask * color_segmentation
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        qimage = QImage(blended.data, blended.shape[1], blended.shape[0],
                        blended.strides[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        view.setScene(scene)
        view.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def swap_datasets(self):
        """Swap dataset 1 and dataset 2."""
        self.dicom_files1, self.dicom_files2 = self.dicom_files2, self.dicom_files1
        self.fat, self.muscle = self.muscle, self.fat
        self.patient_info1, self.patient_info2 = self.patient_info2, self.patient_info1
        self.update_display()

    def _select_patients_for_prediction(self):
        """Checklist dialog letting the user pick which loaded patient(s) to run
        prediction on; all patients are checked by default. Returns the selected
        indices into self.patients, or [] if the user cancelled."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Patients for Prediction")
        dialog.setGeometry(500, 300, 420, 400)

        layout = QVBoxLayout()
        title = QLabel("<h3>Select patient(s) to run prediction on:</h3>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        list_widget = QListWidget()
        for record in self.patients:
            item = QListWidgetItem(record.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            list_widget.addItem(item)
        layout.addWidget(list_widget)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        dialog.setLayout(layout)

        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return []
        return [i for i in range(list_widget.count()) if list_widget.item(i).checkState() == Qt.CheckState.Checked]

    def _run_prediction_for_record(self, record, model, start_slice, end_slice):
        """Export record's slices and run prediction into its own scratch subfolder."""
        images_dir, prediction_dir = self._patient_scratch_dirs(record.scratch_index)

        for file in glob.glob(os.path.join(images_dir, "*")):
            os.remove(file)
        for file in glob.glob(os.path.join(prediction_dir, "*")):
            os.remove(file)

        last_slice = min(end_slice, len(record.dicom_files1), len(record.dicom_files2))
        fat_slices = [record.fat[i].astype(np.uint16) for i in range(start_slice - 1, last_slice)]
        water_slices = [record.muscle[i].astype(np.uint16) for i in range(start_slice - 1, last_slice)]

        export_case_images(
            model, fat_slices, water_slices,
            record.pixel_spacing1 or (1.0, 1.0),
            record.slice_spacing1 or 1.0,
            images_dir,
            first_slice_number=start_slice,
        )

        record.start_slice_used = start_slice
        record.end_slice_used = last_slice

        self.predictor.predict_from_files(images_dir, prediction_dir,
                                    save_probabilities=False, overwrite=False,
                                    num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                    folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

        ext = model.file_ending
        record.prediction_files = sorted(glob.glob(os.path.join(prediction_dir, f"dataset_*{ext}")))
        if self.is_3d_model and record.prediction_files:
            record.segmentation_volume = read_segmentation_array(self.reader_writer, record.prediction_files[0])
        else:
            record.segmentation_volume = None

    def run_prediction(self):
        """Export images in the selected model's expected format and run nnUNet
        prediction for one or more selected patients."""
        if not self.patients:
            QMessageBox.warning(self, "No Patients Loaded", "Load at least one patient before running prediction.")
            return

        if self.selected_model is None:
            QMessageBox.warning(self, "No Model Selected",
                "No trained model is available. Use Run > Select Model... once one is installed.")
            return

        selected_indices = self._select_patients_for_prediction()
        if not selected_indices:
            return

        model = self.selected_model
        start_slice = self.start_slice.value()
        end_slice = self.end_slice.value()

        # Persist the active patient's state before iterating over records.
        if self.current_patient_index >= 0:
            self._save_active_into_record(self.patients[self.current_patient_index])

        processing_dialog = ProcessingDialog()
        processing_dialog.show()
        QApplication.processEvents()

        start_time = time.time()
        errors = []

        try:
            self.predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=self.device,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False
            )
            self.predictor.initialize_from_trained_model_folder(
                model.path,
                use_folds=model.folds,
                checkpoint_name='checkpoint_final.pth',
            )

            for index in selected_indices:
                record = self.patients[index]
                try:
                    self._run_prediction_for_record(record, model, start_slice, end_slice)
                except Exception as e:
                    errors.append(f"{record.name}: {e}")

        except Exception as e:
            errors.insert(0, f"Failed to initialize model: {e}")

        finally:
            processing_dialog.close()

        # Refresh the active patient's display in case its record changed.
        if self.current_patient_index >= 0:
            self._load_record_into_active(self.patients[self.current_patient_index])
            self.update_display()

        elapsed_time = time.time() - start_time
        succeeded = len(selected_indices) - len(errors)
        if errors:
            QMessageBox.critical(self, "Prediction Errors",
                f"Finished {succeeded}/{len(selected_indices)} patient(s) in {elapsed_time:.2f}s.\n\nErrors:\n" + "\n".join(errors))
        else:
            QMessageBox.information(self, "Prediction Complete",
                f"nnUNet prediction finished for {succeeded} patient(s).\nElapsed time: {elapsed_time:.2f} seconds")

    def load_prediction_results(self):
        """Load segmentation results and display them alongside original images."""
        if self.selected_model is None:
            return

        _, prediction_folder = self._patient_scratch_dirs()
        ext = self.selected_model.file_ending

        self.prediction_files = sorted(glob.glob(os.path.join(prediction_folder, f"dataset_*{ext}")))

        if not self.prediction_files:
            QMessageBox.warning(self, "Error", "Prediction results not found!")
            return

        if self.is_3d_model:
            self.segmentation_volume = read_segmentation_array(self.reader_writer, self.prediction_files[0])
        else:
            self.segmentation_volume = None

        self.update_display()

    def load_segmentation(self):
        """Load a segmentation produced earlier (elsewhere, or by a previous run) instead
        of running prediction. Reuses the Start/End slice range to know which absolute
        slices the loaded segmentation corresponds to."""
        if not self.dicom_files1 or not self.dicom_files2:
            QMessageBox.warning(self, "Error", "Load both Fat and Water datasets before loading a segmentation.")
            return
        if self.selected_model is None:
            QMessageBox.warning(self, "No Model Selected",
                "Select a model first (Run > Select Model...) so the expected label format is known.")
            return

        model = self.selected_model
        start_slice = self.start_slice.value()
        end_slice = self.end_slice.value()
        expected_count = end_slice - start_slice + 1

        if model.is_3d:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Segmentation Volume", "", f"Segmentation (*{model.file_ending})")
            if not file_path:
                return
            source_files = [file_path]
        else:
            folder = QFileDialog.getExistingDirectory(self, "Select Folder with Segmentation Slices")
            if not folder:
                return
            source_files = sorted(glob.glob(os.path.join(folder, f"*{model.file_ending}")))
            if not source_files:
                QMessageBox.warning(self, "Error", f"No {model.file_ending} files found in that folder.")
                return
            if len(source_files) != expected_count:
                QMessageBox.warning(self, "Slice Count Mismatch",
                    f"Found {len(source_files)} segmentation file(s), but the current slice range "
                    f"({start_slice}-{end_slice}) expects {expected_count}.\n"
                    f"Adjust the Start/End slice range to match, then try again.")
                return

        _, prediction_folder = self._patient_scratch_dirs()
        for file in glob.glob(os.path.join(prediction_folder, "*")):
            os.remove(file)

        if model.is_3d:
            shutil.copy2(source_files[0], os.path.join(prediction_folder, f"dataset_0001{model.file_ending}"))
        else:
            # Sorted files are assumed to be in slice order; rename them onto the
            # currently selected absolute slice range so display/statistics line up.
            for offset, src in enumerate(source_files):
                dst = os.path.join(prediction_folder, f"dataset_{start_slice + offset:04d}{model.file_ending}")
                shutil.copy2(src, dst)

        self.start_slice_used = start_slice
        self.end_slice_used = end_slice

        self.load_prediction_results()

        if self.is_3d_model and self.segmentation_volume is not None:
            self.end_slice_used = start_slice + self.segmentation_volume.shape[0] - 1

        if self.prediction_files:
            self.actionShowStatistics.setEnabled(True)
            self.save_button.setEnabled(True)
            self.actionSaveSeg.setEnabled(True)

    def show_statistics(self):
        if not self.prediction_files or self.selected_model is None:
            QMessageBox.warning(self, "Error", "No prediction results available!")
            return

        label_names = get_label_names(self.selected_model.labels)

        if self.is_3d_model:
            # Only one case is exported for a 3D model, so report whole-volume voxel
            # ratios rather than a mean of per-slice ratios (not the same statistic,
            # but the more standard one for a single volumetric result).
            segmentation = self.segmentation_volume
            total = np.sum(segmentation > 0) if segmentation is not None else 0
            mean_percentages = {
                label_id: (np.sum(segmentation == label_id) / total * 100 if total > 0 else 0)
                for label_id in label_names
            }
        else:
            per_label_lists = {label_id: [] for label_id in label_names}
            for file in self.prediction_files:
                segmentation = read_segmentation_array(self.reader_writer, file)[0]
                total_tissue = np.sum(segmentation > 0)
                if total_tissue > 0:
                    for label_id in label_names:
                        per_label_lists[label_id].append(np.sum(segmentation == label_id) / total_tissue)
            mean_percentages = {
                label_id: (np.mean(values) * 100 if values else 0)
                for label_id, values in per_label_lists.items()
            }

        lines = [f"Mean Percentage of Tissues in Slices {self.start_slice_used}-{self.end_slice_used}:", ""]
        for label_id, name in label_names.items():
            lines.append(f"{(name + ':').ljust(30)} {mean_percentages[label_id]:6.2f} %")
        stats_message = "\n".join(lines)

        box = QMessageBox(self)
        box.setWindowTitle("Statistics")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(f"<pre>{stats_message}</pre>")
        box.exec()

    def save_segmentations(self):
        """Save colored PNGs plus the raw segmentation files in the model's native format."""
        if self.selected_model is None or not self.prediction_files:
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save Segmentations")
        if not folder:
            return

        if self.is_3d_model:
            colored_volume = self.apply_colormap(self.segmentation_volume)
            for z in range(colored_volume.shape[0]):
                save_path = os.path.join(folder, f"segmentation_{z + 1:04d}.png")
                imageio.imwrite(save_path, colored_volume[z])
        else:
            for i, file_path in enumerate(self.prediction_files):
                segmentation = read_segmentation_array(self.reader_writer, file_path)[0]
                colored_segmentation = self.apply_colormap(segmentation)
                save_path = os.path.join(folder, f"segmentation_{i+1:04d}.png")
                imageio.imwrite(save_path, colored_segmentation)

        raw_folder = os.path.join(folder, "raw_segmentation")
        os.makedirs(raw_folder, exist_ok=True)
        for file_path in self.prediction_files:
            shutil.copy2(file_path, os.path.join(raw_folder, os.path.basename(file_path)))

        QMessageBox.information(self, "Save Complete", f"Segmentations saved in:\n{folder}")

    def apply_colormap(self, segmentation):
        """Map segmentation labels to colors."""
        color_segmentation = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
        for label, color in self.current_colormap.items():
            color_segmentation[segmentation == label] = color
        return color_segmentation

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def rebuild_labels_ui(self):
        """Rebuild the legend/statistics panels from the selected model's own labels."""
        self._clear_layout(self.legend_layout)
        self._clear_layout(self.statistics_layout)
        self.legend_swatches = {}
        self.stat_labels = {}

        if self.selected_model is not None:
            label_names = get_label_names(self.selected_model.labels)
            for row, (label_id, name) in enumerate(label_names.items()):
                swatch = QLabel()
                swatch.setFixedSize(20, 20)
                swatch.setAutoFillBackground(True)
                self.legend_layout.addWidget(swatch, row, 0)
                self.legend_layout.addWidget(QLabel(name), row, 1)
                self.legend_swatches[label_id] = swatch

                self.statistics_layout.addWidget(QLabel(f"{name}:"), row, 0)
                value_label = QLabel("0.0 %")
                self.statistics_layout.addWidget(value_label, row, 1)
                self.stat_labels[label_id] = value_label

            self.is_3d_model = self.selected_model.is_3d
            self.reader_writer = get_reader_writer_cls(self.selected_model.dataset_json)()
        else:
            self.is_3d_model = False
            self.reader_writer = None

        self.segmentation_volume = None
        self.set_colormap(self.current_palette_name)

    def select_model(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Model")
        dialog.setGeometry(500, 300, 480, 360)

        layout = QVBoxLayout()

        title = QLabel("<h3>Select trained model:</h3>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        list_widget = QListWidget()
        layout.addWidget(list_widget)

        for model in self.available_models:
            list_widget.addItem(model.display_name)
        if self.selected_model in self.available_models:
            list_widget.setCurrentRow(self.available_models.index(self.selected_model))

        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        dialog.setLayout(layout)

        def confirm_selection():
            if not self.available_models:
                dialog.reject()
                return
            row = list_widget.currentRow()
            if row >= 0:
                new_model = self.available_models[row]
                if new_model != self.selected_model:
                    self.selected_model = new_model
                    self.rebuild_labels_ui()

                    # Switching models invalidates every patient's cached prediction,
                    # not just the currently active one
                    for record in self.patients:
                        record.prediction_files = []
                        record.segmentation_volume = None
                    self.prediction_files = []
                    self.actionShowStatistics.setEnabled(False)
                    self.save_button.setEnabled(False)
                    self.actionSaveSeg.setEnabled(False)
                    self.update_display()

                if self.dicom_files1 and self.dicom_files2:
                    self.run_button.setEnabled(True)
            dialog.accept()

        ok_button.clicked.connect(confirm_selection)
        cancel_button.clicked.connect(dialog.reject)

        dialog.exec()

    def set_colormap(self, palette_name):
        self.current_palette_name = palette_name
        if self.selected_model is not None:
            self.current_colormap = build_colormap(self.selected_model.labels, palette_name)
        self.update_overlay(1)
        self.update_overlay(2)

        for label_id, swatch in self.legend_swatches.items():
            r, g, b = self.current_colormap.get(label_id, [128, 128, 128])
            swatch.setStyleSheet(f"background-color: rgb({r}, {g}, {b});")

    def update_statistics(self, stats):
        for label_id, widget in self.stat_labels.items():
            widget.setText(f"{stats.get(label_id, 0) * 100:.1f} %")

    def mouse_scroll(self, event):
        """Scroll through slices using mouse wheel."""
        if event.angleDelta().y() > 0:
            if self.current_index < self.slice_slider.maximum():
                self.current_index += 1
        else:
            if self.current_index > 0:
                self.current_index -= 1

        self.slice_slider.setValue(self.current_index+1)

    def scroll_slider_by_one(self, slider, event):
        """Scroll a slider by exactly one step per wheel notch (QSlider's own
        wheelEvent can jump several steps depending on the reported wheel delta)."""
        if event.angleDelta().y() > 0:
            if slider.value() < slider.maximum():
                slider.setValue(slider.value() + 1)
        else:
            if slider.value() > slider.minimum():
                slider.setValue(slider.value() - 1)

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.Wheel and source in [self.fat_image, self.water_image]:
            self.mouse_scroll(event)
            return True
        if event.type() == QEvent.Type.Wheel and source in [self.patient_slider, self.slice_slider]:
            self.scroll_slider_by_one(source, event)
            return True
        return super().eventFilter(source, event)

    def reset(self):
        """Reset the viewer to its initial state, discarding every loaded patient."""
        self.patients = []
        self.current_patient_index = -1
        self._next_scratch_index = 0

        self.dicom_files1 = []
        self.dicom_files2 = []
        self.fat = []
        self.muscle = []
        self.prediction_files = []
        self.current_index = 0
        self.start_slice_used = 0
        self.end_slice_used = 0
        self.rows = 0
        self.cols = 0
        self.slices = 0
        self.patient_info1 = []
        self.patient_info2 = []
        self.pixel_spacing1 = None
        self.pixel_spacing2 = None
        self.slice_spacing1 = None
        self.slice_spacing2 = None
        self.segmentation_volume = None

        self._clear_scratch_trees()

        self.slice_slider.setValue(1)
        self.slice_slider.setMaximum(1)
        self.start_slice.setValue(1)
        self.end_slice.setValue(1)
        self.run_button.setEnabled(False)
        self.actionShowStatistics.setEnabled(False)
        self.save_button.setEnabled(False)
        self.actionSaveSeg.setEnabled(False)
        self.patient_name_label.setText("No patient loaded")
        self._sync_patient_slider_range()
        self.set_title(self.fat_title, "Fat", 0, 0)
        self.set_title(self.water_title, "Water", 0, 0)
        self.update_display()
        self.update_statistics({})

    def set_title(self, title_label, title, slice_index, total_slices):
        """Create a formatted title for the image display."""
        slice_info = f"Slice {slice_index}/{total_slices}"
        formatted_title = f"<div align='center'><b>{title}</b><br>{slice_info}</div>"
        title_label.setText(formatted_title)

    def about(self):
        """Show an About dialog with application information."""
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("About LegSeg")
        about_dialog.setGeometry(500, 300, 400, 200)

        layout = QVBoxLayout()
        label = QLabel("<h2>LegSeg</h2>"
                       "<p>Version 3.0</p>"
                       "<p>Developed by Denisa Hyvlova</p>"
                       "<p>© 2025–2026 ISI CAS</p>")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        about_dialog.setLayout(layout)
        about_dialog.exec()

    def set_device(self):
        device_dialog = QDialog(self)
        device_dialog.setWindowTitle("Select Device")
        device_dialog.setGeometry(500, 300, 380, 240)

        layout = QVBoxLayout()

        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "No GPU detected"

        title = QLabel("<h3>Select computation device:</h3>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        info_label = QLabel(
            f"<p><b>GPU status:</b> {'Available ✅' if gpu_available else 'Not available ❌'}<br>"
            f"<b>Device:</b> {gpu_name}</p>"
        )
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)

        cpu_radio = QRadioButton("CPU")
        gpu_radio = QRadioButton("GPU (CUDA)")
        gpu_radio.setEnabled(gpu_available)

        if self.device.type == "cuda" and gpu_available:
            gpu_radio.setChecked(True)
        else:
            cpu_radio.setChecked(True)

        layout.addWidget(cpu_radio)
        layout.addWidget(gpu_radio)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        device_dialog.setLayout(layout)

        def confirm_selection():
            if gpu_radio.isChecked():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            device_dialog.accept()

        ok_button.clicked.connect(confirm_selection)
        cancel_button.clicked.connect(device_dialog.reject)

        device_dialog.exec()

    def show_dataset_info(self):
        def format_info(info, label):
            if not info:
                return f"{label}:\n    No data loaded.\n"
            return (f"{label}:\n"
                    f"    Name:        {info[0]}\n"
                    f"    ID:          {info[1]}\n"
                    f"    Study Date:  {info[2]}\n"
                    f"    Study Desc:  {info[3]}\n"
                    f"    Series Desc: {info[4]}\n")

        info_text = (
            "<pre>"
            + format_info(self.patient_info1, "Dataset 1") + "\n"
            + format_info(self.patient_info2, "Dataset 2")
            + "</pre>"
        )

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Patient Information")
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setText(info_text)
        msg_box.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LegSegWindow()
    window.show()
    sys.exit(app.exec())

