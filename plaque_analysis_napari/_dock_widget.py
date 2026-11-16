"""Napari dock widget for importing a plate and reviewing its agar boundary."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtWidgets import QFileDialog, QLabel, QPushButton, QVBoxLayout, QWidget

from .agar_region import (
    boundary_paths, detect_agar, mask_from_boundaries,
)
from .plate_io import IMAGE_FILTER, agar_mask_path, display_limits, load_grayscale
from .visibility_shortcut import register_visibility_shortcut


@dataclass
class Plate:
    path: Path
    image: np.ndarray
    detected_mask: np.ndarray | None
    contrast_limits: tuple[float, float]
    detection_error: str | None = None


def analyze_image(path):
    image = load_grayscale(path)
    try:
        mask = detect_agar(image)
    except ValueError as error:
        return Plate(path, image, None, display_limits(image), detection_error=str(error))
    return Plate(path, image, mask, display_limits(image, mask))


class PlateControls(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        viewer = napari_viewer
        self.viewer = viewer
        self.plate = None
        self.image_layer = None
        self.boundary_layer = None
        self.initial_paths = []
        self.worker = None
        self.last_folder = Path.home()

        layout = QVBoxLayout(self)
        self.open_button = QPushButton("Open image…")
        self.open_button.clicked.connect(self.browse)
        layout.addWidget(self.open_button)

        self.boundary_button = QPushButton("Show agar boundary   Q")
        self.boundary_button.setCheckable(True)
        self.boundary_button.setEnabled(False)
        self.boundary_button.clicked.connect(self.set_boundary_visible)
        layout.addWidget(self.boundary_button)

        self.edit_button = QPushButton("Edit boundary")
        self.edit_button.setEnabled(False)
        self.edit_button.clicked.connect(self.edit_boundary)
        self.edit_button.setToolTip("Select the outline’s vertices so you can drag them.")
        layout.addWidget(self.edit_button)

        self.draw_button = QPushButton("Draw boundary")
        self.draw_button.clicked.connect(self.draw_boundary)
        layout.addWidget(self.draw_button)

        self.export_button = QPushButton("Export agar mask")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_mask)
        layout.addWidget(self.export_button)

        self.status = QLabel("Open a full-plate grayscale image.")
        self.status.setWordWrap(True)
        self.status.setMinimumWidth(220)
        layout.addWidget(self.status)

        register_visibility_shortcut(viewer, self, ("boundary_layer", "image_layer"),
                                     self.toggle_boundary)
        viewer.layers.events.removed.connect(self._layer_removed)
        self.sync_controls()

    def browse(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open plate image", str(self.last_folder),
            IMAGE_FILTER,
        )
        if filename:
            self.open_path(Path(filename))

    def open_path(self, path):
        if self.worker is not None:
            return
        path = Path(path).expanduser().absolute()
        self.last_folder = path.parent
        self.status.setText("Detecting the agar region…")
        self.run_worker(analyze_image, (path,), self.show_plate, "Could not open image.")

    def run_worker(self, operation, args, on_result, error_message):
        self.worker = thread_worker(operation, ignore_errors=True)(*args)
        self.worker.returned.connect(on_result)
        self.worker.errored.connect(lambda error: self.status.setText(f"{error_message} {error}"))
        self.worker.finished.connect(self.finish_work)
        self.sync_controls()
        self.worker.start()

    def finish_work(self):
        self.worker = None
        self.sync_controls()

    def show_plate(self, plate):
        for layer in (self.boundary_layer, self.image_layer):
            if layer is not None and layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self.plate = plate
        self.image_layer = self.viewer.add_image(
            plate.image, name=plate.path.name, colormap="gray",
            contrast_limits=plate.contrast_limits, rgb=False,
        )

        self.initial_paths = []
        if plate.detected_mask is None:
            self.status.setText(f"{plate.detection_error} Click Draw boundary to trace it manually.")
        else:
            self.initial_paths = boundary_paths(plate.detected_mask)
            self.add_boundary_layer()
            self.status.setText("Agar region detected.")
        self.sync_controls()
        self.viewer.layers.selection.active = self.image_layer
        if self.boundary_layer is not None and self.viewer.dims.ndisplay == 2:
            self.viewer.reset_view(layers=[self.boundary_layer], margin=0.13)
        else:
            self.viewer.reset_view(layers=[self.image_layer])

    def add_boundary_layer(self):
        self.boundary_layer = self.viewer.add_shapes(
            [path.copy() for path in self.initial_paths], ndim=2, shape_type="polygon",
            edge_width=2.5, edge_color="#26d9ef", face_color="transparent",
            name="Agar boundary",
        )
        self.boundary_layer.events.visible.connect(self.sync_controls)
        self.boundary_layer.events.data.connect(self.sync_controls)

    def _layer_removed(self, event):
        if event.value == self.boundary_layer:
            self.boundary_layer.events.visible.disconnect(self.sync_controls)
            self.boundary_layer.events.data.disconnect(self.sync_controls)
            self.boundary_layer = None
        self.sync_controls()

    def sync_controls(self, event=None):
        layer = self.boundary_layer
        enabled = layer is not None
        visible = enabled and layer.visible
        # Napari's lasso starts out as an open path and becomes a polygon on completion.
        complete = enabled and layer.nshapes > 0 and "path" not in layer.shape_type
        with QSignalBlocker(self.boundary_button):
            self.boundary_button.setEnabled(enabled)
            self.boundary_button.setChecked(visible)
            self.boundary_button.setText(
                "Hide agar boundary   Q" if visible else "Show agar boundary   Q"
            )
        idle = self.worker is None
        self.open_button.setEnabled(idle)
        self.draw_button.setEnabled(self.plate is not None and idle and not complete)
        self.edit_button.setEnabled(complete and idle)
        self.export_button.setEnabled(complete and idle)

    def draw_boundary(self):
        if not self.draw_button.isEnabled():
            return
        if self.boundary_layer is None:
            self.initial_paths = []
            self.add_boundary_layer()
        layer = self.boundary_layer
        self.viewer.dims.ndisplay = 2
        layer.visible = True
        self.viewer.layers.selection.active = layer
        layer.mode = "add_polygon_lasso"
        self.status.setText("Hold and drag to trace the agar edge, then release to close the outline.")
        self.sync_controls()

    def edit_boundary(self):
        layer = self.boundary_layer
        if layer is None:
            return
        self.viewer.dims.ndisplay = 2
        layer.visible = True
        self.viewer.layers.selection.active = layer
        layer.mode = "direct"
        if layer.nshapes and not layer.selected_data:
            layer.selected_data = {0}
        self.status.setText("Drag a vertex to reshape the boundary.")

    def set_boundary_visible(self, visible):
        if self.boundary_layer is not None:
            self.boundary_layer.visible = visible

    def toggle_boundary(self):
        if self.boundary_layer is not None:
            self.set_boundary_visible(not self.boundary_layer.visible)

    def current_mask(self):
        """Return the current counting region in the imported image’s pixel grid."""
        layer = self.boundary_layer
        if layer is None:
            raise ValueError("No agar boundary is available.")
        if not layer.nshapes or "path" in layer.shape_type:
            raise ValueError("Draw and close the agar boundary before exporting.")

        # Napari's transform tool can move the whole layer. Convert its vertices
        # into the image grid so a visible move also appears in the saved mask.
        origin, row, column = (
            np.asarray(self.image_layer.world_to_data(layer.data_to_world(point)))
            for point in ((0, 0), (1, 0), (0, 1))
        )
        transform = np.stack((row - origin, column - origin))
        paths = [path @ transform + origin for path in layer.data]
        if self.plate.detected_mask is not None and len(paths) == len(self.initial_paths) and all(
                current.shape == initial.shape
                and np.allclose(current, initial, rtol=0, atol=1e-6)
                for current, initial in zip(paths, self.initial_paths)
        ):
            return self.plate.detected_mask

        return mask_from_boundaries(self.plate.image.shape, paths)

    def export_mask(self):
        plate = self.plate
        if plate is None:
            return
        destination = agar_mask_path(plate.path)
        try:
            mask = self.current_mask()
            destination.parent.mkdir(exist_ok=True)
            if not cv2.imwrite(str(destination), mask):
                raise OSError("Could not write the mask TIFF.")
        except (OSError, ValueError, cv2.error) as error:
            self.status.setText(f"Export failed. {error}")
            return
        self.status.setText(f"Mask saved in out/{destination.name}")
