"""Count plaques in an original image using a saved agar mask."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtCore import QSignalBlocker, Qt, QTimer
from qtpy.QtGui import QKeySequence, QShortcut
from qtpy.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from plaque_analysis_napari.agar_region import boundary_paths, dish_diameter_px
from plaque_analysis_napari.plate_io import IMAGE_FILTER, agar_mask_path, display_limits, load_grayscale
from plaque_analysis_napari.visibility_shortcut import register_visibility_shortcut

from .plaque_detection import detect_plaques, estimate_dish_diameter_px
from .plaque_split import find_split_proposals, split_from_stroke
from .plaque_statistics import export_plaque_statistics


@dataclass
class PlateImage:
    path: Path
    image: np.ndarray
    diameter_px: float | None
    mask_path: Path | None = None
    mask: np.ndarray | None = None
    contrast_limits: tuple[float, float] | None = None
    mask_error: str | None = None


def read_mask(path, shape):
    # Agar Region writes LZW TIFFs. OpenCV can read these without a separate codec.
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError("Could not read the agar mask TIFF.")
    if mask.shape != shape:
        raise ValueError(f"The agar mask must have the image's dimensions {shape}.")
    if mask.dtype.kind not in "bui" or not np.isin(np.unique(mask), (0, 1, 255)).all():
        raise ValueError("Choose a binary agar mask exported by Agar Region.")
    inside = mask != 0
    if not inside.any() or inside.all():
        raise ValueError("The agar mask must contain both included and excluded pixels.")
    return inside.astype(np.uint8)


def read_plate(path):
    image = load_grayscale(path)
    try:
        diameter_px = dish_diameter_px(image)
    except ValueError:
        diameter_px = None
    plate = PlateImage(path, image, diameter_px)
    expected = agar_mask_path(path)
    if expected.is_file():
        try:
            plate.mask = read_mask(expected, image.shape)
            plate.mask_path = expected
        except ValueError as error:
            plate.mask_error = str(error)
    plate.contrast_limits = display_limits(image, plate.mask)
    return plate


def count_from_files(plate, mask_path):
    mask = read_mask(mask_path, plate.image.shape)
    diameter_px = plate.diameter_px
    if diameter_px is None:
        diameter_px = estimate_dish_diameter_px(mask)
    labels = detect_plaques(plate.image, mask, diameter_px)
    return mask_path, mask, labels, find_split_proposals(labels, plate.image)


def plaque_masks_path(image_path):
    return image_path.parent / "out" / f"{image_path.stem}_plaque_masks.tif"


def read_saved_plaques(path, image):
    """Restore the integer IDs and scan them for review suggestions."""
    try:
        labels = tifffile.imread(path)
    except (OSError, ValueError) as error:
        raise ValueError("Could not read the plaque labels TIFF.") from error
    if labels.shape != image.shape:
        raise ValueError(f"Mask dimensions {labels.shape} must match the image {image.shape}.")
    if labels.dtype.kind not in "iu" or labels.min() < 0 or labels.max() > np.iinfo(np.uint32).max:
        raise ValueError("Use a plaque-label TIFF with nonnegative integer IDs.")
    labels = labels.astype(np.uint32, copy=False)
    return labels, find_split_proposals(labels, image)


class CountControls(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.plate = None
        self.image_layer = None
        self.boundary_layer = None
        self.plaque_layer = None
        self.drawing_layer = None
        self.plaque_tool = None
        self.split_proposal = None
        self.split_preview_layer = None
        self.split_flags_layer = None
        self.pending_splits = deque()
        self.reviewing_splits = False
        self.plaque_ids = set()
        self.worker = None
        self.last_folder = Path.home()

        self._recount = QTimer(self)
        self._recount.setSingleShot(True)
        self._recount.setInterval(150)
        self._recount.timeout.connect(self.update_plaque_count)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        self.open_button = QPushButton("Open original image…")
        self.open_button.clicked.connect(self.browse)
        layout.addWidget(self.open_button)

        self.mask_button = QPushButton("Choose agar mask…")
        self.mask_button.clicked.connect(self.browse_mask)
        layout.addWidget(self.mask_button)

        self.mask_name = QLabel("No agar mask loaded.")
        self.mask_name.setWordWrap(True)
        self.mask_name.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.mask_name)

        self.count_button = QPushButton("Count plaques")
        self.count_button.clicked.connect(self.count_plaques)
        layout.addWidget(self.count_button)

        self.open_plaques_button = QPushButton("Open saved plaque masks…")
        self.open_plaques_button.clicked.connect(self.browse_saved_plaques)
        self.open_plaques_button.setToolTip("Resume editing from an exported plaque-label TIFF.")
        layout.addWidget(self.open_plaques_button)

        self.plaque_count = QLabel()
        layout.addWidget(self.plaque_count)

        self.visibility_button = QPushButton("Show plaque masks   Q")
        self.visibility_button.setCheckable(True)
        self.visibility_button.clicked.connect(self.set_plaque_visibility)
        layout.addWidget(self.visibility_button)

        self.new_plaque_button = QPushButton("New plaque   N")
        self.new_plaque_button.clicked.connect(self.new_plaque)
        self.new_plaque_button.setToolTip("Hold and drag around a plaque, then release to fill it.")
        layout.addWidget(self.new_plaque_button)

        self.remove_plaque_button = QPushButton("Remove plaque   R")
        self.remove_plaque_button.clicked.connect(self.start_removing_plaque)
        self.remove_plaque_button.setToolTip("Activate removal, then click a colored mask to delete it.")
        layout.addWidget(self.remove_plaque_button)

        self.review_splits_button = QPushButton("Scan for splits")
        self.review_splits_button.clicked.connect(self.review_splits)
        self.review_splits_button.setToolTip("Review plate-wide suggestions one at a time.")
        layout.addWidget(self.review_splits_button)

        self.split_button = QPushButton("Manual split   S")
        self.split_button.clicked.connect(self.start_split)
        self.split_button.setToolTip("Drag a divider through one mask, from outside to outside.")
        layout.addWidget(self.split_button)

        self.split_actions = QWidget()
        actions_layout = QHBoxLayout(self.split_actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        self.accept_split_button = QPushButton("Accept split   A")
        self.accept_split_button.clicked.connect(self.accept_split)
        actions_layout.addWidget(self.accept_split_button)
        self.cancel_split_button = QPushButton("Cancel")
        self.cancel_split_button.clicked.connect(self.skip_or_cancel_split)
        actions_layout.addWidget(self.cancel_split_button)
        layout.addWidget(self.split_actions)

        self.undo_button = QPushButton("Undo   U")
        self.undo_button.clicked.connect(self.undo_plaque_edit)
        self.undo_button.setToolTip("Undo the last addition, removal, split, or Labels edit.")
        layout.addWidget(self.undo_button)

        self.export_plaques_button = QPushButton("Export plaque masks")
        self.export_plaques_button.clicked.connect(self.export_plaques)
        layout.addWidget(self.export_plaques_button)

        self.statistics_button = QPushButton("Plaque statistics")
        self.statistics_button.setToolTip("Save a CSV and two histograms from the current plaque masks.")
        self.statistics_button.clicked.connect(self.save_plaque_statistics)
        layout.addWidget(self.statistics_button)

        self.status = QLabel("Open the original plate image to load its saved agar mask.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setMinimumWidth(220)
        self.status.setMinimumHeight(4 * self.status.fontMetrics().lineSpacing())
        layout.addWidget(self.status)

        register_visibility_shortcut(
            self.viewer, self, ("plaque_layer", "image_layer", "boundary_layer", "drawing_layer",
                                "split_preview_layer", "split_flags_layer"),
            self.toggle_plaque_visibility,
        )
        self.edit_shortcuts = []
        window = self.viewer.window.main_menu.window()
        for key, button in (("N", self.new_plaque_button),
                            ("R", self.remove_plaque_button), ("S", self.split_button),
                            ("U", self.undo_button), ("A", self.accept_split_button)):
            shortcut = QShortcut(QKeySequence(key), window)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(button.click)
            self.destroyed.connect(shortcut.deleteLater)
            self.edit_shortcuts.append((shortcut, button))

        self.viewer.layers.selection.events.active.connect(self.cancel_plaque_tool)
        self.viewer.layers.events.removed.connect(self._layer_removed)
        self.sync_controls()

    def browse(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open original plate image", str(self.last_folder), IMAGE_FILTER,
        )
        if filename:
            self.open_path(Path(filename))

    def open_path(self, path):
        if self.worker is not None or not self.confirm_replacement(
            "Open another plate?", "Replace the current plate and its plaque masks?",
        ):
            return
        path = Path(path).expanduser().absolute()
        self.last_folder = path.parent
        self.run_worker(read_plate, (path,), self.show_plate, "Could not open image.")
        self.status.setText("Loading the plate image and saved agar mask…")

    def browse_mask(self):
        if self.plate is None or self.worker is not None:
            return
        path = self.plate.mask_path or agar_mask_path(self.plate.path)
        folder = path.parent if path.parent.is_dir() else self.plate.path.parent
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open saved agar mask", str(folder), "TIFF mask (*.tif *.tiff)",
        )
        if filename:
            self.open_mask(Path(filename))

    def open_mask(self, path):
        if self.image_layer is None or self.worker is not None:
            return
        if not self.confirm_replacement(
            "Change agar mask?", "Replace the agar mask and clear the current plaque masks?",
        ):
            return
        try:
            mask = read_mask(path, self.plate.image.shape)
        except ValueError as error:
            self.status.setText(str(error))
            return
        self.remove_layer(self.plaque_layer)
        self.set_mask(path, mask, display_limits(self.plate.image, mask))
        self.status.setText("Count plaques or open saved plaque masks.")

    def browse_saved_plaques(self):
        if self.image_layer is None or self.worker is not None:
            return
        expected = plaque_masks_path(self.plate.path)
        start = expected if expected.parent.is_dir() else self.plate.path.parent
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open saved plaque masks", str(start), "Plaque masks (*.tif *.tiff)",
        )
        if filename:
            self.open_saved_plaques(Path(filename))

    def open_saved_plaques(self, path):
        if self.image_layer is None or self.worker is not None:
            return
        if not self.confirm_replacement(
            "Open saved plaque masks?", "Replace the current plaque masks and any manual edits?",
        ):
            return
        self.run_worker(
            read_saved_plaques, (path, self.plate.image),
            self.show_saved_plaques, "Could not load saved plaques.",
        )
        self.status.setText("Loading saved plaque masks…")

    def confirm_replacement(self, title, message):
        return self.plaque_layer is None or QMessageBox.question(
            self, title, message, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes

    def run_worker(self, operation, args, on_result, error_message, *, cancel_tool=True):
        if cancel_tool:
            self.cancel_plaque_tool()
        self.worker = thread_worker(operation, ignore_errors=True)(*args)
        self.worker.returned.connect(on_result)
        self.worker.errored.connect(lambda error: self.status.setText(f"{error_message} {error}"))
        self.worker.finished.connect(self.finish_work)
        self.sync_controls()
        self.worker.start()

    def finish_work(self):
        self.worker = None
        self.sync_controls()

    def remove_layer(self, layer):
        if layer is not None and layer in self.viewer.layers:
            self.viewer.layers.remove(layer)

    def show_plate(self, plate):
        for layer in (self.plaque_layer, self.boundary_layer, self.image_layer):
            self.remove_layer(layer)
        self.plate = plate
        self.image_layer = self.viewer.add_image(
            plate.image, name=plate.path.name, colormap="gray", rgb=False,
            contrast_limits=plate.contrast_limits,
        )
        if plate.mask is not None:
            self.set_mask(plate.mask_path, plate.mask, plate.contrast_limits)
            self.status.setText("Count plaques or open saved plaque masks.")
        else:
            self.mask_name.setText("No agar mask loaded.")
            self.mask_name.setToolTip(str(agar_mask_path(plate.path)))
            hint = plate.mask_error or "No matching agar mask found in the image’s out folder."
            self.status.setText(f"{hint} Choose an agar mask to count, or open saved plaque masks.")
            self.viewer.reset_view(layers=[self.image_layer])
        self.sync_controls()

    def set_mask(self, path, mask, contrast_limits):
        self.plate.mask_path = path
        self.plate.mask = mask
        self.mask_name.setText("Agar mask loaded")
        self.mask_name.setToolTip(str(path))
        self.image_layer.contrast_limits = contrast_limits
        self.show_boundary(mask, fit=True)
        self.sync_controls()

    def show_boundary(self, mask, *, fit=False):
        paths = boundary_paths(mask)
        image = self.image_layer
        if self.boundary_layer is None and paths:
            layer = self.viewer.add_shapes(
                paths, name="Saved agar boundary", shape_type="polygon", edge_width=2.5,
                edge_color="#26d9ef", face_color="transparent",
                scale=image.scale, translate=image.translate, rotate=image.rotate,
                shear=image.shear, affine=image.affine.affine_matrix,
            )
            layer.editable = False
            self.boundary_layer = layer
        elif self.boundary_layer is not None:
            self.boundary_layer.data = paths
        if fit:
            self.viewer.layers.selection.active = self.image_layer
            layers = [self.boundary_layer] if paths and self.viewer.dims.ndisplay == 2 else [self.image_layer]
            self.viewer.reset_view(layers=layers, margin=0.13)

    def _layer_removed(self, event):
        if event.value == self.boundary_layer:
            self.boundary_layer = None
        if event.value == self.image_layer:
            self.image_layer = None
        if self.drawing_layer is not None and event.value == self.drawing_layer:
            self.drawing_layer = None
            self.cancel_plaque_tool()
        if event.value == self.split_flags_layer:
            self.split_flags_layer = None
        if event.value == self.split_preview_layer:
            self.split_preview_layer = None
            if self.reviewing_splits:
                self.cancel_plaque_tool()
            else:
                self.clear_split_preview()
        if event.value == self.plaque_layer:
            self.cancel_plaque_tool()
            self._recount.stop()
            self.plaque_layer.events.paint.disconnect(self.schedule_recount)
            self.plaque_layer.events.labels_update.disconnect(self.schedule_recount)
            self.plaque_layer.events.set_data.disconnect(self.schedule_recount)
            self.plaque_layer.events.visible.disconnect(self.sync_plaque_visibility)
            self.plaque_layer.events.mode.disconnect(self.cancel_plaque_tool)
            self.plaque_layer.mouse_drag_callbacks.remove(self.handle_plaque_click)
            self.plaque_layer = None
            self.plaque_ids.clear()
            self.pending_splits.clear()
            self.remove_layer(self.split_flags_layer)
        self.sync_controls()

    def sync_controls(self, event=None):
        idle = self.worker is None
        has_image = self.image_layer is not None
        has_plaques = self.plaque_layer is not None
        self.open_button.setEnabled(idle)
        self.mask_button.setEnabled(has_image and idle)
        self.count_button.setEnabled(has_image and idle)
        self.open_plaques_button.setEnabled(has_image and idle)
        self.plaque_count.setVisible(has_plaques)
        visible = has_plaques and self.plaque_layer.visible
        with QSignalBlocker(self.visibility_button):
            self.visibility_button.setEnabled(has_plaques)
            self.visibility_button.setChecked(visible)
            self.visibility_button.setText(
                "Hide plaque masks   Q" if visible else "Show plaque masks   Q"
            )
        self.new_plaque_button.setEnabled(has_plaques and idle)
        self.new_plaque_button.setText(
            "Stop adding   N" if self.plaque_tool == "draw" else "New plaque   N"
        )
        self.undo_button.setEnabled(has_plaques and idle)
        self.remove_plaque_button.setEnabled(
            has_plaques and idle and (bool(self.plaque_ids) or self.plaque_tool == "remove")
        )
        self.remove_plaque_button.setText(
            "Stop removing   R" if self.plaque_tool == "remove" else "Remove plaque   R"
        )
        previewing = self.split_proposal is not None
        can_split = has_plaques and idle and bool(self.plaque_ids)
        self.review_splits_button.setEnabled(can_split and (not previewing or self.reviewing_splits))
        if self.reviewing_splits:
            review_text = "Stop split review"
        elif self.pending_splits:
            review_text = f"Review splits ({len(self.pending_splits)})"
        else:
            review_text = "Scan for splits"
        self.review_splits_button.setText(review_text)
        self.split_button.setEnabled(can_split)
        self.split_button.setText(
            "Stop manual split   S" if self.plaque_tool == "split" else "Manual split   S"
        )
        self.split_actions.setVisible(previewing)
        self.accept_split_button.setEnabled(previewing and idle)
        self.cancel_split_button.setText("Skip" if self.reviewing_splits else "Cancel")
        self.export_plaques_button.setEnabled(has_plaques and idle)
        self.statistics_button.setEnabled(has_plaques and idle)
        for shortcut, button in self.edit_shortcuts:
            shortcut.setEnabled(button.isEnabled())

    def count_plaques(self):
        if self.worker is not None or self.image_layer is None:
            return
        path = self.plate.mask_path or agar_mask_path(self.plate.path)
        if self.plate.mask_path is None and not path.is_file():
            self.status.setText("Save an agar mask with Agar Region or click Choose agar mask.")
            return
        if not self.confirm_replacement(
            "Count plaques again?", "Replace the current plaque masks and any manual edits?",
        ):
            return
        self.run_worker(
            count_from_files, (self.plate, path),
            self.show_plaques, "Could not count plaques.",
        )
        if self.plate.diameter_px is None:
            self.status.setText("Using the saved agar mask to estimate dish size and count plaques…")
        else:
            self.status.setText("Detecting plaques inside the saved agar mask…")

    def show_plaques(self, result):
        if self.image_layer is None:
            self.status.setText("Open the original image again to count plaques.")
            return
        path, mask, labels, suggestions = result
        if not np.array_equal(mask, self.plate.mask):
            self.plate.mask_path = path
            self.plate.mask = mask
            self.mask_name.setText("Agar mask loaded")
            self.mask_name.setToolTip(str(path))
            self.show_boundary(mask)
        self.mount_plaque_masks(labels, suggestions)

    def show_saved_plaques(self, result):
        if self.image_layer is None:
            self.status.setText("Open the original image again to load saved masks.")
            return
        labels, suggestions = result
        self.mount_plaque_masks(labels, suggestions)
        if suggestions:
            self.status.setText(f"Saved masks loaded. {len(suggestions)} split suggestions to review.")
        else:
            self.status.setText("Saved masks loaded. You can resume editing.")

    def mount_plaque_masks(self, labels, suggestions):
        self.remove_layer(self.plaque_layer)
        image = self.image_layer
        layer = self.viewer.add_labels(
            labels, name="Plaque masks", opacity=0.65,
            scale=image.scale, translate=image.translate, rotate=image.rotate,
            shear=image.shear, affine=image.affine.affine_matrix,
        )
        layer.preserve_labels = True
        layer.brush_size = 7
        layer.events.paint.connect(self.schedule_recount)
        layer.events.labels_update.connect(self.schedule_recount)
        layer.events.set_data.connect(self.schedule_recount)
        layer.events.visible.connect(self.sync_plaque_visibility)
        layer.events.mode.connect(self.cancel_plaque_tool)
        layer.mouse_drag_callbacks.append(self.handle_plaque_click)
        self.plaque_layer = layer
        self.update_plaque_count()
        self.set_split_suggestions(suggestions)

    def schedule_recount(self, event=None):
        self._recount.start()

    def update_plaque_count(self):
        layer = self.plaque_layer
        if layer is None:
            return
        self.plaque_ids = set(np.unique(layer.data).tolist()) - {0}
        self.plaque_count.setText(f"Plaques: {len(self.plaque_ids)}")
        self.sync_controls()

    def set_plaque_visibility(self, visible):
        if self.plaque_layer is not None:
            self.plaque_layer.visible = visible

    def toggle_plaque_visibility(self):
        self.set_plaque_visibility(not self.plaque_layer.visible)

    def sync_plaque_visibility(self, event=None):
        visible = self.plaque_layer.visible
        for overlay in (self.split_preview_layer, self.split_flags_layer, self.drawing_layer):
            if overlay is not None:
                overlay.visible = visible
        self.update_tool_navigation()
        self.sync_controls()

    def new_plaque(self):
        self.toggle_plaque_tool("draw")

    def start_removing_plaque(self):
        self.toggle_plaque_tool("remove")

    def start_split(self):
        self.toggle_plaque_tool("split")

    def toggle_plaque_tool(self, tool):
        layer = self.plaque_layer
        if layer is None or self.worker is not None:
            return
        if self.plaque_tool == tool:
            self.cancel_plaque_tool()
            return

        self.prepare_plaque_tool(layer)
        if tool in {"draw", "split"}:
            outline = self.viewer.add_shapes(
                name="Plaque edit stroke", ndim=2, shape_type="path",
                edge_width=2, edge_color="#f7d14e", face_color="transparent",
                scale=layer.scale, translate=layer.translate, rotate=layer.rotate,
                shear=layer.shear, affine=layer.affine.affine_matrix,
            )
            outline.editable = False
            self.viewer.layers.selection.active = layer
            self.drawing_layer = outline
        self.plaque_tool = tool
        self.update_tool_navigation()
        self.sync_controls()
        self.describe_plaque_tool()

    def describe_plaque_tool(self):
        instructions = {
            "draw": "Drag around a plaque and release to fill. Press N to stop adding.",
            "remove": "Click each colored mask to remove it. Press R to stop removing.",
            "split": "Drag a divider across one mask, from outside to outside. Press S to stop.",
        }
        self.status.setText(instructions.get(self.plaque_tool,
                                            "Use the Labels tools to review the plaque masks."))

    def update_tool_navigation(self):
        layer = self.plaque_layer
        if layer is not None and self.plaque_tool in {"draw", "split"}:
            layer.mouse_pan = not layer.visible or self.split_proposal is not None

    def prepare_plaque_tool(self, layer):
        self.cancel_plaque_tool()
        self.viewer.dims.ndisplay = 2
        layer.visible = True
        layer.mode = "pan_zoom"
        self.viewer.layers.selection.active = layer

    def clear_split_preview(self):
        preview = self.split_preview_layer
        self.split_proposal = None
        self.split_preview_layer = None
        self.reviewing_splits = False
        self.remove_layer(preview)
        self.update_tool_navigation()
        self.sync_controls()

    def cancel_plaque_tool(self, event=None):
        if self.plaque_tool is None and self.split_proposal is None:
            return
        outline = self.drawing_layer
        self.drawing_layer = None
        self.plaque_tool = None
        if self.plaque_layer is not None:
            self.plaque_layer.mouse_pan = self.plaque_layer.mode == "pan_zoom"
        self.clear_split_preview()
        self.remove_layer(outline)
        self.sync_controls()
        self.describe_plaque_tool()

    def handle_plaque_click(self, layer, event):
        if event.button != 1 or self.worker is not None or not layer.visible:
            return
        if self.plaque_tool == "remove":
            event.handled = True
            self.remove_plaque_at(layer, event.position)
            return
        outline = self.drawing_layer
        if outline is None or self.split_proposal is not None:
            return

        points = [layer.world_to_data(event.position)]
        last_cursor = np.asarray(event.pos)
        yield
        while event.type == "mouse_move" and self.drawing_layer == outline:
            cursor = np.asarray(event.pos)
            if np.linalg.norm(cursor - last_cursor) >= 1:
                points.append(layer.world_to_data(event.position))
                outline.data = [np.asarray(points)]
                outline.editable = False
                last_cursor = cursor
            yield

        if self.drawing_layer != outline:
            return
        outline.data = []
        outline.editable = False
        if not layer.visible:
            return
        points.append(layer.world_to_data(event.position))
        vertices = np.rint(points).astype(np.int32)
        if self.plaque_tool == "split":
            try:
                proposal = split_from_stroke(layer.data, vertices)
            except ValueError as error:
                self.status.setText(str(error))
                return
            self.show_split_preview(proposal)
        else:
            self.add_drawn_plaque(layer, vertices)

    def add_drawn_plaque(self, layer, vertices):
        if len(np.unique(vertices, axis=0)) < 3 or cv2.contourArea(vertices[:, ::-1]) < 1:
            self.status.setText("Outline too small. Drag around a plaque to try again.")
            return
        new_label = int(layer.data.max()) + 1
        layer.paint_polygon(vertices, new_label)
        layer.selected_label = new_label
        self.update_plaque_count()
        self.describe_plaque_tool()

    def undo_plaque_edit(self):
        if self.plaque_layer is None:
            return
        if self.split_proposal is not None and not self.reviewing_splits:
            self.clear_split_preview()
            self.describe_plaque_tool()
            return
        self.plaque_layer.undo()
        self.update_plaque_count()
        if self.reviewing_splits:
            self.describe_split_preview()
        else:
            self.describe_plaque_tool()

    def remove_plaque_at(self, layer, position):
        label = layer.get_value(position, world=True)
        if not label:
            self.status.setText("No mask there. Click inside a colored plaque mask.")
            return
        layer.data_setitem(np.where(layer.data == label), 0)
        self.update_plaque_count()
        self.status.setText(f"Removed plaque {label}. Click another, press R to stop, or U to undo.")

    def set_split_suggestions(self, suggestions):
        if self.plaque_layer is None:
            return
        self.pending_splits = deque(suggestions)
        self.update_split_flags()
        self.sync_controls()
        if suggestions:
            self.status.setText(
                f"{len(suggestions)} likely touching pairs flagged. Click Review splits to inspect them."
            )
        else:
            self.status.setText("No likely touching pairs found. Use Manual split to draw a divider.")

    def update_split_flags(self):
        if not self.pending_splits:
            self.remove_layer(self.split_flags_layer)
            return

        boxes = []
        for proposal in self.pending_splits:
            row, col = proposal.bounds
            boxes.append([[row.start - 4, col.start - 4], [row.start - 4, col.stop + 4],
                          [row.stop + 4, col.stop + 4], [row.stop + 4, col.start - 4]])
        if self.split_flags_layer is not None:
            self.split_flags_layer.data = boxes
            self.split_flags_layer.editable = False
            return

        layer = self.plaque_layer
        selection = self.viewer.layers.selection
        with selection.events.active.blocker(self.cancel_plaque_tool):
            flags = self.viewer.add_shapes(
                boxes, name="Likely touching plaques", shape_type="rectangle",
                edge_width=3, edge_color="#f7d14e", face_color="transparent",
                visible=layer.visible, scale=layer.scale, translate=layer.translate,
                rotate=layer.rotate, shear=layer.shear, affine=layer.affine.affine_matrix,
            )
            flags.editable = False
            selection.active = layer
        self.split_flags_layer = flags

    def review_splits(self):
        if self.reviewing_splits:
            self.cancel_plaque_tool()
        elif self.pending_splits:
            self.prepare_plaque_tool(self.plaque_layer)
            self.show_next_split()
        else:
            self.run_worker(
                find_split_proposals, (self.plaque_layer.data.copy(), self.plate.image),
                self.set_split_suggestions, "Could not scan for splits.",
            )
            self.status.setText("Scanning the plate for touching plaques…")

    def show_next_split(self):
        while self.pending_splits:
            proposal = self.pending_splits[0]
            if self.proposal_is_current(proposal):
                self.update_split_flags()
                self.show_split_preview(proposal, review=True)
                return
            self.pending_splits.popleft()
        self.update_split_flags()
        self.sync_controls()
        self.status.setText("Split review complete.")

    def skip_or_cancel_split(self):
        reviewing = self.reviewing_splits
        self.clear_split_preview()
        if reviewing:
            self.pending_splits.popleft()
            self.show_next_split()
        else:
            self.describe_plaque_tool()

    def show_split_preview(self, proposal, *, review=False):
        # Keep the proposal in an independent overlay until the user accepts it.
        self.clear_split_preview()
        layer = self.plaque_layer
        overlay = np.zeros(layer.data.shape, dtype=np.uint8)
        overlay[proposal.bounds] = proposal.regions
        colormap = DirectLabelColormap(color_dict={
            None: (0, 0, 0, 0), 1: (0.14, 0.80, 0.94, 1), 2: (1, 0.66, 0.20, 1),
        })
        selection = self.viewer.layers.selection
        with selection.events.active.blocker(self.cancel_plaque_tool):
            preview = self.viewer.add_labels(
                overlay, name="Proposed plaque split", opacity=0.93, colormap=colormap,
                visible=layer.visible, scale=layer.scale, translate=layer.translate,
                rotate=layer.rotate, shear=layer.shear, affine=layer.affine.affine_matrix,
            )
            preview.editable = False
            selection.active = layer
        self.split_preview_layer = preview
        self.split_proposal = proposal
        self.reviewing_splits = review
        self.update_tool_navigation()
        if review:
            row, col = proposal.bounds
            corners = np.array([layer.data_to_world((y, x))
                                for y in (row.start, row.stop) for x in (col.start, col.stop)])
            camera = self.viewer.scene.camera
            camera.center = (*camera.center[:-2], *corners.mean(axis=0))
            extent = max(np.ptp(corners, axis=0))
            camera.zoom = min(24, 0.38 * min(self.viewer.canvas.size) / extent)
        self.sync_controls()
        self.describe_split_preview()

    def describe_split_preview(self):
        proposal = self.split_proposal
        sizes = np.bincount(proposal.regions.ravel(), minlength=3)
        suffix = (f"{len(self.pending_splits)} remaining"
                  if self.reviewing_splits else "Accept split or cancel")
        self.status.setText(f"Plaque {proposal.label} → {sizes[1]} px² + {sizes[2]} px²\n{suffix}")

    def proposal_is_current(self, proposal):
        label_pixels = self.plaque_layer.data == proposal.label
        proposed = proposal.regions != 0
        return (np.array_equal(proposed, label_pixels[proposal.bounds])
                and np.count_nonzero(label_pixels) == np.count_nonzero(proposed))

    def accept_split(self):
        proposal = self.split_proposal
        layer = self.plaque_layer
        if proposal is None or layer is None:
            return
        reviewing = self.reviewing_splits
        current = self.proposal_is_current(proposal)
        self.clear_split_preview()
        if not current:
            if reviewing:
                self.pending_splits.popleft()
                self.show_next_split()
            else:
                self.status.setText("The mask changed. Draw the divider again.")
            return

        rows, cols = np.nonzero(proposal.regions == 2)
        indices = (rows + proposal.bounds[0].start, cols + proposal.bounds[1].start)
        new_label = int(layer.data.max()) + 1
        layer.data_setitem(indices, new_label)
        self.update_plaque_count()
        if reviewing:
            self.pending_splits.popleft()
            self.show_next_split()
        else:
            self.pending_splits = deque(
                item for item in self.pending_splits if item.label != proposal.label
            )
            self.update_split_flags()
            self.sync_controls()
            self.status.setText(f"Split plaque {proposal.label} into IDs {proposal.label} and {new_label}. "
                                "Draw another divider, press S to stop, or U to undo.")

    def export_plaques(self):
        if self.plaque_layer is None:
            return
        destination = plaque_masks_path(self.plate.path)
        try:
            destination.parent.mkdir(exist_ok=True)
            tifffile.imwrite(destination, self.plaque_layer.data, photometric="minisblack",
                             compression="zlib")
        except (OSError, ValueError) as error:
            self.status.setText(f"Export failed. {error}")
            return
        self.status.setText(f"Plaque masks saved in out/{destination.name}")

    def save_plaque_statistics(self):
        if self.plaque_layer is None or self.worker is not None:
            return
        self.run_worker(
            export_plaque_statistics, (self.plate.path, self.plaque_layer.data.copy()),
            self.show_statistics_saved, "Could not save plaque statistics.", cancel_tool=False,
        )
        self.status.setText("Saving plaque statistics…")

    def show_statistics_saved(self, count):
        self.status.setText(f"Saved statistics for {count} plaques in out/ (CSV and two PNGs).")
