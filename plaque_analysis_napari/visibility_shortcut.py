"""Route Q to the plate tool associated with the current dock or layer."""

from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence, QShortcut
from qtpy.QtWidgets import QApplication


class VisibilityShortcut(QShortcut):
    def __init__(self, window, viewer):
        super().__init__(QKeySequence("Q"), window)
        self.setObjectName("plaque_analysis_visibility")
        self.setContext(Qt.ShortcutContext.WindowShortcut)
        self.setAutoRepeat(False)
        self.viewer = viewer
        self.targets = []
        self.activated.connect(self.toggle)

    def add_target(self, widget, layer_names, toggle):
        target = (widget, layer_names, toggle)
        self.targets.append(target)
        widget.destroyed.connect(lambda: self.targets.remove(target))

    def toggle(self):
        focus = QApplication.focusWidget()
        active = self.viewer.layers.selection.active
        available = [(widget, layers, toggle) for widget, layers, toggle in self.targets
                     if getattr(widget, layers[0]) is not None]
        for widget, _, toggle in available:
            if focus == widget or (focus is not None and widget.isAncestorOf(focus)):
                toggle()
                return
        for widget, layers, toggle in available:
            if active is not None and any(active == getattr(widget, name) for name in layers):
                toggle()
                return
        if available:
            available[-1][2]()


def register_visibility_shortcut(viewer, widget, layer_names, toggle):
    window = viewer.window.main_menu.window()
    shortcut = window.findChild(VisibilityShortcut, "plaque_analysis_visibility")
    if shortcut is None:
        shortcut = VisibilityShortcut(window, viewer)
    shortcut.add_target(widget, layer_names, toggle)
