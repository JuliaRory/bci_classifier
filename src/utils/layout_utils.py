from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt

def create_hbox(widgets, spacing=6, margins=(0, 0, 0, 0)):
    layout = QHBoxLayout()
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)
    for w in widgets:
        layout.addWidget(w)
    layout.setAlignment(Qt.AlignLeft)
    layout.addStretch()
    return layout

def create_vbox(widgets, spacing=6, margins=(0, 0, 0, 0)):
    layout = QVBoxLayout()
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)
    for w in widgets:
        layout.addWidget(w)
    return layout