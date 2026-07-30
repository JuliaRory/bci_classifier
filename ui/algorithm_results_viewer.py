import os
import subprocess
import sys
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from h5py import File
from mne.viz import plot_topomap
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.utils.montage_processing import find_ch_idx, get_channel_names, get_topo_positions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "algorithm"
ALL_VALUE = "All"
MONTAGE_PATH = PROJECT_ROOT / "resources" / "mks64_standard.ced"
BAD_CHANNELS = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]


TABLES = {
    "Summary": {
        "file": "agency_ebci_component_scores_summary.csv",
        "columns": [
            "subject",
            "record",
            "matrices_count",
            "matrices_calculated",
            "epoch_path",
        ],
    },
    "Component Scores": {
        "file": "agency_ebci_component_scores_all.csv",
        "columns": [
            "file",
            "band",
            "matrix",
            "component_1based",
            "component_relative",
            "edge_group",
            "eigenvalue",
            "eigen_score_abs_diff",
            "eigen_score_logit",
            "eigengap",
            "weighted_contra",
            "weighted_ipsi",
            "locality",
            "gof",
            "gof_coef",
            "final_score_1",
            "final_score_2",
            "final_score_3",
            "final_score_4",
            "final_score_5",
        ],
    },
    "Component Ranking": {
        "file": "agency_ebci_component_scores_ranked_by_file.csv",
        "columns": [
            "rank_within_file",
            "final_score_method",
            "band",
            "component_1based",
            "component_relative",
            "edge_group",
            "final_score",
            "eigenvalue",
            "eigengap",
            "weighted_contra",
            "weighted_ipsi",
            "locality",
            "gof",
            "gof_coef",
            "file",
            "matrix",
        ],
    },
    "CV All Sets": {
        "file": "agency_ebci_timeseries_component_set_cv_all.csv",
        "columns": [
            "band",
            "components",
            "absolute_components",
            "n_components",
            "balanced_accuracy",
            "brier_score",
            "final_score_1_mean_score",
            "final_score_1_comp_set_score",
            "final_score_2_mean_score",
            "final_score_2_comp_set_score",
            "final_score_3_mean_score",
            "final_score_3_comp_set_score",
            "final_score_4_mean_score",
            "final_score_4_comp_set_score",
            "final_score_5_mean_score",
            "final_score_5_comp_set_score",
            "file",
            "matrix",
        ],
    },
}

for index in range(1, 6):
    TABLES[f"CV Ranking final_score_{index}"] = {
        "file": f"agency_ebci_timeseries_component_set_rankings_final_score_{index}.csv",
        "columns": [
            "rank_within_file",
            "rank_global",
            "band",
            "components",
            "absolute_components",
            "n_components",
            "mean_score",
            "balanced_accuracy",
            "brier_score",
            "comp_set_score",
            "component_score_values",
            "file",
            "matrix",
            "probability_plot",
        ],
    }


def open_path(path):
    if not path:
        return
    path = Path(path)
    if not path.exists():
        return
    if hasattr(os, "startfile"):
        os.startfile(str(path))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def resolve_workspace_path(value):
    if not isinstance(value, str) or not value.strip() or value.lower() == "nan":
        return None

    raw = Path(value)
    if raw.exists():
        return raw

    text = value.replace("\\", "/")
    for marker in ("results/", "data/", "models/"):
        index = text.find(marker)
        if index >= 0:
            candidate = PROJECT_ROOT / Path(text[index:])
            if candidate.exists():
                return candidate
    return raw


def display_text(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def sanitize_filename_part(value):
    text = display_text(value)
    text = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in text)
    return "_".join(text.split()) or "value"


def read_table(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def parse_component_list(value):
    if value is None or pd.isna(value):
        return []
    if isinstance(value, (list, tuple)):
        return [int(component) for component in value]
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return [int(component) for component in parsed]


def good_topomap_positions():
    labels = get_channel_names(str(MONTAGE_PATH))
    good_indices = [
        find_ch_idx(channel, str(MONTAGE_PATH))
        for channel in labels
        if channel not in BAD_CHANNELS
    ]
    return get_topo_positions(str(MONTAGE_PATH))[good_indices]


def component_grid_indices(n_components):
    first = list(range(min(3, n_components)))
    last_start = max(len(first), n_components - 3)
    last = list(range(last_start, n_components))
    return first + [component for component in last if component not in first]


def component_label(component, n_components):
    if component < 3:
        return str(component)
    return str(component - n_components)


def selected_components_from_row(row):
    components = parse_component_list(row.get("absolute_components", None))
    if components:
        return components
    component = row.get("component", None)
    if component is not None and not pd.isna(component):
        return [int(component)]
    return []


def component_grid_plot_path(results_root, row):
    matrix_path = resolve_workspace_path(row.get("matrix_path", ""))
    if not matrix_path:
        return None
    rank = display_text(row.get("rank_within_file", "component"))
    method = display_text(row.get("final_score_method", "component_scores")) or "component_scores"
    file_part = sanitize_filename_part(row.get("file", row.get("record", "record")))
    band_part = sanitize_filename_part(row.get("band", "band"))
    components_part = sanitize_filename_part(row.get("components", row.get("component_1based", "component")))
    filename = f"{file_part}_{method}_rank_{rank}_{band_part}_{components_part}_local_scale.png"
    return Path(results_root) / "component_grid_plots" / filename


def ensure_component_grid_plot(results_root, row):
    matrix_path = resolve_workspace_path(row.get("matrix_path", ""))
    if not matrix_path or not matrix_path.exists():
        return None

    output_path = component_grid_plot_path(results_root, row)
    if output_path and output_path.exists():
        return output_path

    with File(matrix_path, "r") as h5f:
        spatial_patterns = h5f["projForward"][:]
        eigenvalues = h5f["evals"][:]

    n_components = spatial_patterns.shape[1]
    grid_components = component_grid_indices(n_components)
    selected_components = set(selected_components_from_row(row))
    positions = good_topomap_positions()

    fig = plt.figure(figsize=(15.5, 7.2))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.5, 1, 1, 1], wspace=0.34, hspace=0.34)
    ax_eig = fig.add_subplot(gs[:, 0])
    x = np.arange(1, len(eigenvalues) + 1)
    ax_eig.plot(x, eigenvalues, color="black", linewidth=1.2)
    ax_eig.scatter(x, eigenvalues, color="black", s=14, alpha=0.65)
    if selected_components:
        selected_x = [component + 1 for component in selected_components if 0 <= component < len(eigenvalues)]
        selected_y = [eigenvalues[component - 1] for component in selected_x]
        ax_eig.scatter(selected_x, selected_y, color="crimson", s=60, zorder=4)
    ax_eig.set_ylim(-0.02, 1.02)
    ax_eig.set_xlabel("CSP component")
    ax_eig.set_ylabel("eigenvalue")
    ax_eig.set_title("Eigenvalues")
    ax_eig.grid(alpha=0.25)

    for order, component in enumerate(grid_components):
        row_index = 0 if order < 3 else 1
        col_index = order % 3 + 1
        ax = fig.add_subplot(gs[row_index, col_index])
        component_values = spatial_patterns[:, component]
        vmax = float(np.nanmax(np.abs(component_values)))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        image, _ = plot_topomap(
            component_values,
            positions,
            axes=ax,
            show=False,
            contours=0,
            sphere=0.5,
            image_interp="cubic",
            extrapolate="head",
            cmap="jet",
            vlim=(-vmax, vmax),
        )
        is_selected = component in selected_components
        title = (
            f"CSP {component + 1}\n"
            f"rel {component_label(component, n_components)}, "
            f"eig {eigenvalues[component]:.3f}"
        )
        ax.set_title(title, fontsize=10, fontweight="bold" if is_selected else "normal")
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_ticks([-vmax, vmax])
        cbar.ax.set_yticklabels([f"{-vmax:.2g}", f"{vmax:.2g}"])
        cbar.ax.tick_params(labelsize=7)
        if is_selected:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("crimson")
                spine.set_linewidth(3)

    title = (
        f"{display_text(row.get('file', ''))} | "
        f"{display_text(row.get('final_score_method', ''))} "
        f"rank {display_text(row.get('rank_within_file', ''))}\n"
        f"band {display_text(row.get('band', ''))} | "
        f"components {display_text(row.get('components', row.get('component_1based', '')))} | "
        f"BA {display_text(row.get('balanced_accuracy', ''))} | "
        f"Brier {display_text(row.get('brier_score', ''))}"
    )
    fig.suptitle(title, fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


class ImagePane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path = None
        self.pixmap = QPixmap()

        self.path_label = QLabel("No plot selected")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.path_label.setWordWrap(True)

        self.image_label = QLabel("No plot selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(420, 300)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.image_label)

        self.open_button = QPushButton("Open Plot")
        self.open_folder_button = QPushButton("Open Folder")
        self.open_button.clicked.connect(lambda: open_path(self.image_path))
        self.open_folder_button.clicked.connect(self.open_folder)

        buttons = QHBoxLayout()
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.open_folder_button)
        buttons.addStretch()

        layout = QVBoxLayout(self)
        layout.addWidget(self.path_label)
        layout.addWidget(self.scroll, stretch=1)
        layout.addLayout(buttons)

    def set_image(self, path):
        self.image_path = Path(path) if path else None
        if not self.image_path or not self.image_path.exists():
            self.pixmap = QPixmap()
            self.path_label.setText("No plot for selected row")
            self.image_label.setText("No plot for selected row")
            self.image_label.setPixmap(QPixmap())
            return

        self.pixmap = QPixmap(str(self.image_path))
        self.path_label.setText(str(self.image_path))
        if self.pixmap.isNull():
            self.image_label.setText("Could not load plot")
            self.image_label.setPixmap(QPixmap())
            return
        self.image_label.setText("")
        self._update_scaled_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self.pixmap.isNull():
            return
        viewport = self.scroll.viewport().size()
        scaled = self.pixmap.scaled(
            max(420, viewport.width() - 8),
            max(300, viewport.height() - 8),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def open_folder(self):
        if self.image_path:
            open_path(self.image_path.parent)


class AlgorithmResultsViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.results_root = DEFAULT_RESULTS_ROOT
        self.table_name = ""
        self.table_path = None
        self.raw_df = pd.DataFrame()
        self.filtered_df = pd.DataFrame()

        self.setWindowTitle("Algorithm Results Viewer")
        self.resize(1640, 920)
        self._setup_widgets()
        self._setup_layout()
        self._connect()
        self.refresh_all()

    def _setup_widgets(self):
        self.root_label = QLabel(str(self.results_root))
        self.root_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.choose_root_button = QPushButton("Choose Results Folder")
        self.refresh_button = QPushButton("Refresh")
        self.open_root_button = QPushButton("Open Folder")

        self.table_combo = QComboBox()
        self.subject_combo = QComboBox()
        self.record_combo = QComboBox()
        self.band_combo = QComboBox()
        self.method_combo = QComboBox()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search in visible table")
        self.rank_limit_spin = QSpinBox()
        self.rank_limit_spin.setRange(0, 100000)
        self.rank_limit_spin.setValue(3)
        self.rank_limit_spin.setSpecialValueText("All")

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(130)

        self.component_pane = ImagePane()
        self.plot_pane = ImagePane()

        self.open_table_button = QPushButton("Open Source Table")
        self.open_component_button = QPushButton("Open Component Plot")
        self.open_plot_button = QPushButton("Open Probability Plot")
        self.open_table_button.clicked.connect(lambda: open_path(self.table_path))
        self.open_component_button.clicked.connect(self.open_component_plot)
        self.open_plot_button.clicked.connect(self.open_selected_plot)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.component_pane, "Components + Eigenvalues")
        self.tabs.addTab(self.plot_pane, "Probability Plot")
        self.tabs.addTab(self.details, "Row Details")

    def _setup_layout(self):
        central = QWidget()
        self.setCentralWidget(central)

        source_box = QGroupBox("Source")
        source_layout = QGridLayout(source_box)
        source_layout.addWidget(QLabel("Folder"), 0, 0)
        source_layout.addWidget(self.root_label, 0, 1, 1, 5)
        source_layout.addWidget(self.choose_root_button, 0, 6)
        source_layout.addWidget(self.open_root_button, 0, 7)
        source_layout.addWidget(self.refresh_button, 0, 8)
        source_layout.addWidget(QLabel("Table"), 1, 0)
        source_layout.addWidget(self.table_combo, 1, 1, 1, 3)
        source_layout.addWidget(self.open_table_button, 1, 4)
        source_layout.addWidget(self.open_component_button, 1, 5)
        source_layout.addWidget(self.open_plot_button, 1, 6)

        filter_box = QGroupBox("Filters")
        filter_layout = QGridLayout(filter_box)
        filter_layout.addWidget(QLabel("Subject"), 0, 0)
        filter_layout.addWidget(self.subject_combo, 0, 1)
        filter_layout.addWidget(QLabel("Record"), 0, 2)
        filter_layout.addWidget(self.record_combo, 0, 3)
        filter_layout.addWidget(QLabel("Band"), 0, 4)
        filter_layout.addWidget(self.band_combo, 0, 5)
        filter_layout.addWidget(QLabel("Method"), 1, 0)
        filter_layout.addWidget(self.method_combo, 1, 1)
        filter_layout.addWidget(QLabel("Max rank"), 1, 2)
        filter_layout.addWidget(self.rank_limit_spin, 1, 3)
        filter_layout.addWidget(QLabel("Search"), 1, 4)
        filter_layout.addWidget(self.search_edit, 1, 5)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(source_box)
        left_layout.addWidget(filter_box)
        left_layout.addWidget(self.summary_label)
        left_layout.addWidget(self.table, stretch=1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.tabs)
        splitter.setSizes([1050, 590])

        layout = QVBoxLayout(central)
        layout.addWidget(splitter, stretch=1)

    def _connect(self):
        self.choose_root_button.clicked.connect(self.choose_results_root)
        self.refresh_button.clicked.connect(self.refresh_all)
        self.open_root_button.clicked.connect(lambda: open_path(self.results_root))
        self.table_combo.currentTextChanged.connect(self.load_selected_table)
        self.subject_combo.currentTextChanged.connect(self.apply_filters)
        self.record_combo.currentTextChanged.connect(self.apply_filters)
        self.band_combo.currentTextChanged.connect(self.apply_filters)
        self.method_combo.currentTextChanged.connect(self.apply_filters)
        self.search_edit.textChanged.connect(self.apply_filters)
        self.rank_limit_spin.valueChanged.connect(self.apply_filters)
        self.table.itemSelectionChanged.connect(self.on_table_selection)

    def choose_results_root(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose results folder", str(self.results_root))
        if folder:
            self.results_root = Path(folder)
            self.root_label.setText(str(self.results_root))
            self.refresh_all()

    def refresh_all(self):
        available = []
        for name, spec in TABLES.items():
            if (self.results_root / spec["file"]).exists():
                available.append(name)

        self.table_combo.blockSignals(True)
        self.table_combo.clear()
        self.table_combo.addItems(available)
        self.table_combo.blockSignals(False)

        if available:
            default_table = "CV Ranking final_score_1"
            self.table_combo.setCurrentText(default_table if default_table in available else available[0])
            self.load_selected_table()
        else:
            self.raw_df = pd.DataFrame()
            self.filtered_df = pd.DataFrame()
            self.populate_filter_combos()
            self.populate_table()
            self.summary_label.setText(f"No result tables found in {self.results_root}")

    def load_selected_table(self):
        self.table_name = self.table_combo.currentText()
        if not self.table_name:
            return

        self.table_path = self.results_root / TABLES[self.table_name]["file"]
        self.raw_df = read_table(self.table_path)
        self.populate_filter_combos()
        self.apply_filters()

    def combo_value(self, combo):
        value = combo.currentText()
        return None if value == ALL_VALUE else value

    def populate_filter_combos(self):
        subject = self.subject_combo.currentText()
        record = self.record_combo.currentText()
        band = self.band_combo.currentText()
        method = self.method_combo.currentText()

        self._fill_combo(self.subject_combo, self.values_for("subject"), subject)
        self._fill_combo(self.record_combo, self.values_for("record"), record)
        self._fill_combo(self.band_combo, self.values_for("band"), band)
        self._fill_combo(self.method_combo, self.values_for("final_score_method"), method)

    def _fill_combo(self, combo, values, previous):
        values = [ALL_VALUE] + values
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(values)
        combo.setCurrentText(previous if previous in values else ALL_VALUE)
        combo.blockSignals(False)

    def values_for(self, column):
        if self.raw_df.empty or column not in self.raw_df.columns:
            return []
        values = self.raw_df[column].dropna().astype(str).unique().tolist()
        return sorted(values)

    def apply_filters(self):
        df = self.raw_df.copy()
        for column, combo in (
            ("subject", self.subject_combo),
            ("record", self.record_combo),
            ("band", self.band_combo),
            ("final_score_method", self.method_combo),
        ):
            value = self.combo_value(combo)
            if value and column in df.columns:
                df = df[df[column].astype(str) == value]

        rank_limit = self.rank_limit_spin.value()
        if rank_limit > 0 and "rank_within_file" in df.columns:
            df = df[pd.to_numeric(df["rank_within_file"], errors="coerce") <= rank_limit]

        query = self.search_edit.text().strip().lower()
        if query:
            display_columns = self.display_columns(df)
            if display_columns:
                mask = (
                    df[display_columns]
                    .astype(str)
                    .apply(lambda row: query in " ".join(row).lower(), axis=1)
                )
                df = df[mask]

        self.filtered_df = df.reset_index(drop=True)
        self.populate_table()
        self.update_summary()

    def display_columns(self, df=None):
        df = self.raw_df if df is None else df
        configured = TABLES.get(self.table_name, {}).get("columns", [])
        columns = [column for column in configured if column in df.columns]
        if columns:
            return columns
        return list(df.columns)

    def populate_table(self):
        columns = self.display_columns(self.filtered_df)
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(self.filtered_df))

        for row_idx, (_, row) in enumerate(self.filtered_df.iterrows()):
            for col_idx, column in enumerate(columns):
                value = row[column]
                item = QTableWidgetItem(display_text(value))
                item.setData(Qt.UserRole, row_idx)
                numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.notna(numeric):
                    item.setData(Qt.DisplayRole, float(numeric))
                self.table.setItem(row_idx, col_idx, item)

        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)
        if self.table.rowCount() > 0:
            self.table.selectRow(0)
        else:
            self.details.clear()
            self.component_pane.set_image(None)
            self.plot_pane.set_image(None)

    def update_summary(self):
        parts = [
            f"{len(self.filtered_df)} shown",
            f"{len(self.raw_df)} total",
            str(self.table_path) if self.table_path else "",
        ]
        if "file" in self.filtered_df.columns:
            parts.insert(2, f"{self.filtered_df['file'].nunique()} files")
        if "matrix_path" in self.filtered_df.columns:
            parts.insert(3, f"{self.filtered_df['matrix_path'].nunique()} matrices")
        self.summary_label.setText(" | ".join(part for part in parts if part))

    def selected_row(self):
        selected = self.table.selectedItems()
        if not selected:
            return None
        source_row = selected[0].data(Qt.UserRole)
        if source_row is None or source_row >= len(self.filtered_df):
            return None
        return self.filtered_df.iloc[int(source_row)]

    def on_table_selection(self):
        row = self.selected_row()
        if row is None:
            return
        self.details.setPlainText(self.format_row_details(row))
        plot_path = resolve_workspace_path(row.get("probability_plot", ""))
        self.plot_pane.set_image(plot_path)
        try:
            component_plot_path = ensure_component_grid_plot(self.results_root, row)
        except Exception as exc:
            component_plot_path = None
            self.details.append(f"\ncomponent_grid_error: {exc}")
        self.component_pane.set_image(component_plot_path)

    def format_row_details(self, row):
        preferred = [
            "rank_within_file",
            "rank_global",
            "final_score_method",
            "file",
            "band",
            "matrix",
            "components",
            "absolute_components",
            "n_components",
            "mean_score",
            "balanced_accuracy",
            "brier_score",
            "comp_set_score",
            "component_score_values",
            "probability_plot",
            "epoch_path",
            "matrix_path",
        ]
        ordered = [column for column in preferred if column in row.index]
        ordered.extend(column for column in row.index if column not in ordered)
        return "\n".join(f"{column}: {display_text(row[column])}" for column in ordered)

    def open_selected_plot(self):
        row = self.selected_row()
        if row is None:
            return
        plot_path = resolve_workspace_path(row.get("probability_plot", ""))
        if not plot_path or not plot_path.exists():
            QMessageBox.information(self, "No plot", "Selected row has no probability plot.")
            return
        open_path(plot_path)

    def open_component_plot(self):
        row = self.selected_row()
        if row is None:
            return
        try:
            plot_path = ensure_component_grid_plot(self.results_root, row)
        except Exception as exc:
            QMessageBox.warning(self, "Component plot error", str(exc))
            return
        if not plot_path or not plot_path.exists():
            QMessageBox.information(self, "No plot", "Selected row has no component plot.")
            return
        open_path(plot_path)


def main():
    app = QApplication(sys.argv)
    window = AlgorithmResultsViewer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
