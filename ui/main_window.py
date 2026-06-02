import sys
import os
import json
from pathlib import Path
import subprocess
import ast

import numpy as np
import pandas as pd
from h5py import File
from matplotlib import colormaps as cm
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.pyplot import close
from mne.viz import plot_topomap
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QListWidget, QPushButton, QLabel, QGroupBox,
    QRadioButton, QCheckBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QMessageBox, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QSizePolicy, QSplitter, QScrollArea, QDialog,
    QDialogButtonBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import brier_score_loss
from scipy.signal import butter


from settings.settings import Settings
from settings.settings_handler import SettingsHandler

from src.utils.ui_helpers import *
from src.utils.layout_utils import create_hbox, create_vbox
from src.utils.montage_processing import find_ch_idx, get_channel_names, get_topo_positions
from src.analysis.features import get_csp_features
from src.analysis.preprocessing import bandpass_filter, detect_bad_epochs, read_good_epoch_mask
from src.analysis.csp_component_scores import build_component_assessment, get_selected_component_indices
from src.visualization.ROC_curve import plot_proba
from src.visualization.plot_csp_components import plot_10_csp_components

from scripts.create_dataset import process_records

COMPONENT_GROUP_TEMPLATES = [
    (0, -1),
    (0, 1, -1),
    (0, -2, -1),
    (0, 1, -2, -1),
]
BAD_CHANNELS = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]
MONTAGE_PATH = r"resources/mks64_standard.ced"
VIRIDIS_BIG = cm.get_cmap("jet")
CSP_COLORMAP = "jet" #ListedColormap(VIRIDIS_BIG(np.linspace(0, 1, 15)))


class EpochReviewDialog(QDialog):
    def __init__(self, epochs, labels, detection, initial_bad_mask, channel_names=None, fs=1000, parent=None):
        super().__init__(parent)
        self.epochs = np.asarray(epochs)
        self.labels = np.asarray(labels).reshape(-1)
        self.detection = detection
        self.bad_mask = np.asarray(initial_bad_mask, dtype=bool).copy()
        self.channel_names = list(channel_names or [])
        self.fs = fs

        self.setWindowTitle("Проверка плохих эпох")
        self.resize(1100, 720)
        self._setup_widgets()
        self._setup_layout()
        self._populate_table()
        self._update_summary()
        if self.table.rowCount() > 0:
            self.table.selectRow(0)

    def _setup_widgets(self):
        self.summary_label = QLabel()
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels(
            ["Плохая", "Эпоха", "Метка", "Score", "P2P", "RMS", "Max abs", "Flat %", "Причина"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)

        self.figure = Figure(figsize=(7.0, 4.2), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.button_mark_auto = QPushButton("Автовыбор")
        self.button_mark_all_good = QPushButton("Все хорошие")
        self.button_mark_all_bad = QPushButton("Все плохие")
        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Save).setText("Сохранить")
        self.buttons.button(QDialogButtonBox.Cancel).setText("Отмена")

        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._draw_selected_epoch)
        self.button_mark_auto.clicked.connect(self._mark_auto)
        self.button_mark_all_good.clicked.connect(lambda: self._set_all(False))
        self.button_mark_all_bad.clicked.connect(lambda: self._set_all(True))
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def _setup_layout(self):
        layout = QVBoxLayout(self)
        layout.addWidget(self.summary_label)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(self.canvas)
        splitter.setSizes([560, 540])
        layout.addWidget(splitter, stretch=1)
        controls = QHBoxLayout()
        controls.addWidget(self.button_mark_auto)
        controls.addWidget(self.button_mark_all_good)
        controls.addWidget(self.button_mark_all_bad)
        controls.addStretch()
        controls.addWidget(self.buttons)
        layout.addLayout(controls)

    def _populate_table(self):
        metrics = self.detection["metrics"]
        scores = self.detection["scores"]
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.epochs))
        for row in range(len(self.epochs)):
            check_item = QTableWidgetItem()
            check_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            check_item.setCheckState(Qt.Checked if self.bad_mask[row] else Qt.Unchecked)
            self.table.setItem(row, 0, check_item)

            values = [
                row,
                int(self.labels[row]) if row < len(self.labels) else "",
                scores[row],
                metrics["p2p"][row],
                metrics["rms"][row],
                metrics["max_abs"][row],
                metrics["flat_fraction"][row] * 100,
                metrics["reasons"][row],
            ]
            for col, value in enumerate(values, start=1):
                if isinstance(value, (float, np.floating)):
                    text = f"{value:.3g}"
                else:
                    text = str(value)
                self.table.setItem(row, col, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)

    def _on_item_changed(self, item):
        if item.column() != 0:
            return
        self.bad_mask[item.row()] = item.checkState() == Qt.Checked
        self._update_summary()
        self._draw_selected_epoch()

    def _set_all(self, is_bad):
        self.table.blockSignals(True)
        self.bad_mask[:] = is_bad
        for row in range(self.table.rowCount()):
            self.table.item(row, 0).setCheckState(Qt.Checked if is_bad else Qt.Unchecked)
        self.table.blockSignals(False)
        self._update_summary()
        self._draw_selected_epoch()

    def _mark_auto(self):
        auto_mask = np.asarray(self.detection["bad_mask"], dtype=bool)
        self.table.blockSignals(True)
        self.bad_mask = auto_mask.copy()
        for row in range(self.table.rowCount()):
            self.table.item(row, 0).setCheckState(Qt.Checked if self.bad_mask[row] else Qt.Unchecked)
        self.table.blockSignals(False)
        self._update_summary()
        self._draw_selected_epoch()

    def _update_summary(self):
        n_bad = int(self.bad_mask.sum())
        n_total = len(self.bad_mask)
        n_good = n_total - n_bad
        self.summary_label.setText(
            f"Плохие эпохи: {n_bad} / {n_total}. Хорошие эпохи: {n_good}. "
            "Отметьте строки для исключения и нажмите Save."
        )

    def _draw_selected_epoch(self):
        selected = self.table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        if row < 0 or row >= len(self.epochs):
            return

        epoch = self.epochs[row]
        n_channels = min(epoch.shape[1], 16)
        data = epoch[:, :n_channels]
        channel_scale = np.nanmedian(np.nanstd(data, axis=0))
        if not np.isfinite(channel_scale) or channel_scale <= np.finfo(float).eps:
            channel_scale = 1.0

        t = np.arange(epoch.shape[0]) / float(self.fs)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        offsets = np.arange(n_channels)[::-1] * 4.0
        for ch in range(n_channels):
            trace = (data[:, ch] - np.nanmedian(data[:, ch])) / channel_scale
            ax.plot(t, trace + offsets[ch], linewidth=0.8)

        y_labels = [
            self.channel_names[ch] if ch < len(self.channel_names) else str(ch)
            for ch in range(n_channels)
        ]
        ax.set_yticks(offsets)
        ax.set_yticklabels(y_labels)
        state = "BAD" if self.bad_mask[row] else "GOOD"
        ax.set_title(f"Эпоха {row}. Метка {self.labels[row] if row < len(self.labels) else '-'} . {state}")
        ax.set_xlabel("Time, s")
        ax.grid(True, alpha=0.2)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def selected_bad_mask(self):
        return self.bad_mask.copy()

class MainWindow(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()

        # Инициализация обработчика настроек
        self.settings = Settings()
        
        
        # Переменные для хранения текущих данных
        self._current_records = []   # Выбранный файл
        self._current_folder = None   # Выбранная папка
        self._pair_scores_view_df = pd.DataFrame()
        self._pair_scores_best_df = pd.DataFrame()
        
        # Вызов методов структуры
        self.init_state()
        self.setup_widgets()
        self.setup_layout()
        self.setup_connections()
        self.finalize()
    
    # ==================== СТРУКТУРНЫЕ МЕТОДЫ ====================
    
    def init_state(self):
        """Инициализация начального состояния"""

        # Устанавливаем заголовок и размер окна
        self.setWindowTitle("CSP Analysis Tool")
        self.setMinimumSize(800, 600)
        
        # Загружаем иконку (укажите свой путь или оставьте как есть)
        icon_path = "app_icon.ico"  # Замените на путь к вашей иконке
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
    
    def setup_widgets(self):
        """Создание всех виджетов"""
        
        # ===== Группа выбора папок и файлов =====
        self.folder_group = QGroupBox("Выбор данных")
        
        self.project_combo = QComboBox()
        self.project_combo.addItem("-- Выберите папку --")

        self.stage_combo = QComboBox()
        self.stage_combo.addItems(["test", "exp"])

        self.session_combo = QComboBox()
        self.session_combo.addItem("-- Выберите папку --")
        
        self.files_list = QListWidget()
        self.files_list.setSelectionMode(QListWidget.MultiSelection)

        self.dataset_list = QListWidget()
        self.dataset_list.setSelectionMode(QListWidget.MultiSelection)

        self.widgets_prepross()
        self.widgets_csp()
        self.widgets_results()
    
    def widgets_prepross(self):
        s = self.settings.preprocess
        self.spin_box_baseline_ms = create_spin_box(0, 5000, s.baseline_ms)
        self.spin_box_trial_dur_ms = create_spin_box(0, 50000, s.trial_dur_ms, step=100)
        self.spin_box_start_shift_ms = create_spin_box(0, 5000, s.start_shift_ms)
        self.spin_box_class1_photo = create_spin_box(1, 3, s.class1_photo)
        self.spin_box_class2_photo = create_spin_box(1, 3, s.class2_photo)

        self.button_preprocess = create_button("Обработать")
        self.button_review_bad_epochs = create_button("Проверить эпохи")

    def widgets_csp(self):
        # ===== Группа настроек =====
        self.settings_group = QGroupBox("Настройки обработки")
        
        s = self.settings.CSP

        self.combo_cov_type = create_combo_box(["ohcov", "standard"], curr_item=s.covariance_type)
        self.checkbox_regul = create_check_box(s.use_regularization, text="Использовать регуляризацию")
        self.spin_box_regul_alpha = create_spin_box(0.001, 1.0, s.alpha_reg, data_type="float")
        self.spin_box_regul_alpha.setEnabled(self.checkbox_regul.isChecked())
        self.checkbox_cov = create_check_box(s.average_cov, text="Усреднять ковариации")

        self.bands_group = QGroupBox("Частотные диапазоны (Гц)")
        self.bands_layout = QVBoxLayout()
        
        self.bands_inputs = []  # Список для хранения полей ввода диапазонов и кнопок просмотра
        self.add_band_button = QPushButton("+ Добавить диапазон")
        self.remove_band_button = QPushButton("- Удалить последний")
        
        # Загружаем сохраненные диапазоны
        for low, high in s.freq_bands:
            self.add_band_input(low, high)
        
        # Если нет ни одного диапазона, добавляем пустой
        if not self.bands_inputs:
            self.add_band_input(8.0, 12.0)
        
        self.button_calculate_csp = create_button("Рассчитать CSP")
        self.button_show_csp_plot = create_button("построить вероятности")

    def widgets_results(self):
        self.components_table = self._create_results_table()
        self.pair_scores_table = self._create_results_table()
        self.classifier_path_edit = QLineEdit()
        self.classifier_path_edit.setPlaceholderText("models/{project}/{stage}/{session}/feat{components}_{band}_{record_stem}.json")
        self._classifier_path_auto = True
        self.button_save_classifier = create_button("Сохранить классификатор")
        self.best_pair_label = QLabel("Subject -. Record -. Band -. Components -. Component assessment score: -. Balanced accuracy: -. Brier score: -. Ranking score: -.")
        self.best_pair_label.setWordWrap(True)
        self.best_pair_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.best_components_figure = Figure(figsize=(5.8, 3.4), dpi=100)
        self.best_components_canvas = FigureCanvas(self.best_components_figure)
        self.best_components_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.best_components_plot_scroll = QScrollArea()
        self.best_components_plot_scroll.setWidgetResizable(True)
        self.best_components_plot_scroll.setWidget(self.best_components_canvas)
        self.best_components_plot_scroll.setMinimumHeight(320)

    def setup_layout(self):
        """Настройка компоновки"""
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Левая панель (выбор данных)
        left_widget = QWidget()
        left_panel = QVBoxLayout()
        left_widget.setLayout(left_panel)
        
        # Компоновка для выбора папки
        pr_layout = create_hbox([QLabel("Проект:"), self.project_combo, QLabel("этап:"), self.stage_combo, QLabel("Сессия:"), self.session_combo])
        lists_layout = create_hbox([self.files_list, self.dataset_list])
        layout_preprocess = self.layout_preprocess()

        folder_layout = QVBoxLayout()
        folder_layout.addLayout(pr_layout)
        folder_layout.addWidget(QLabel("Файлы в выбранной папке:"))
        folder_layout.addLayout(lists_layout)
        folder_layout.addLayout(layout_preprocess)
        folder_layout.addLayout(create_hbox([self.button_preprocess, self.button_review_bad_epochs]))
        self.folder_group.setLayout(folder_layout)
        
        left_panel.addWidget(self.folder_group)
        tables_splitter = QSplitter(Qt.Vertical)
        tables_splitter.addWidget(self._table_section("CSP компоненты и оценки", self.components_table))
        tables_splitter.addWidget(self._table_section("Cross-validation scores", self.pair_scores_table))
        tables_splitter.setSizes([250, 250])
        left_panel.addWidget(tables_splitter, stretch=1)

        right_widget = QWidget()
        right_panel = self.layout_csp()
        right_widget.setLayout(right_panel)
        
        # Добавляем панели в главный layout
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([900, 500])
        main_layout.addWidget(main_splitter)

    def _table_section(self, title, table):
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(title))
        layout.addWidget(table)
        return section
    
    def layout_preprocess(self):
        preprocess_layout = QHBoxLayout()
        preprocess_layout.addLayout(create_hbox([QLabel("baseline:"), self.spin_box_baseline_ms]))
        preprocess_layout.addLayout(create_hbox([QLabel("trial_dur:"), self.spin_box_trial_dur_ms]))
        preprocess_layout.addLayout(create_hbox([QLabel("start_shift_ms:"), self.spin_box_start_shift_ms]))
        preprocess_layout.addLayout(create_hbox([QLabel("class1_photo:"), self.spin_box_class1_photo]))
        preprocess_layout.addLayout(create_hbox([QLabel("class2_photo:"), self.spin_box_class2_photo]))
        return preprocess_layout
    
    def layout_csp(self):
        right_panel = QVBoxLayout()

        # Собираем общие настройки
        settings_layout = QVBoxLayout()
        settings_layout.addLayout(create_hbox([QLabel("Тип ковариации:"), self.combo_cov_type, self.checkbox_cov]))
        settings_layout.addLayout(create_hbox([self.checkbox_regul, QLabel("коэффициент:"), self.spin_box_regul_alpha]))
        self.settings_group.setLayout(settings_layout)
        
        # Частотные диапазоны
        bands_controls = QHBoxLayout()
        bands_controls.addWidget(self.add_band_button)
        bands_controls.addWidget(self.remove_band_button)
        
        self.bands_group.setLayout(self.bands_layout)
        bands_main_layout = QVBoxLayout()
        bands_main_layout.addWidget(self.bands_group)
        bands_main_layout.addLayout(bands_controls)

        # Собираем правую панель
        right_panel.addWidget(self.settings_group)
        right_panel.addLayout(bands_main_layout)
        right_panel.addWidget(self.button_calculate_csp)
        right_panel.addWidget(self.button_show_csp_plot)
        right_panel.addLayout(create_hbox([QLabel("Путь классификатора:"), self.classifier_path_edit]))
        right_panel.addWidget(self.button_save_classifier)
        right_panel.addWidget(QLabel("Лучшая пара компонент-диапазон"))
        right_panel.addWidget(self.best_pair_label)
        right_panel.addWidget(self.best_components_plot_scroll, stretch=1)
        # right_panel.addWidget(self.status_label)

        

        return right_panel

    def setup_connections(self):
        """Настройка сигналов и слотов"""
        
        # Выбор папки
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        self.stage_combo.currentTextChanged.connect(self._on_stage_changed)
        self.session_combo.currentTextChanged.connect(self.on_folder_selected)
        
        # Выбор файла
        self.files_list.itemClicked.connect(self.on_file_selected)
        self.dataset_list.itemSelectionChanged.connect(self.on_dataset_selection_changed)
        self.pair_scores_table.itemSelectionChanged.connect(self.on_pair_score_selected)
        
        # Кнопки
        self.button_preprocess.clicked.connect(self.on_process_file)
        self.button_review_bad_epochs.clicked.connect(self.on_review_bad_epochs)
        self.button_calculate_csp.clicked.connect(self.on_calc_csp)
        self.button_show_csp_plot.clicked.connect(self.on_show_csp_components_plot)
        self.button_save_classifier.clicked.connect(self.on_save_classifier)
        self.classifier_path_edit.textEdited.connect(self._mark_classifier_path_manual)
        self.checkbox_regul.stateChanged.connect(
            lambda: self.spin_box_regul_alpha.setEnabled(self.checkbox_regul.isChecked())
        )
        # self.train_classifier_btn.clicked.connect(self.on_train_classifier)
        # self.show_components_btn.clicked.connect(self.on_show_components)
        
        # Частотные диапазоны
        self.add_band_button.clicked.connect(self.on_add_band)
        self.remove_band_button.clicked.connect(self.on_remove_band)
        
        # # Заполнение списка папок
        self.load_folders()
    
    def finalize(self):
        """Завершающие действия"""
        # self.status_label.setText("Приложение запущено")
        self.settings_handler = SettingsHandler(self, self.settings)

        # self._on_project_changed()
        self.show()
    
    # ==================== ЛОГИКА РАБОТЫ ====================

    def _create_results_table(self):
        table = QTableWidget()
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.horizontalHeader().setSectionsMovable(True)
        table.horizontalHeader().setStretchLastSection(False)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.verticalHeader().setVisible(False)
        table.setMinimumHeight(90)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return table
    
    def update_folder(self, folder, combo):
        if os.path.exists(folder):
            combo.clear()
            if len(os.listdir(folder)) == 0:
                combo.addItem("-- Выберите папку --")
            else:
                for item in os.listdir(folder):
                    item_path = os.path.join(folder, item)
                    if os.path.isdir(item_path):
                        combo.addItem(item)

    def _set_combo_text_if_present(self, combo, text):
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)
            return True
        return False

    def _populate_sessions_combo(self, project):
        self.session_combo.clear()
        sessions = self._available_sessions(project, self.settings.stage)
        if not sessions:
            self.session_combo.addItem("-- Выберите папку --")
            return
        for session in sessions:
            self.session_combo.addItem(session)

    def _available_sessions(self, project, stage):
        candidate_roots = [
            os.path.join(r"data", project, "raw", stage),
            os.path.join(r"data", project, "trans", stage),
            os.path.join(r"data", project, "features", "csp", stage),
            os.path.join(r"results", project, stage),
        ]
        sessions = set()
        for folder in candidate_roots:
            if not os.path.exists(folder):
                continue
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                if os.path.isdir(item_path):
                    sessions.add(item)
        return sorted(sessions)
        
    def load_folders(self):
        configured_project = self.settings.project
        configured_stage = self.settings.stage
        configured_session = self.settings.session

        self.project_combo.blockSignals(True)
        self.stage_combo.blockSignals(True)
        self.session_combo.blockSignals(True)

        self.update_folder(self.settings.folder_data, self.project_combo)
        self._set_combo_text_if_present(self.project_combo, configured_project)
        selected_project = self.project_combo.currentText()
        self._set_combo_text_if_present(self.stage_combo, configured_stage)
        self._populate_sessions_combo(selected_project)
        self._set_combo_text_if_present(self.session_combo, configured_session)

        self.settings.project = self.project_combo.currentText()
        self.settings.stage = self.stage_combo.currentText()
        self.settings.session = self.session_combo.currentText()

        self.project_combo.blockSignals(False)
        self.stage_combo.blockSignals(False)
        self.session_combo.blockSignals(False)

        self.on_folder_selected(self.settings.session)

    def _on_project_changed(self, project):
        self.settings.project = project
        self._populate_sessions_combo(project)
        if self.session_combo.count() > 0:
            self.on_folder_selected(self.session_combo.currentText())

    def _on_stage_changed(self, stage):
        self.settings.stage = stage
        self._populate_sessions_combo(self.settings.project)
        if self.session_combo.count() > 0:
            self.on_folder_selected(self.session_combo.currentText())

    def on_folder_selected(self, session):
        """При выборе папки загружает список файлов"""
        if not session or session.startswith("--"):
            return

        self.settings.session = session
        self.pair_scores_table.clearSelection()
        print("FOLDER SELECTED", session)

        self._current_folder = os.path.join(
            r"data",
            self.settings.project,
            "raw",
            self.settings.stage,
            session,
        )

        self._current_dataset_folder = os.path.join(
            r"data",
            self.settings.project,
            "trans",
            self.settings.stage,
            session,
        )

        self._update_list_widget(self.files_list, self._current_folder)
        self._update_list_widget(self.dataset_list, self._current_dataset_folder)
        self.refresh_csp_results()
        self._update_classifier_output_path()
    

    def _update_list_widget(self, list_widget, folder):
        list_widget.clear()
        if os.path.exists(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    list_widget.addItem(file)

    def _folder_csp(self):
        s = self.settings
        return os.path.join(r"data", s.project, "features", "csp", s.stage, s.session)

    def _folder_csp_plots(self):
        s = self.settings
        return os.path.join(r"results", s.project, s.stage, s.session, "CSP_components")

    def _folder_csp_plots_clear(self):
        s = self.settings
        return os.path.join(r"results", s.project, s.stage, s.session, "CSP_components") #_clear")

    def _folder_selected_component_plots(self):
        s = self.settings
        return os.path.join(r"results", s.project, s.stage, s.session, "selected_components")

    def _folder_probability_plots(self):
        s = self.settings
        return os.path.join(r"results", s.project, s.stage, s.session, "PROBA_selected")

    def _folder_final_probability_plots(self):
        s = self.settings
        return os.path.join(r"results", s.project, s.stage, s.session, "PROBA_final")

    def _folder_models(self):
        s = self.settings
        return os.path.join(r"models", s.project, s.stage, s.session)

    def _folder_cv_scores(self):
        s = self.settings
        return os.path.join(r"results", s.project, s.stage, s.session, "cv_scores")

    def _folder_autoselection(self):
        s = self.settings
        return os.path.join(r"results", s.project, s.stage, s.session, "autoselection")

    def _selected_dataset_records(self):
        return [item.text() for item in self.dataset_list.selectedItems()]

    def _selected_record_stems(self):
        stems = []
        for record in self._selected_dataset_records():
            stem = Path(record).stem
            if stem.startswith("EPOCHS_"):
                stem = stem[len("EPOCHS_"):]
            stems.append(stem)
        return stems

    def _record_stem_from_row(self, row):
        if row is None or "record" not in row.index:
            return None
        return Path(str(row["record"])).stem

    def _find_epochs_dataset_path(self, row=None):
        folder_epochs = Path(self._current_dataset_folder)
        if not folder_epochs.exists():
            return None

        record_stem = self._record_stem_from_row(row)
        if record_stem:
            candidate = folder_epochs / f"EPOCHS_{record_stem}.hdf"
            if candidate.exists():
                return candidate

        selected_records = self._selected_dataset_records()
        if selected_records:
            candidate = folder_epochs / selected_records[0]
            if candidate.exists():
                return candidate

        matches = sorted(folder_epochs.glob("EPOCHS_*.hdf"))
        return matches[0] if matches else None

    def _topomap_positions(self):
        labels = get_channel_names(MONTAGE_PATH)
        good_channel_indices = np.array(
            [find_ch_idx(ch, MONTAGE_PATH) for ch in labels if ch not in BAD_CHANNELS]
        )
        return get_topo_positions(MONTAGE_PATH)[good_channel_indices]

    def _read_component_tables(self):
        folder_csp = Path(self._folder_csp())
        if not folder_csp.exists():
            return pd.DataFrame()

        stems = self._selected_record_stems()
        files = []
        if stems:
            for stem in stems:
                files.extend(sorted(folder_csp.glob(f"DATAFRAME_*_{stem}.xlsx")))
        else:
            files = sorted(folder_csp.glob("DATAFRAME_*.xlsx"))

        if not files:
            return pd.DataFrame()

        return pd.concat([pd.read_excel(file) for file in files], ignore_index=True)

    def _read_cv_scores(self):
        folder_cv = Path(self._folder_cv_scores())
        if not folder_cv.exists():
            return pd.DataFrame()

        stems = self._selected_record_stems()
        files = []
        if stems:
            for stem in stems:
                files.extend(sorted(folder_cv.glob(f"{stem}.xlsx")))
        else:
            files = sorted(folder_cv.glob("*.xlsx"))

        if not files:
            return pd.DataFrame()

        df_scores = pd.concat([pd.read_excel(file) for file in files], ignore_index=True)
        return self._average_cv_scores_across_folds(df_scores)

    def _sort_pair_scores(self, df):
        if df is None or df.empty:
            return df

        if "ranking_score" in df.columns:
            sort_columns = ["ranking_score"]
            ascending = [False]
        elif "component_assessment_score" in df.columns:
            sort_columns = ["component_assessment_score", "balanced accuracy", "brier score"]
            ascending = [False, False, True]
        else:
            sort_columns = ["balanced accuracy", "brier score"]
            ascending = [False, True]

        return df.sort_values(sort_columns, ascending=ascending, ignore_index=True)

    def _prepare_pair_scores_view_df(self):
        df_cv = self._read_cv_scores()
        if df_cv.empty:
            return pd.DataFrame()

        df_cv = self._attach_component_assessment_scores(df_cv)
        if "pipeline" in df_cv.columns:
            df_cv = df_cv[df_cv["pipeline"] == "split_before_csp"].copy()

        if df_cv.empty:
            return pd.DataFrame()

        if "components" not in df_cv.columns and "sel_comp" in df_cv.columns:
            df_cv = df_cv.copy()
            df_cv["components"] = df_cv["sel_comp"]

        return df_cv.reset_index(drop=True)

    def _average_cv_scores_across_folds(self, df_scores):
        if df_scores.empty or "fold" not in df_scores.columns:
            return df_scores

        group_columns = [
            column
            for column in ["session", "record", "classifier", "band", "pipeline", "sel_comp"]
            if column in df_scores.columns
        ]
        numeric_columns = [
            column
            for column in df_scores.select_dtypes(include="number").columns
            if column != "fold"
        ]

        if not group_columns or not numeric_columns:
            return df_scores

        return (
            df_scores.groupby(group_columns, as_index=False)[numeric_columns]
            .mean()
            .sort_values(group_columns, ignore_index=True)
        )

    def _read_best_pair_row(self):
        row = self._read_selected_pair_row_from_table()
        if row is not None:
            return row

        if self._pair_scores_best_df is not None and not self._pair_scores_best_df.empty:
            row = self._pair_scores_best_df.iloc[0].copy()
            if "components" not in row.index and "sel_comp" in row.index:
                row["components"] = row["sel_comp"]
            return row

        row = self._read_best_pair_row_from_all_results()
        if row is not None:
            return row

        row = self._read_best_pair_row_from_cv_scores()
        if row is not None:
            return row

        if self._selected_record_stems():
            return None

        top3_path = Path(self._folder_autoselection()) / "top3_band_component_pairs.json"
        if not top3_path.exists():
            return None

        try:
            df_top = pd.read_json(top3_path)
        except Exception:
            return None

        if df_top.empty:
            return None

        return df_top.iloc[0]

    def _read_best_pair_row_from_all_results(self):
        all_results_path = Path(self._folder_autoselection()) / "all_autoselection_results.xlsx"
        if not all_results_path.exists():
            return None

        try:
            df_best = pd.read_excel(all_results_path, sheet_name="cv_split_before_csp")
        except Exception:
            return None

        if df_best.empty:
            return None

        df_best = self._attach_component_assessment_scores(df_best)

        stems = set(self._selected_record_stems())
        if stems and "record" in df_best.columns:
            df_best = df_best[
                df_best["record"].apply(lambda record: Path(str(record)).stem in stems)
            ]

        if df_best.empty:
            return None

        if "ranking_score" in df_best.columns:
            sort_columns = ["ranking_score"]
            ascending = [False]
        elif "component_assessment_score" in df_best.columns:
            sort_columns = ["component_assessment_score", "balanced accuracy", "brier score"]
            ascending = [False, False, True]
        else:
            sort_columns = ["balanced accuracy", "brier score"]
            ascending = [False, True]

        df_best = df_best.sort_values(
            sort_columns,
            ascending=ascending,
            ignore_index=True,
        )
        return df_best.iloc[0]

    def _read_best_pair_row_from_cv_scores(self):
        df_cv = self._sort_pair_scores(self._prepare_pair_scores_view_df())
        if df_cv.empty:
            return None
        row = df_cv.iloc[0].copy()
        row["components"] = row["sel_comp"]
        return row

    def _read_top_pair_rows_from_all_results(self, top_n=3):
        all_results_path = Path(self._folder_autoselection()) / "all_autoselection_results.xlsx"
        if not all_results_path.exists():
            return pd.DataFrame()

        try:
            df_best = pd.read_excel(all_results_path, sheet_name="cv_split_before_csp")
        except Exception:
            return pd.DataFrame()

        if df_best.empty:
            return pd.DataFrame()

        df_best = self._attach_component_assessment_scores(df_best)

        stems = set(self._selected_record_stems())
        if stems and "record" in df_best.columns:
            df_best = df_best[
                df_best["record"].apply(lambda record: Path(str(record)).stem in stems)
            ]

        if df_best.empty:
            return pd.DataFrame()

        if "ranking_score" in df_best.columns:
            sort_columns = ["ranking_score"]
            ascending = [False]
        elif "component_assessment_score" in df_best.columns:
            sort_columns = ["component_assessment_score", "balanced accuracy", "brier score"]
            ascending = [False, False, True]
        else:
            sort_columns = ["balanced accuracy", "brier score"]
            ascending = [False, True]

        df_best = df_best.sort_values(
            sort_columns,
            ascending=ascending,
            ignore_index=True,
        ).head(top_n)

        if "components" not in df_best.columns and "sel_comp" in df_best.columns:
            df_best = df_best.copy()
            df_best["components"] = df_best["sel_comp"]
        return df_best

    def _read_top_pair_rows_from_cv_scores(self, top_n=3):
        df_cv = self._pair_scores_best_df.copy() if self._pair_scores_best_df is not None else pd.DataFrame()
        if df_cv.empty:
            df_cv = self._sort_pair_scores(self._prepare_pair_scores_view_df())
        if df_cv.empty:
            return pd.DataFrame()

        required_columns = {"band", "sel_comp", "balanced accuracy", "brier score"}
        if df_cv.empty or not required_columns.issubset(df_cv.columns):
            return pd.DataFrame()
        df_cv = df_cv.head(top_n).copy()
        if "components" not in df_cv.columns:
            df_cv["components"] = df_cv["sel_comp"]
        return df_cv

    def _attach_component_assessment_scores(self, df_cv):
        if df_cv.empty:
            return self._ensure_ranking_score(df_cv.copy())

        df_components = self._read_component_tables()
        if df_components.empty:
            return self._ensure_ranking_score(df_cv.copy())

        component_scores_by_band = self._component_scores_by_band(df_components)
        if not component_scores_by_band:
            return self._ensure_ranking_score(df_cv.copy())

        df_cv = df_cv.copy()
        if "sel_comp" in df_cv.columns:
            df_cv["sel_comp"] = df_cv["sel_comp"].apply(
                lambda value: str(tuple(ast.literal_eval(value))) if isinstance(value, str) else str(tuple(value))
            )

        df_cv = df_cv.drop(columns=["component_assessment_score", "ranking_score"], errors="ignore")
        df_cv["component_assessment_score"] = df_cv.apply(
            lambda row: self._score_selected_components(row, component_scores_by_band),
            axis=1,
        )
        if "component_assessment_score" in df_cv.columns and "brier score" in df_cv.columns:
            df_cv["ranking_score"] = df_cv["component_assessment_score"] * (2 - df_cv["brier score"])
        return df_cv

    def _component_scores_by_band(self, df_components):
        scores_by_band = {}
        required_columns = {"band", "final_score_contra", "final_score_ipsi"}
        if df_components.empty or not required_columns.issubset(df_components.columns):
            return scores_by_band

        for band, df_band in df_components.groupby("band", sort=False):
            contra_score = pd.to_numeric(df_band["final_score_contra"], errors="coerce")
            ipsi_score = pd.to_numeric(df_band["final_score_ipsi"], errors="coerce")
            component_scores = contra_score.add(ipsi_score, fill_value=0).to_numpy()
            scores_by_band[band] = component_scores
            scores_by_band[str(band)] = component_scores

        return scores_by_band

    def _score_selected_components(self, row, component_scores_by_band):
        if "band" not in row.index:
            return np.nan

        component_scores = component_scores_by_band.get(row["band"])
        if component_scores is None:
            component_scores = component_scores_by_band.get(str(row["band"]))
        if component_scores is None:
            return np.nan

        components = self._coerce_components_value(self._row_components(row))
        if not components:
            return np.nan

        try:
            selected_scores = [component_scores[component] for component in components]
        except (IndexError, TypeError):
            return np.nan

        return float(np.mean(selected_scores))

    def _ensure_ranking_score(self, df):
        if df is None or df.empty:
            return df
        if "ranking_score" not in df.columns and {"component_assessment_score", "brier score"}.issubset(df.columns):
            df = df.copy()
            df["ranking_score"] = (
                pd.to_numeric(df["component_assessment_score"], errors="coerce")
                * (2 - pd.to_numeric(df["brier score"], errors="coerce"))
            )
        return df

    def _read_selected_pair_row_from_table(self):
        if self._pair_scores_view_df is None or self._pair_scores_view_df.empty:
            return None

        selected_items = self.pair_scores_table.selectedItems()
        if not selected_items:
            return None

        row_index = selected_items[0].row()
        if row_index < 0 or row_index >= len(self._pair_scores_view_df):
            return None

        row = self._pair_scores_view_df.iloc[row_index].copy()
        if "components" not in row.index and "sel_comp" in row.index:
            row["components"] = row["sel_comp"]
        return row

    def _read_best_pair_text(self):
        df_top = self._read_top_pair_rows_from_cv_scores(top_n=3)
        if df_top.empty:
            df_top = self._read_top_pair_rows_from_all_results(top_n=3)
        if df_top.empty:
            return "Subject -. Record -. Band -. Components -. Component assessment score: -. Balanced accuracy: -. Brier score: -. Ranking score: -."

        subject = df_top.iloc[0]["session"] if "session" in df_top.columns else "-"
        record = df_top.iloc[0]["record"] if "record" in df_top.columns else "-"
        lines = []
        for idx, (_, row) in enumerate(df_top.iterrows(), start=1):
            component_assessment_text = "-"
            if "component_assessment_score" in row.index and pd.notna(row["component_assessment_score"]):
                component_assessment_text = f"{float(row['component_assessment_score']):.3f}"
            ranking_score_text = "-"
            if "ranking_score" in row.index and pd.notna(row["ranking_score"]):
                ranking_score_text = f"{float(row['ranking_score']):.3f}"
            lines.append(
                f"{idx}. Band {row['band']} Hz. "
                f"{self._row_components(row)} "
                f"Comps: {component_assessment_text}. "
                f"Bal acc: {float(row['balanced accuracy']):.3f}. "
                f"Brier score: {float(row['brier score']):.3f}. "
                f"FINAL: {ranking_score_text}. "
            )
        return f"Subject {subject}. Record {record}.\n" + "\n".join(lines)

    def _row_components(self, row):
        if "components" in row.index:
            return row["components"]
        if "sel_comp" in row.index:
            return row["sel_comp"]
        return []

    def _coerce_band_value(self, band):
        if isinstance(band, str):
            try:
                return ast.literal_eval(band)
            except (SyntaxError, ValueError):
                return None
        return band

    def _coerce_components_value(self, components):
        if isinstance(components, str):
            try:
                return list(ast.literal_eval(components))
            except (SyntaxError, ValueError):
                return []
        return list(components)

    def _find_csp_matrix(self, band, record_stem=None):
        folder_csp = Path(self._folder_csp())
        if not folder_csp.exists() or band is None:
            return None

        band_variants = self._band_text_variants(band)
        stems = [record_stem] if record_stem else self._selected_record_stems()
        candidates = [
            path
            for path in folder_csp.iterdir()
            if path.suffix == ".hdf"
            and any(path.name.startswith(f"MATRIX_{band_text}_") for band_text in band_variants)
        ]

        if stems:
            candidates = [
                path
                for path in candidates
                if any(path.stem.endswith(f"_{stem}") for stem in stems)
            ]

        return sorted(candidates)[0] if candidates else None

    def _build_probability_features(self, epochs, spatial_filters, band, components):
        config = self._build_preprocess_config()
        epochs_band = np.array(
            [
                bandpass_filter(epoch, fs=config["Fs"], low=band[0], high=band[1])[0]
                for epoch in epochs
            ]
        )
        epochs_csp = np.array([epoch @ spatial_filters[:, components] for epoch in epochs_band])
        return get_csp_features(epochs_csp)

    def _mark_classifier_path_manual(self, *_):
        self._classifier_path_auto = False

    def _format_number_for_filename(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return str(value)
        return str(int(value)) if value.is_integer() else str(value).replace(".", "p")

    def _sanitize_filename_part(self, value):
        text = str(value).strip()
        for char in '<>:"/\\|?*':
            text = text.replace(char, "_")
        return text

    def _classifier_path_context(self, row=None):
        band = None
        components = []
        record_stem = None

        if row is not None:
            band = self._coerce_band_value(row["band"])
            components = self._coerce_components_value(self._row_components(row))
            record_stem = self._record_stem_from_row(row)

        if record_stem is None:
            selected_stems = self._selected_record_stems()
            record_stem = selected_stems[0] if selected_stems else "record"

        if band is None:
            band_text = "band"
        else:
            band_text = "-".join(self._format_number_for_filename(value) for value in band)

        if components:
            components_text = "_".join(str(component) for component in components)
        else:
            components_text = "components"

        return {
            "project": self._sanitize_filename_part(self.settings.project),
            "stage": self._sanitize_filename_part(self.settings.stage),
            "session": self._sanitize_filename_part(self.settings.session),
            "record_stem": self._sanitize_filename_part(record_stem),
            "band": self._sanitize_filename_part(band_text),
            "components": self._sanitize_filename_part(components_text),
        }

    def _default_classifier_output_path(self, row=None):
        template = getattr(
            self.settings,
            "classifier_output_path_template",
            r"models/{project}/{stage}/{session}/feat{components}_{band}_{record_stem}.json",
        )
        return template.format(**self._classifier_path_context(row))

    def _update_classifier_output_path(self, row=None, force=False):
        if not force and not self._classifier_path_auto and self.classifier_path_edit.text().strip():
            return

        if row is None:
            row = self._read_selected_pair_row_from_table()
        self.classifier_path_edit.setText(self._default_classifier_output_path(row))
        self._classifier_path_auto = True

    def _train_classifier_for_row(self, row):
        band = self._coerce_band_value(row["band"])
        components = self._coerce_components_value(self._row_components(row))
        if band is None or not components:
            raise ValueError("Не удалось прочитать band/components для выбранной строки.")

        dataset_path = self._find_epochs_dataset_path(row)
        if dataset_path is None or not dataset_path.exists():
            raise FileNotFoundError("Не найден EPOCHS-файл для выбранной записи.")

        record_stem = self._record_stem_from_row(row) or dataset_path.stem[len("EPOCHS_") :]
        print("record_stem", record_stem)
        matrix_path = self._find_csp_matrix(band, record_stem=record_stem)
        if matrix_path is None:
            raise FileNotFoundError(f"Не найдена CSP matrix для band {band} и record {record_stem}.")

        with File(dataset_path, "r") as h5f:
            epochs = h5f["epochs"][:]
            labels = h5f["labels"][:].squeeze().astype(int)
            good_epoch_mask = read_good_epoch_mask(h5f, len(epochs))
        epochs = epochs[good_epoch_mask]
        labels = labels[good_epoch_mask]

        with File(matrix_path, "r") as h5f:
            spatial_filters = h5f["projForward"][:]
            spatial_patterns = h5f["projInverse"][:]

        features = self._build_probability_features(epochs, spatial_filters, band, components)
        classifier = LDA()
        classifier.fit(features, labels)
        return classifier, spatial_filters, spatial_patterns, band, components, features, labels

    def _save_classifier_for_row(self, row, output_path):
        classifier, spatial_filters, _, band, components, _, _ = self._train_classifier_for_row(row)
        config = self._build_preprocess_config()

        sos_basic = butter(
            4,
            [1, 40],
            btype="bandpass",
            output="sos",
            fs=config["Fs"],
        )
        sos = butter(4, band, btype="bandpass", output="sos", fs=config["Fs"])

        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "spatialW": spatial_filters[:, components].tolist(),
            "sos_basic": sos_basic.tolist(),
            "sos": sos.tolist(),
            "band": [float(value) for value in band],
            "features_type": "csp",
            "Cref": None,
            "inv_sqrt": None,
            "w_lda": classifier.coef_[0].tolist(),
            "b_lda": float(classifier.intercept_[0]),
            "fs": config["Fs"],
            "n_components": len(components),
        }

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(model_data, file, indent=4)

        return output_path

    def _save_final_classifier_and_probability_plot(self, row, output_path, plot_output_path):
        classifier, spatial_filters, _, band, components, features, labels = self._train_classifier_for_row(row)
        config = self._build_preprocess_config()

        sos_basic = butter(
            4,
            [1, 40],
            btype="bandpass",
            output="sos",
            fs=config["Fs"],
        )
        sos = butter(4, band, btype="bandpass", output="sos", fs=config["Fs"])

        output_path = Path(output_path)
        plot_output_path = Path(plot_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_output_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "spatialW": spatial_filters[:, components].tolist(),
            "sos_basic": sos_basic.tolist(),
            "sos": sos.tolist(),
            "band": [float(value) for value in band],
            "features_type": "csp",
            "Cref": None,
            "inv_sqrt": None,
            "w_lda": classifier.coef_[0].tolist(),
            "b_lda": float(classifier.intercept_[0]),
            "fs": config["Fs"],
            "n_components": len(components),
        }

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(model_data, file, indent=4)

        y_proba = classifier.predict_proba(features)[:, 1]
        brier = brier_score_loss(labels, y_proba)
        fig = plot_proba(labels, y_proba)
        fig.suptitle(f"Brier score = {brier:.3f}")
        fig.savefig(plot_output_path, dpi=300, bbox_inches="tight")
        close(fig)
        return output_path, plot_output_path

    def _save_probability_plot_for_row(self, row):
        if row is None:
            raise ValueError("Нет выбранной пары band-components.")

        band = self._coerce_band_value(row["band"])
        components = self._coerce_components_value(self._row_components(row))
        if band is None or not components:
            raise ValueError("Не удалось прочитать band/components для построения вероятностей.")

        dataset_path = self._find_epochs_dataset_path(row)
        if dataset_path is None or not dataset_path.exists():
            raise FileNotFoundError("Не найден EPOCHS-файл для выбранной записи.")

        record_stem = self._record_stem_from_row(row) or dataset_path.stem[len("EPOCHS_") :]
        matrix_path = self._find_csp_matrix(band, record_stem=record_stem)
        if matrix_path is None:
            raise FileNotFoundError(f"Не найдена CSP matrix для band {band} и record {record_stem}.")

        with File(dataset_path, "r") as h5f:
            epochs = h5f["epochs"][:]
            labels = h5f["labels"][:].squeeze().astype(int)
            good_epoch_mask = read_good_epoch_mask(h5f, len(epochs))
        epochs = epochs[good_epoch_mask]
        labels = labels[good_epoch_mask]

        with File(matrix_path, "r") as h5f:
            spatial_filters = h5f["projForward"][:]

        features = self._build_probability_features(epochs, spatial_filters, band, components)
        classifier = LDA()
        classifier.fit(features, labels)
        y_proba = classifier.predict_proba(features)[:, 1]
        brier = brier_score_loss(labels, y_proba)

        fig = plot_proba(labels, y_proba)
        title_scores = f"Plot in-sample Brier = {brier:.3f}"
        if "brier score" in row.index and pd.notna(row["brier score"]):
            title_scores = f"CV Brier = {float(row['brier score']):.3f}. {title_scores}"
        fig.suptitle(
            f"{self.settings.session}. {record_stem}. Band {band}. Components {tuple(components)}. "
            f"{title_scores}"
        )

        band_text = str([int(x) if float(x).is_integer() else x for x in band])
        components_text = "_".join(str(component) for component in components)
        folder_output = Path(self._folder_probability_plots())
        folder_output.mkdir(parents=True, exist_ok=True)
        output_path = folder_output / f"{band_text}_{components_text}_{record_stem}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        close(fig)
        return output_path

    def _draw_empty_best_components_plot(self, message):
        self.best_components_figure.clear()
        ax = self.best_components_figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        ax.set_axis_off()
        self.best_components_canvas.draw_idle()

    def _update_best_components_plot(self):
        row = self._read_best_pair_row()
        if row is None:
            self._draw_empty_best_components_plot("No selected best pair found.")
            return

        band = self._coerce_band_value(row["band"])
        components = self._coerce_components_value(self._row_components(row))
        matrix_path = self._find_csp_matrix(band, self._record_stem_from_row(row))
        if matrix_path is None:
            self._draw_empty_best_components_plot(f"No CSP matrix found for band {row['band']}.")
            return

        try:
            with File(matrix_path, "r") as h5f:
                patterns = h5f["projInverse"][:]
                evals = h5f["evals"][:]
        except Exception as exc:
            self._draw_empty_best_components_plot(f"Could not load CSP matrix:\n{exc}")
            return

        n_components = patterns.shape[1]
        selected_pool = get_selected_component_indices(n_components)
        try:
            absolute_components = [selected_pool[component] for component in components]
        except IndexError:
            self._draw_empty_best_components_plot("Selected components are out of bounds for this CSP matrix.")
            return

        self._plot_selected_csp_components(
            patterns=patterns,
            evals=evals,
            absolute_components=absolute_components,
            relative_components=components,
            band=band,
        )

    def _plot_selected_csp_components(self, patterns, evals, absolute_components, relative_components, band):
        self.best_components_figure.clear()
        n_maps = len(absolute_components)
        figure_width = min(7.2, 2.2 + 1.25 * n_maps)
        self.best_components_figure.set_size_inches(figure_width, 3.4, forward=True)

        gs = self.best_components_figure.add_gridspec(
            1,
            n_maps + 1,
            width_ratios=[1.6] + [1.0] * n_maps,
            wspace=0.35,
        )

        ax_eigs = self.best_components_figure.add_subplot(gs[0, 0])
        ax_eigs.plot(evals, color="black", linewidth=1.2)
        ax_eigs.scatter(np.arange(len(evals)), evals, s=12, color="black")
        ax_eigs.scatter(absolute_components, evals[absolute_components], s=38, color="crimson", zorder=3)
        ax_eigs.set_ylim(0, 1)
        ax_eigs.set_title("Eigenvalues", fontsize=9)
        ax_eigs.tick_params(labelsize=8)

        xy = self._topomap_positions()
        selected_patterns = patterns[:, absolute_components]
        vmax = np.nanmax(np.abs(selected_patterns))
        vlim = (-vmax, vmax) if vmax > 0 else (None, None)

        image = None
        for i, (absolute_component, relative_component) in enumerate(
            zip(absolute_components, relative_components),
            start=1,
        ):
            ax_map = self.best_components_figure.add_subplot(gs[0, i])
            image, _ = plot_topomap(
                patterns[:, absolute_component],
                xy,
                axes=ax_map,
                show=False,
                contours=0,
                sphere=0.6,
                image_interp='cubic',
                extrapolate="head",
                cmap=CSP_COLORMAP,
                vlim=vlim,
            )
            ax_map.set_title(
                f"comp {relative_component}\nCSP #{absolute_component}",
                fontsize=9,
            )

        if image is not None:
            cbar = self.best_components_figure.colorbar(
                image,
                ax=self.best_components_figure.axes,
                fraction=0.035,
                pad=0.02,
            )
            cbar.ax.tick_params(labelsize=8)

        self.best_components_figure.suptitle(
            f"Band {band}. Selected components",
            fontsize=10,
        )
        self.best_components_figure.tight_layout(rect=[0, 0, 1, 0.92])
        self.best_components_canvas.draw_idle()

    def _show_dataframe(self, table, df, columns=None, max_rows=200):
        table.clear()

        if df is None or df.empty:
            table.setRowCount(0)
            table.setColumnCount(1)
            table.setHorizontalHeaderLabels(["Нет данных"])
            return

        if columns is not None:
            columns = [column for column in columns if column in df.columns]
            df = df[columns]

        df = df.head(max_rows).copy()
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        for col_idx, column in enumerate(df.columns):
            header_item = QTableWidgetItem(str(column))
            header_item.setToolTip(str(column))
            table.setHorizontalHeaderItem(col_idx, header_item)

        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, value in enumerate(row):
                if isinstance(value, float):
                    text = f"{value:.3f}"
                else:
                    text = str(value)
                table.setItem(row_idx, col_idx, QTableWidgetItem(text))

        table.resizeColumnsToContents()
        table.resizeRowsToContents()

    def refresh_csp_results(self):
        self.pair_scores_table.blockSignals(True)
        self.pair_scores_table.clearSelection()
        try:
            df_components = self._read_component_tables()
        except Exception as exc:
            print(f"Не удалось загрузить результаты CSP: {exc}")
            df_components = pd.DataFrame()

        self._show_dataframe(
            self.components_table,
            df_components,
            columns=[
                "record",
                "band",
                "n_comp",
                "evals",
                "eigscore",
                "score",
                "final_score",
                "final_score_contra",
                "final_score_ipsi",
            ],
        )

        if df_components.empty:
            self._pair_scores_view_df = self._prepare_pair_scores_view_df().head(1000).copy()
            self._pair_scores_best_df = self._sort_pair_scores(self._pair_scores_view_df.copy())
            self.best_pair_label.setText(self._read_best_pair_text())
            self._update_best_components_plot()
            self._show_dataframe(self.pair_scores_table, self._pair_scores_view_df, max_rows=1000)
            self.pair_scores_table.blockSignals(False)
            self._select_best_pair_row()
            self._update_classifier_output_path()
            return

        try:
            self._pair_scores_view_df = self._prepare_pair_scores_view_df().head(1000).copy()
            self._pair_scores_best_df = self._sort_pair_scores(self._pair_scores_view_df.copy())
            best_pair_text = self._read_best_pair_text()
        except Exception as exc:
            print(f"Не удалось загрузить cross-validation scores: {exc}")
            self.best_pair_label.setText("Subject -. Record -. Band -. Components -. Component assessment score: -. Balanced accuracy: -. Brier score: -. Ranking score: -.")
            self._draw_empty_best_components_plot("No component plot selected.")
            self._pair_scores_view_df = pd.DataFrame()
            self._pair_scores_best_df = pd.DataFrame()
            self._show_dataframe(self.pair_scores_table, pd.DataFrame())
            self.pair_scores_table.blockSignals(False)
            self._update_classifier_output_path()
            return

        self.best_pair_label.setText(best_pair_text)
        self._update_best_components_plot()
        self._show_dataframe(self.pair_scores_table, self._pair_scores_view_df, max_rows=1000)
        self.pair_scores_table.blockSignals(False)
        self._select_best_pair_row()
        self._update_classifier_output_path()

    def _select_best_pair_row(self):
        if self._pair_scores_view_df is None or self._pair_scores_view_df.empty:
            return
        if self._pair_scores_best_df is None or self._pair_scores_best_df.empty:
            return
        if self.pair_scores_table.rowCount() == 0:
            return

        best_row = self._pair_scores_best_df.iloc[0]
        match_columns = [
            column
            for column in ["session", "record", "classifier", "band", "pipeline", "sel_comp"]
            if column in self._pair_scores_view_df.columns and column in best_row.index
        ]
        if not match_columns:
            return

        for row_index, (_, row) in enumerate(self._pair_scores_view_df.iterrows()):
            if row_index >= self.pair_scores_table.rowCount():
                break
            if all(str(row[column]) == str(best_row[column]) for column in match_columns):
                self.pair_scores_table.selectRow(row_index)
                return

    def on_dataset_selection_changed(self):
        self.refresh_csp_results()

        if not self._selected_dataset_records():
            return

        row = self._read_best_pair_row()
        if row is None:
            print("Автосохранение классификатора: лучшая пара не найдена.")
            return

        self._update_classifier_output_path(row=row, force=True)
        output_path_text = self.classifier_path_edit.text().strip()

        try:
            output_path = self._save_classifier_for_row(row, output_path_text)
        except Exception as exc:
            print(f"Автосохранение классификатора не выполнено: {exc}")
            return

        self.classifier_path_edit.setText(str(output_path))
        print(f"Автосохранение классификатора: {output_path}")

    def on_pair_score_selected(self):
        self._update_best_components_plot()
        self._update_classifier_output_path()

    def _score_component_groups(self, df_components):
        output_columns = ["band", "components", "absolute_components", "component_assessment_score"]
        rows = []
        for band, df_band in df_components.groupby("band", sort=False):
            df_band = df_band.copy()
            contra_score = pd.to_numeric(df_band["final_score_contra"], errors="coerce")
            ipsi_score = pd.to_numeric(df_band["final_score_ipsi"], errors="coerce")
            df_band["component_score"] = contra_score.add(ipsi_score, fill_value=0)
            component_scores = df_band["component_score"].to_numpy()

            for components in COMPONENT_GROUP_TEMPLATES:
                try:
                    scores = [component_scores[component] for component in components]
                except IndexError:
                    continue

                rows.append(
                    {
                        "band": band,
                        "components": list(components),
                        "absolute_components": [int(df_band["n_comp"].iloc[component]) for component in components],
                        "component_assessment_score": float(np.mean(scores)),
                    }
                )

        if not rows:
            return pd.DataFrame(columns=output_columns)

        return pd.DataFrame(rows).sort_values(
            ["band", "component_assessment_score"],
            ascending=[True, False],
            ignore_index=True,
        )

    def _select_best_component_group_per_band(self, df_component_groups):
        if df_component_groups.empty:
            return df_component_groups.copy()

        return (
            df_component_groups.sort_values(
                ["band", "component_assessment_score"],
                ascending=[True, False],
            )
            .groupby("band", as_index=False, sort=False)
            .head(1)
            .reset_index(drop=True)
        )

    
    def on_file_selected(self, item):
        """При выборе файла сохраняет его в self._current_record"""
        selected_items = self.files_list.selectedItems()
        self._current_records = [item.text() for item in selected_items]

        if self._current_records:
            files_text = ", ".join(self._current_records)
            print(f"Выбрано файлов: {len(self._current_records)}: {files_text}")
        else:
            print("Файлы не выбраны")
        # self.status_label.setText(f"Выбран файл: {self._current_record}")
    
    
    def add_band_input(self, low=0.0, high=0.0):
        """Добавляет поля для ввода частотного диапазона"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        low_input = QLineEdit()
        low_input.setPlaceholderText("Нижняя частота")
        low_input.setText(str(low))
        
        high_input = QLineEdit()
        high_input.setPlaceholderText("Верхняя частота")
        high_input.setText(str(high))
        
        layout.addWidget(QLabel("От:"))
        layout.addWidget(low_input)
        layout.addWidget(QLabel("До:"))
        layout.addWidget(high_input)

        plot_button = QPushButton("...")
        plot_button.setFixedWidth(32)
        plot_button.setToolTip("Показать CSP компоненты для этого диапазона")
        plot_button.clicked.connect(
            lambda: self.on_show_band_csp_components_plot(low_input, high_input)
        )
        layout.addWidget(plot_button)
        
        self.bands_layout.addWidget(container)
        self.bands_inputs.append((low_input, high_input, plot_button))
    
    def on_add_band(self):
        """Добавляет новый частотный диапазон"""
        self.add_band_input(0.0, 0.0)
    
    def on_remove_band(self):
        """Удаляет последний частотный диапазон"""
        if self.bands_inputs:
            last_container = self.bands_layout.takeAt(self.bands_layout.count() - 1)
            if last_container and last_container.widget():
                last_container.widget().deleteLater()
            self.bands_inputs.pop()
    
    # ==================== ОБРАБОТЧИКИ КНОПОК ====================

    def _build_preprocess_config(self):
        s = self.settings.preprocess
        return {
            "Fs": s.Fs,
            "do_filtering": s.do_filtering,
            "low_freq": s.low_freq,
            "high_freq": s.high_freq,
            "baseline_ms": s.baseline_ms,
            "trial_dur_ms": s.trial_dur_ms,
            "start_shift_ms": s.start_shift_ms,
            "end_shift_ms": s.end_shift_ms,
            "epoch_len_ms": None,
            "epochs_step_ms": None,
            "idxs_keys": f"{s.class1_photo}-{s.class2_photo}",
        }

    def _selected_epoch_dataset_paths(self):
        if not self._current_dataset_folder:
            return []
        return [
            Path(self._current_dataset_folder) / item.text()
            for item in self.dataset_list.selectedItems()
        ]

    def _epoch_channel_names(self, n_channels):
        try:
            labels = list(get_channel_names(MONTAGE_PATH))
        except Exception:
            labels = []
        names = [channel for channel in labels if channel not in BAD_CHANNELS]
        if len(names) < n_channels:
            names.extend(str(index) for index in range(len(names), n_channels))
        return names[:n_channels]

    def _save_bad_epoch_review(self, dataset_path, bad_mask, detection):
        metrics = detection["metrics"]
        payload = {
            "auto_bad_epochs": int(np.asarray(detection["bad_mask"], dtype=bool).sum()),
            "confirmed_bad_epochs": int(np.asarray(bad_mask, dtype=bool).sum()),
            "total_epochs": int(len(bad_mask)),
            "reasons": list(metrics["reasons"]),
        }
        datasets = {
            "bad_epoch_mask": np.asarray(bad_mask, dtype=bool),
            "epoch_quality_scores": np.asarray(detection["scores"], dtype=float),
            "epoch_quality_p2p": np.asarray(metrics["p2p"], dtype=float),
            "epoch_quality_rms": np.asarray(metrics["rms"], dtype=float),
            "epoch_quality_max_abs": np.asarray(metrics["max_abs"], dtype=float),
            "epoch_quality_flat_fraction": np.asarray(metrics["flat_fraction"], dtype=float),
        }

        with File(dataset_path, "r+") as h5f:
            for name in list(datasets) + ["epoch_quality_metadata"]:
                if name in h5f:
                    del h5f[name]
            for name, values in datasets.items():
                h5f.create_dataset(name, data=values)
            h5f.create_dataset("epoch_quality_metadata", data=json.dumps(payload, ensure_ascii=False))

    def _review_epoch_dataset(self, dataset_path):
        with File(dataset_path, "r") as h5f:
            epochs = h5f["epochs"][:]
            labels = h5f["labels"][:].squeeze().astype(int)
            if "bad_epoch_mask" in h5f:
                initial_bad_mask = np.asarray(h5f["bad_epoch_mask"][:], dtype=bool)
            else:
                initial_bad_mask = None

        detection = detect_bad_epochs(epochs)
        if initial_bad_mask is None or len(initial_bad_mask) != len(epochs):
            initial_bad_mask = detection["bad_mask"]

        dialog = EpochReviewDialog(
            epochs=epochs,
            labels=labels,
            detection=detection,
            initial_bad_mask=initial_bad_mask,
            channel_names=self._epoch_channel_names(epochs.shape[2]),
            fs=self.settings.preprocess.Fs,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return False

        bad_mask = dialog.selected_bad_mask()
        if bad_mask.all():
            QMessageBox.warning(
                self,
                "Проверка эпох",
                f"Все эпохи отмечены как плохие: {dataset_path.name}. Маска не сохранена.",
            )
            return False

        self._save_bad_epoch_review(dataset_path, bad_mask, detection)
        print(f"Bad epochs saved -> {dataset_path}: {int(bad_mask.sum())} / {len(bad_mask)}")
        return True

    def on_review_bad_epochs(self):
        dataset_paths = self._selected_epoch_dataset_paths()
        if not dataset_paths:
            QMessageBox.warning(
                self,
                "Проверка эпох",
                "Сначала выберите хотя бы один EPOCHS-файл в списке обработанных dataset-файлов.",
            )
            return

        saved = 0
        for dataset_path in dataset_paths:
            if not dataset_path.exists():
                QMessageBox.warning(self, "Проверка эпох", f"Файл не найден: {dataset_path}")
                continue
            try:
                if self._review_epoch_dataset(dataset_path):
                    saved += 1
            except Exception as exc:
                QMessageBox.critical(self, "Ошибка проверки эпох", f"{dataset_path.name}\n{exc}")
                raise

        if saved:
            self.refresh_csp_results()
            QMessageBox.information(self, "Проверка эпох", f"Сохранена разметка для файлов: {saved}")

    def _read_csp_bands(self):
        bands = []
        for low_input, high_input, _ in self.bands_inputs:
            low, high = self._read_single_band(low_input, high_input)
            if low is None and high is None:
                continue

            bands.append([low, high])

        if not bands:
            raise ValueError("Добавьте хотя бы один частотный диапазон для CSP.")

        self.settings.CSP.freq_bands = bands
        return bands

    def _read_single_band(self, low_input, high_input):
        low_text = low_input.text().strip().replace(",", ".")
        high_text = high_input.text().strip().replace(",", ".")
        if not low_text and not high_text:
            return None, None

        try:
            low = float(low_text)
            high = float(high_text)
        except ValueError:
            raise ValueError("Частотные диапазоны должны быть числами.")

        if low <= 0 or high <= 0 or low >= high:
            raise ValueError("Для каждого диапазона должно выполняться: 0 < нижняя частота < верхняя частота.")

        return low, high

    def _build_csp_config(self):
        s = self.settings.CSP
        s.use_regularization = self.checkbox_regul.isChecked()
        s.alpha_reg = self.spin_box_regul_alpha.value()
        s.average_cov = self.checkbox_cov.isChecked()
        s.covariance_type = self.combo_cov_type.currentText()
        s.robust_cov = s.covariance_type == "ohcov"

        return {
            "bands": self._read_csp_bands(),
            "robust": s.robust_cov,
            "concat": not s.average_cov,
            "regularization": s.use_regularization,
            "alpha": s.alpha_reg,
        }

    def _build_cv_config(self):
        feature_groups = sorted(COMPONENT_GROUP_TEMPLATES, key=lambda group: (len(group), group))
        return {
            "n_splits": 3,
            "test_size": 5,
            "feature_groups": feature_groups,
            "classifier": "lda",
        }

    def _build_cv_config_for_record(self, record_name):
        config_cv = self._build_cv_config()
        config_cv["feature_groups"] = self._feature_groups_from_component_scores(record_name)
        return config_cv

    def _feature_groups_from_component_scores(self, record_name):
        record_stem = Path(record_name).stem
        if record_stem.startswith("EPOCHS_"):
            record_stem = record_stem[len("EPOCHS_"):]

        groups = {
            (0, -1),
            (0, 1),
            (0, 1, -1),
            (0, -2, -1),
            (0, 1, -2, -1),
        }
        assessment_files = sorted(Path(self._folder_csp()).glob(f"DATAFRAME_*_{record_stem}.xlsx"))
        for assessment_file in assessment_files:
            try:
                df = pd.read_excel(assessment_file)
            except Exception:
                continue
            if "n_comp" not in df.columns or "final_score" not in df.columns:
                continue
            component_2_rows = df[df["n_comp"].astype(int) == 2]
            if not component_2_rows.empty and float(component_2_rows["final_score"].max()) > 3.0:
                groups.add((0, 2, -1))
                groups.add((0, 1, 2, -1))

        return sorted(groups, key=lambda group: (len(group), group))

    def _save_fair_cv_summary(self):
        folder_cv = Path(self._folder_cv_scores())
        cv_files = sorted(path for path in folder_cv.glob("*.xlsx") if path.name != "fair_cv_summary.xlsx")
        rows = []
        for cv_file in cv_files:
            try:
                df = pd.read_excel(cv_file)
            except Exception:
                continue
            required_columns = {"pipeline", "session", "record", "classifier", "band", "sel_comp"}
            if not required_columns.issubset(df.columns):
                continue
            df_fair = df[df["pipeline"] == "split_before_csp"].copy()
            if df_fair.empty:
                continue
            rows.append(
                df_fair.groupby(["session", "record", "classifier", "band", "sel_comp"], as_index=False)
                .agg(
                    folds=("fold", "nunique"),
                    balanced_accuracy_mean=("balanced accuracy", "mean"),
                    balanced_accuracy_std=("balanced accuracy", "std"),
                    accuracy_mean=("accuracy", "mean"),
                    f1_mean=("f1", "mean"),
                    recall_mean=("recall", "mean"),
                    precision_mean=("precision", "mean"),
                    roc_auc_mean=("roc-auc", "mean"),
                    brier_score_mean=("brier score", "mean"),
                    log_loss_mean=("log loss", "mean"),
                )
            )

        if not rows:
            return None

        df_summary = pd.concat(rows, ignore_index=True).sort_values(
            by=["session", "record", "balanced_accuracy_mean", "brier_score_mean"],
            ascending=[True, True, False, True],
        )
        output_path = folder_cv / "fair_cv_summary.xlsx"
        df_summary.to_excel(output_path, index=False)
        print("output file ->", output_path)
        return output_path

    def _top_final_model_rows(self, record_name, top_n=3):
        record_stem = Path(record_name).stem
        if record_stem.startswith("EPOCHS_"):
            record_stem = record_stem[len("EPOCHS_"):]
        cv_scores_path = Path(self._folder_cv_scores()) / f"{record_stem}.xlsx"
        if not cv_scores_path.exists():
            return pd.DataFrame()

        df = pd.read_excel(cv_scores_path)
        record_value = Path(record_name[len("EPOCHS_") :]).name if record_name.startswith("EPOCHS_") else record_name
        if "pipeline" in df.columns:
            df = df[df["pipeline"] == "split_before_csp"].copy()
        if "record" in df.columns:
            df = df[df["record"] == record_value].copy()
        if df.empty:
            return df

        summary = self._average_cv_scores_across_folds(df)
        return summary.sort_values(
            ["balanced accuracy", "brier score"],
            ascending=[False, True],
            ignore_index=True,
        ).head(top_n)

    def _save_final_models_for_records(self, records, top_n=3):
        saved_paths = []
        folder_models = Path(self._folder_models())
        folder_proba = Path(self._folder_final_probability_plots())
        folder_models.mkdir(parents=True, exist_ok=True)
        folder_proba.mkdir(parents=True, exist_ok=True)

        for record_name in records:
            top_rows = self._top_final_model_rows(record_name, top_n=top_n)
            if top_rows.empty:
                print(f"Финальные модели: нет CV-строк для {record_name}")
                continue
            record_stem = Path(record_name).stem
            if record_stem.startswith("EPOCHS_"):
                record_stem = record_stem[len("EPOCHS_"):]
            for rank, (_, row) in enumerate(top_rows.iterrows(), start=1):
                row = row.copy()
                if "components" not in row.index and "sel_comp" in row.index:
                    row["components"] = row["sel_comp"]
                band = self._coerce_band_value(row["band"])
                components = self._coerce_components_value(self._row_components(row))
                band_text = json.dumps([int(value) if float(value).is_integer() else value for value in band])
                output_name = f"rank{rank}_feat{tuple(components)}_{band_text}_{record_stem}.json"
                model_path = folder_models / output_name
                plot_path = folder_proba / f"{model_path.stem}.png"
                saved_model, saved_plot = self._save_final_classifier_and_probability_plot(row, model_path, plot_path)
                saved_paths.extend([saved_model, saved_plot])
                print("model ->", saved_model)
                print("probability plot ->", saved_plot)

        return saved_paths

    def _redraw_csp_component_images(self):
        folder_csp = Path(self._folder_csp())
        if not folder_csp.exists():
            return 0

        xy = self._topomap_positions()
        total = 0
        for matrix_path in sorted(folder_csp.glob("MATRIX_*.hdf")):
            try:
                with File(matrix_path, "r") as h5f:
                    proj_inverse = h5f["projInverse"][:]
                    evals = h5f["evals"][:]
                    metadata_raw = h5f["metadata_csp"][()] if "metadata_csp" in h5f else None
            except Exception as exc:
                print(f"Не удалось прочитать CSP matrix {matrix_path}: {exc}")
                continue

            metadata_csp = {}
            if metadata_raw is not None:
                if isinstance(metadata_raw, bytes):
                    metadata_raw = metadata_raw.decode("utf-8")
                try:
                    metadata_csp = json.loads(metadata_raw)
                except (TypeError, json.JSONDecodeError):
                    metadata_csp = {}

            band = metadata_csp.get("band")
            component_scores = build_component_assessment(proj_inverse, evals)
            filename = matrix_path.name[len("MATRIX_") :]
            if filename.endswith(".hdf"):
                filename = filename[:-4] + ".png"

            outputs = [
                (Path(self._folder_csp_plots_clear()) / filename, True),
                (Path(self._folder_csp_plots()) / filename, False),
            ]
            for output_path, same_vlim in outputs:
                fig = plot_10_csp_components(
                    abs(evals),
                    proj_inverse,
                    xy,
                    component_scores=component_scores,
                    same_vlim=same_vlim,
                )
                fig.suptitle(f"CSP: {band} Hz", fontsize=16)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output_path, dpi=300, bbox_inches="tight")
                close(fig)
                print("output file ->", output_path)
                total += 1
        return total

    def on_process_file(self):
        """Обработка выбранного файла"""
        if len(self._current_records) == 0:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите хотя бы один файл!")
            return
        config = self._build_preprocess_config()

        s = self.settings
        folder_datasets = os.path.join(r"data", s.project, "trans", s.stage, s.session)
        print(folder_datasets)
        os.makedirs(folder_datasets, exist_ok=True)
        process_records(self._current_folder, self._current_records, folder_datasets, config)

        self._update_list_widget(self.dataset_list, self._current_dataset_folder)

        print(f"Обработка файлов: {self._current_records}")
        
    
    def on_calc_csp(self):
        """Расчет CSP"""
        selected_items = self.dataset_list.selectedItems()
        records = [item.text() for item in selected_items]

        if len(records) == 0:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите хотя бы один обработанный dataset-файл!")
            return

        try:
            config = self._build_preprocess_config()
            config_csp = self._build_csp_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Ошибка настроек CSP", str(exc))
            return

        s = self.settings
        folder_input = self._current_dataset_folder
        folder_output = os.path.join(r"data", s.project, "features", "csp", s.stage, s.session)
        folder_cv_output = self._folder_cv_scores()
        os.makedirs(folder_output, exist_ok=True)
        os.makedirs(folder_cv_output, exist_ok=True)

        try:
            from scripts.calculate_csp import process_records_csp
            from scripts.cross_validated_test import process_records_cross_validated

            print("Расчет CSP с текущими настройками")
            print("config_csp:", config_csp)
            process_records_csp(folder_input, records, folder_output, config, config_csp)
            saved_csp_images = self._redraw_csp_component_images()
            print(f"CSP-картинки сохранены: {saved_csp_images}")

            print("Расчет cross-validation таблиц")
            for record in records:
                config_cv = self._build_cv_config_for_record(record)
                print(f"Feature groups for {record}: {config_cv['feature_groups']}")
                process_records_cross_validated(
                    folder_input,
                    [record],
                    folder_cv_output,
                    config,
                    config_csp,
                    config_cv,
                )

            self._save_fair_cv_summary()
            self._save_final_models_for_records(records, top_n=3)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка расчета CSP", str(exc))
            raise

        self.refresh_csp_results()
        QMessageBox.information(
            self,
            "CSP",
            f"CSP рассчитан для файлов: {len(records)}\nРезультаты сохранены в {folder_output}",
        )

    def _band_text_variants(self, band):
        low, high = band
        variants = {
            str([low, high]),
            f"[{low:g}, {high:g}]",
        }
        if float(low).is_integer() and float(high).is_integer():
            variants.add(str([int(low), int(high)]))
        return variants

    def _find_csp_component_plots(self, band=None, record_stem=None):
        folder_plots = Path(self._folder_csp_plots())
        if not folder_plots.exists():
            return []

        stems = [record_stem] if record_stem else self._selected_record_stems()
        plots = []
        if stems:
            for stem in stems:
                plots.extend(sorted(folder_plots.glob(f"*{stem}.png")))
        else:
            plots = sorted(folder_plots.glob("*.png"))

        if band is not None:
            band_variants = self._band_text_variants(band)
            plots = [
                plot
                for plot in plots
                if any(plot.name.startswith(f"{band_text}_") for band_text in band_variants)
            ]

        return plots

    def _find_csp_component_plots_clear(self, band=None, record_stem=None):
        folder_plots = Path(self._folder_csp_plots_clear())
        if not folder_plots.exists():
            return []

        stems = [record_stem] if record_stem else self._selected_record_stems()
        plots = []
        if stems:
            for stem in stems:
                plots.extend(sorted(folder_plots.glob(f"*{stem}.png")))
        else:
            plots = sorted(folder_plots.glob("*.png"))

        if band is not None:
            band_variants = self._band_text_variants(band)
            plots = [
                plot
                for plot in plots
                if any(plot.name.startswith(f"{band_text}_") for band_text in band_variants)
            ]

        return plots

    def _save_current_best_components_plot(self):
        row = self._read_best_pair_row()
        if row is None:
            return None

        band = self._coerce_band_value(row["band"])
        components = self._coerce_components_value(self._row_components(row))
        if band is None or not components:
            return None

        band_text = str([int(x) if float(x).is_integer() else x for x in band])
        record_stems = self._selected_record_stems()
        record_text = record_stems[0] if record_stems else "all_records"
        components_text = "_".join(str(component) for component in components)

        folder_output = Path(self._folder_selected_component_plots())
        folder_output.mkdir(parents=True, exist_ok=True)
        output_path = folder_output / f"{band_text}_{components_text}_{record_text}.png"
        self.best_components_figure.savefig(output_path, dpi=300, bbox_inches="tight")
        return output_path

    def _open_csp_component_plot(self, band=None, record_stem=None):
        plots = self._find_csp_component_plots(band, record_stem=record_stem)
        if not plots:
            QMessageBox.warning(self, "CSP компоненты", "Графики CSP для выбранных данных не найдены.")
            return

        try:
            os.startfile(str(plots[0]))
        except AttributeError:
            subprocess.Popen(["xdg-open", str(plots[0])])

    def _open_csp_component_plot_clear(self, band=None, record_stem=None):
        plots = self._find_csp_component_plots_clear(band, record_stem=record_stem)
        if not plots:
            QMessageBox.warning(self, "CSP компоненты", "Графики CSP для выбранных данных не найдены.")
            return

        try:
            os.startfile(str(plots[0]))
        except AttributeError:
            subprocess.Popen(["xdg-open", str(plots[0])])

    def on_show_csp_components_plot(self):
        row = self._read_best_pair_row()
        if row is None:
            QMessageBox.warning(self, "Вероятности", "Нет выбранной пары компонентов для построения вероятностей.")
            return

        try:
            output_path = self._save_probability_plot_for_row(row)
        except Exception as exc:
            QMessageBox.warning(self, "Вероятности", str(exc))
            return

        try:
            os.startfile(str(output_path))
        except AttributeError:
            subprocess.Popen(["xdg-open", str(output_path)])

    def on_show_band_csp_components_plot(self, low_input, high_input):
        try:
            low, high = self._read_single_band(low_input, high_input)
        except ValueError as exc:
            QMessageBox.warning(self, "Ошибка диапазона CSP", str(exc))
            return

        if low is None or high is None:
            QMessageBox.warning(self, "Ошибка диапазона CSP", "Заполните частотный диапазон.")
            return

        if not Path(self._folder_csp_plots_clear()).exists():
            QMessageBox.warning(self, "CSP компоненты", "Папка с графиками CSP пока не найдена.")
            return

        row = self._read_best_pair_row()
        record_stem = self._record_stem_from_row(row)
        self._open_csp_component_plot_clear([low, high], record_stem=record_stem)
    
    def on_save_classifier(self):
        row = self._read_selected_pair_row_from_table()
        if row is None:
            QMessageBox.warning(
                self,
                "Сохранение классификатора",
                "Выберите строку в таблице Cross-validation scores.",
            )
            return

        output_path_text = self.classifier_path_edit.text().strip()
        if not output_path_text:
            output_path_text = self._default_classifier_output_path(row)
            self.classifier_path_edit.setText(output_path_text)
            self._classifier_path_auto = True

        try:
            output_path = self._save_classifier_for_row(row, output_path_text)
        except Exception as exc:
            QMessageBox.warning(self, "Сохранение классификатора", str(exc))
            return

        self.classifier_path_edit.setText(str(output_path))
        QMessageBox.information(
            self,
            "Сохранение классификатора",
            f"Классификатор сохранен:\n{output_path}",
        )

    def on_train_classifier(self):
        """Обучение классификатора"""
        self.status_label.setText("Обучение классификатора...")
        # Здесь ваша логика обучения
        print("Обучение классификатора")
    
    def on_show_components(self):
        """Показать компоненты"""
        self.status_label.setText("Показ компонентов...")
        # Здесь ваша логика отображения компонентов
        print("Показ компонентов")
