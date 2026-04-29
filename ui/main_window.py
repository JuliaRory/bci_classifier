import sys
import os
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
    QAbstractItemView, QSizePolicy, QSplitter, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import brier_score_loss


from settings.settings import Settings
from settings.settings_handler import SettingsHandler

from src.utils.ui_helpers import *
from src.utils.layout_utils import create_hbox, create_vbox
from src.utils.montage_processing import find_ch_idx, get_channel_names, get_topo_positions
from src.analysis.features import get_csp_features
from src.analysis.preprocessing import bandpass_filter
from src.analysis.csp_component_scores import get_selected_component_indices
from src.visualization.ROC_curve import plot_proba

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
CSP_COLORMAP = ListedColormap(VIRIDIS_BIG(np.linspace(0, 1, 15)))

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
        self.spin_box_trial_dur_ms = create_spin_box(0, 5000, s.trial_dur_ms)
        self.spin_box_start_shift_ms = create_spin_box(0, 5000, s.start_shift_ms)
        self.spin_box_class1_photo = create_spin_box(1, 3, s.class1_photo)
        self.spin_box_class2_photo = create_spin_box(1, 3, s.class2_photo)

        self.button_preprocess = create_button("Обработать")

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
        folder_layout.addWidget(self.button_preprocess)
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
        self.dataset_list.itemSelectionChanged.connect(self.refresh_csp_results)
        self.pair_scores_table.itemSelectionChanged.connect(self.on_pair_score_selected)
        
        # Кнопки
        self.button_preprocess.clicked.connect(self.on_process_file)
        self.button_calculate_csp.clicked.connect(self.on_calc_csp)
        self.button_show_csp_plot.clicked.connect(self.on_show_csp_components_plot)
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

    def _folder_selected_component_plots(self):
        s = self.settings
        return os.path.join(r"results", s.project, s.stage, s.session, "selected_components")

    def _folder_probability_plots(self):
        s = self.settings
        return os.path.join(r"results", s.project, s.stage, s.session, "PROBA_selected")

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

        df_best = self._ensure_ranking_score(df_best)

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
        df_cv = self._read_cv_scores()
        if df_cv.empty:
            return None

        if "pipeline" in df_cv.columns:
            df_cv = df_cv[df_cv["pipeline"] == "split_before_csp"].copy()

        required_columns = {"band", "sel_comp", "balanced accuracy", "brier score"}
        if df_cv.empty or not required_columns.issubset(df_cv.columns):
            return None

        if "ranking_score" in df_cv.columns:
            sort_columns = ["ranking_score"]
            ascending = [False]
        elif "component_assessment_score" in df_cv.columns:
            sort_columns = ["component_assessment_score", "balanced accuracy", "brier score"]
            ascending = [False, False, True]
        else:
            sort_columns = ["balanced accuracy", "brier score"]
            ascending = [False, True]

        df_cv = df_cv.sort_values(sort_columns, ascending=ascending, ignore_index=True)
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

        df_best = self._ensure_ranking_score(df_best)

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
        df_cv = self._pair_scores_view_df.copy() if self._pair_scores_view_df is not None else pd.DataFrame()
        if df_cv.empty:
            df_cv = self._read_cv_scores()
        if df_cv.empty:
            return pd.DataFrame()

        df_cv = self._attach_component_assessment_scores(df_cv)

        if "pipeline" in df_cv.columns:
            df_cv = df_cv[df_cv["pipeline"] == "split_before_csp"].copy()

        required_columns = {"band", "sel_comp", "balanced accuracy", "brier score"}
        if df_cv.empty or not required_columns.issubset(df_cv.columns):
            return pd.DataFrame()

        if "ranking_score" in df_cv.columns:
            sort_columns = ["ranking_score"]
            ascending = [False]
        elif "component_assessment_score" in df_cv.columns:
            sort_columns = ["component_assessment_score", "balanced accuracy", "brier score"]
            ascending = [False, False, True]
        else:
            sort_columns = ["balanced accuracy", "brier score"]
            ascending = [False, True]

        df_cv = df_cv.sort_values(sort_columns, ascending=ascending, ignore_index=True).head(top_n).copy()
        if "components" not in df_cv.columns:
            df_cv["components"] = df_cv["sel_comp"]
        return df_cv

    def _attach_component_assessment_scores(self, df_cv):
        if df_cv.empty or "component_assessment_score" in df_cv.columns:
            return self._ensure_ranking_score(df_cv.copy())

        df_components = self._read_component_tables()
        if df_components.empty:
            return df_cv

        df_groups = self._score_component_groups(df_components)
        if df_groups.empty:
            return df_cv

        df_groups = df_groups.copy()
        df_groups["sel_comp"] = df_groups["components"].apply(lambda value: str(tuple(value)))

        df_cv = df_cv.copy()
        if "sel_comp" in df_cv.columns:
            df_cv["sel_comp"] = df_cv["sel_comp"].apply(
                lambda value: str(tuple(ast.literal_eval(value))) if isinstance(value, str) else str(tuple(value))
            )

        df_cv = df_cv.merge(
            df_groups[["band", "sel_comp", "component_assessment_score"]],
            on=["band", "sel_comp"],
            how="left",
        )
        if "component_assessment_score" in df_cv.columns and "brier score" in df_cv.columns:
            df_cv["ranking_score"] = df_cv["component_assessment_score"] * (2 - df_cv["brier score"])
        return df_cv

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
                f"{self._row_components(row)}"
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

        band_text = str([int(x) if float(x).is_integer() else x for x in band])
        stems = [record_stem] if record_stem else self._selected_record_stems()
        candidates = [
            path
            for path in folder_csp.iterdir()
            if path.name.startswith(f"MATRIX_{band_text}_") and path.suffix == ".hdf"
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

        with File(matrix_path, "r") as h5f:
            spatial_filters = h5f["projForward"][:]

        features = self._build_probability_features(epochs, spatial_filters, band, components)
        classifier = LDA()
        classifier.fit(features, labels)
        y_proba = classifier.predict_proba(features)[:, 1]
        brier = brier_score_loss(labels, y_proba)

        fig = plot_proba(labels, y_proba)
        fig.suptitle(
            f"{self.settings.session}. {record_stem}. Band {band}. Components {tuple(components)}. "
            f"Brier score = {brier:.3f}"
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
            self._pair_scores_view_df = self._read_cv_scores().head(1000).copy()
            self.best_pair_label.setText(self._read_best_pair_text())
            self._update_best_components_plot()
            self._show_dataframe(self.pair_scores_table, self._pair_scores_view_df, max_rows=1000)
            self.pair_scores_table.blockSignals(False)
            return

        try:
            df_cv_scores = self._read_cv_scores()
            self._pair_scores_view_df = df_cv_scores.head(1000).copy()
            best_pair_text = self._read_best_pair_text()
        except Exception as exc:
            print(f"Не удалось загрузить cross-validation scores: {exc}")
            self.best_pair_label.setText("Subject -. Record -. Band -. Components -. Component assessment score: -. Balanced accuracy: -. Brier score: -. Ranking score: -.")
            self._draw_empty_best_components_plot("No component plot selected.")
            self._pair_scores_view_df = pd.DataFrame()
            self._show_dataframe(self.pair_scores_table, pd.DataFrame())
            self.pair_scores_table.blockSignals(False)
            return

        self.best_pair_label.setText(best_pair_text)
        self._update_best_components_plot()
        self._show_dataframe(self.pair_scores_table, self._pair_scores_view_df, max_rows=1000)
        self.pair_scores_table.blockSignals(False)

    def on_pair_score_selected(self):
        self._update_best_components_plot()

    def _normalize_series(self, series):
        series = pd.to_numeric(series, errors="coerce")
        span = series.max() - series.min()
        if pd.isna(span) or span == 0:
            return pd.Series([1.0] * len(series), index=series.index)
        return (series - series.min()) / span

    def _score_component_groups(self, df_components):
        output_columns = ["band", "components", "absolute_components", "component_assessment_score"]
        rows = []
        for band, df_band in df_components.groupby("band", sort=False):
            df_band = df_band.copy()
            contra_score = self._normalize_series(df_band["final_score_contra"])
            ipsi_score = self._normalize_series(df_band["final_score_ipsi"])
            df_band["component_score"] = pd.concat([contra_score, ipsi_score], axis=1).max(axis=1)
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
                        "component_assessment_score": float(sum(scores)),
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
        os.makedirs(folder_output, exist_ok=True)

        try:
            from scripts.calculate_csp import process_records_csp

            print("Расчет CSP с текущими настройками")
            print("config_csp:", config_csp)
            process_records_csp(folder_input, records, folder_output, config, config_csp)
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

        if not Path(self._folder_csp_plots()).exists():
            QMessageBox.warning(self, "CSP компоненты", "Папка с графиками CSP пока не найдена.")
            return

        row = self._read_best_pair_row()
        record_stem = self._record_stem_from_row(row)
        self._open_csp_component_plot([low, high], record_stem=record_stem)
    
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
