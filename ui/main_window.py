import sys
import os
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QListWidget, QPushButton, QLabel, QGroupBox,
    QRadioButton, QCheckBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon


from settings.settings import Settings
from settings.settings_handler import SettingsHandler

from src.utils.ui_helpers import *
from src.utils.layout_utils import create_hbox, create_vbox

from scripts.create_dataset import process_records

class MainWindow(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()

        # Инициализация обработчика настроек
        self.settings = Settings()
        
        
        # Переменные для хранения текущих данных
        self._current_records = []   # Выбранный файл
        self._current_folder = None   # Выбранная папка
        
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
        self.stage_combo.setEnabled(False)
        self.stage_combo.addItems(["test", "exp"])

        self.session_combo = QComboBox()
        self.session_combo.addItem("-- Выберите папку --")
        
        self.files_list = QListWidget()
        self.files_list.setSelectionMode(QListWidget.MultiSelection)

        self.dataset_list = QListWidget()
        self.dataset_list.setSelectionMode(QListWidget.MultiSelection)

        self.widgets_prepross()
        self.widgets_csp()
    
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
        
        self.bands_inputs = []  # Список для хранения полей ввода диапазонов
        self.add_band_button = QPushButton("+ Добавить диапазон")
        self.remove_band_button = QPushButton("- Удалить последний")
        
        # Загружаем сохраненные диапазоны
        for low, high in s.freq_bands:
            self.add_band_input(low, high)
        
        # Если нет ни одного диапазона, добавляем пустой
        if not self.bands_inputs:
            self.add_band_input(8.0, 12.0)
        
        self.button_calculate_csp = create_button("Рассчитать CSP")

    def setup_layout(self):
        """Настройка компоновки"""
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Левая панель (выбор данных)
        left_panel = QVBoxLayout()
        
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
        left_panel.addStretch()

        right_panel = self.layout_csp()
        
        # Добавляем панели в главный layout
        main_layout.addLayout(left_panel, stretch=1)
        main_layout.addLayout(right_panel, stretch=2)
    
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
        right_panel.addStretch()
        # right_panel.addWidget(self.status_label)

        

        return right_panel

    def setup_connections(self):
        """Настройка сигналов и слотов"""
        
        # Выбор папки
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        # self.stage_combo.currentTextChanged.connect(self.on_folder_selected)
        self.session_combo.currentTextChanged.connect(self.on_folder_selected)
        
        # Выбор файла
        self.files_list.itemClicked.connect(self.on_file_selected)
        
        # Кнопки
        self.button_preprocess.clicked.connect(self.on_process_file)
        self.button_calculate_csp.clicked.connect(self.on_calc_csp)
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
        
    def load_folders(self):
        folder = self.settings.folder_data
        self.update_folder(folder, self.project_combo)
        
        folder_session = os.path.join(
            r"data",
            self.settings.project,
            "raw",
            self.settings.stage
        )
        print(folder_session)
        print(os.listdir(folder_session))
        self.update_folder(folder_session, self.session_combo)

    def _on_project_changed(self, project):
        folder = os.path.join(r"data", project, "raw", self.settings.stage)
        self.settings.project = project

        self.session_combo.clear()

        if os.path.exists(folder):
            for session in os.listdir(folder):
                dir_path = os.path.join(folder, session)
                if os.path.isdir(dir_path):
                    self.session_combo.addItem(session)

        # self.on_folder_selected(self.session_combo.currentText())

    def on_folder_selected(self, session):
        """При выборе папки загружает список файлов"""
        self.settings.session = session
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
    

    def _update_list_widget(self, list_widget, folder):
        list_widget.clear()
        if os.path.exists(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    list_widget.addItem(file)

    
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
        
        self.bands_layout.addWidget(container)
        self.bands_inputs.append((low_input, high_input))
    
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
    
    def on_process_file(self):
        """Обработка выбранного файла"""
        if len(self._current_records) == 0:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите хотя бы один файл!")
            return
        # create config 
        s = self.settings.preprocess
        config = {
            "Fs": 1000,
            "do_filtering": True, 
            "low_freq": 5, 
            "high_freq": 35,
            "baseline_ms": s.baseline_ms,
            "trial_dur_ms": s.trial_dur_ms,
            "start_shift_ms": s.start_shift_ms,
            "end_shift_ms": 0,
            "epoch_len_ms": None,
            "epochs_step_ms": None, 
            "idxs_keys": f"{s.class1_photo}-{s.class2_photo}" # "2-3" "1-2", 
        }

        s = self.settings
        folder_datasets = os.path.join(r"data", s.project, "trans", s.stage, s.session)
        print(folder_datasets)
        os.makedirs(folder_datasets, exist_ok=True)
        process_records(self._current_folder, self._current_records, folder_datasets, config)

        self._update_list_widget(self.dataset_list, self._current_dataset_folder)

        print(f"Обработка файлов: {self._current_records}")
        
    
    def on_calc_csp(self):
        """Расчет CSP"""
        if len(self._current_records) == 0:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите файл!")
            return
        
        # self.status_label.setText("Расчет CSP...")
        # Здесь ваша логика расчета CSP
        print("Расчет CSP с текущими настройками")
    
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

