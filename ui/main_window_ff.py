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

# from settings.settings_handler import SettingsHandler

class MainWindow(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        # Инициализация обработчика настроек
        # self.settings_handler = SettingsHandler()
        
        # Переменные для хранения текущих данных
        self._current_record = None  # Выбранный файл
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
        # Создаем папку folder1, если её нет (для примера)
        # if not os.path.exists(self.settings_handler.get_setting("FOLDER1_PATH")):
        #     os.makedirs(self.settings_handler.get_setting("FOLDER1_PATH"))
        #     # Создаем тестовые подпапки и файлы
        #     for subfolder in ["subfolder_1", "subfolder_2"]:
        #         sub_path = os.path.join(self.settings_handler.get_setting("FOLDER1_PATH"), subfolder)
        #         os.makedirs(sub_path, exist_ok=True)
        #         for i in range(3):
        #             with open(os.path.join(sub_path, f"file_{i+1}.txt"), 'w') as f:
        #                 f.write(f"Test file {i+1} in {subfolder}")
        
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
        
        self.folder_combo = QComboBox()
        self.folder_combo.addItem("-- Выберите папку --")
        
        self.files_list = QListWidget()
        self.files_list.setSelectionMode(QListWidget.SingleSelection)
        
        # ===== Группа настроек =====
        self.settings_group = QGroupBox("Настройки обработки")
        
        # 1) Тип ковариации
        self.cov_robust_radio = QRadioButton("Робастная ковариация")
        self.cov_ordinary_radio = QRadioButton("Обычная ковариация")
        self.cov_robust_radio.setChecked(
            self.settings_handler.get_setting("COVARIANCE_TYPE") == "robust"
        )
        
        # 2) Регуляризация (вкл/выкл)
        self.reg_checkbox = QCheckBox("Использовать регуляризацию")
        self.reg_checkbox.setChecked(
            self.settings_handler.get_setting("USE_REGULARIZATION")
        )
        
        # 3) Коэффициент регуляризации
        self.reg_spinbox = QDoubleSpinBox()
        self.reg_spinbox.setRange(0.0001, 1.0)
        self.reg_spinbox.setSingleStep(0.001)
        self.reg_spinbox.setDecimals(4)
        self.reg_spinbox.setValue(
            self.settings_handler.get_setting("REGULARIZATION_COEFF")
        )
        self.reg_spinbox.setEnabled(self.reg_checkbox.isChecked())
        
        # 4) Усреднять ковариации или считать одну
        self.avg_cov_radio = QRadioButton("Усреднять ковариации")
        self.single_cov_radio = QRadioButton("Считать одну ковариацию")
        self.avg_cov_radio.setChecked(
            self.settings_handler.get_setting("AVERAGE_COVARIANCES")
        )
        self.single_cov_radio.setChecked(
            not self.settings_handler.get_setting("AVERAGE_COVARIANCES")
        )
        
        # 5) Частотные диапазоны
        self.bands_group = QGroupBox("Частотные диапазоны (Гц)")
        self.bands_layout = QVBoxLayout()
        
        self.bands_inputs = []  # Список для хранения полей ввода диапазонов
        self.add_band_button = QPushButton("+ Добавить диапазон")
        self.remove_band_button = QPushButton("- Удалить последний")
        
        # Загружаем сохраненные диапазоны
        for low, high in self.settings_handler.get_setting("FREQUENCY_BANDS"):
            self.add_band_input(low, high)
        
        # Если нет ни одного диапазона, добавляем пустой
        if not self.bands_inputs:
            self.add_band_input(8.0, 12.0)
        
        # ===== Кнопки действий =====
        self.process_file_btn = QPushButton("Обработать файл")
        self.calc_csp_btn = QPushButton("Рассчитать CSP")
        self.train_classifier_btn = QPushButton("Обучить классификатор")
        self.show_components_btn = QPushButton("Показать компоненты")
        
        # ===== Статус/Информация =====
        self.status_label = QLabel("Готов к работе")
        self.status_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; }")
    
    def setup_layout(self):
        """Настройка компоновки"""
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Левая панель (выбор данных)
        left_panel = QVBoxLayout()
        
        # Компоновка для выбора папки
        folder_layout = QVBoxLayout()
        folder_layout.addWidget(QLabel("Папки в folder1:"))
        folder_layout.addWidget(self.folder_combo)
        folder_layout.addWidget(QLabel("Файлы в выбранной папке:"))
        folder_layout.addWidget(self.files_list)
        self.folder_group.setLayout(folder_layout)
        
        left_panel.addWidget(self.folder_group)
        left_panel.addStretch()
        
        # Правая панель (настройки и кнопки)
        right_panel = QVBoxLayout()
        
        # Настройки ковариации
        cov_layout = QVBoxLayout()
        cov_layout.addWidget(QLabel("Тип ковариации:"))
        cov_layout.addWidget(self.cov_robust_radio)
        cov_layout.addWidget(self.cov_ordinary_radio)
        
        # Регуляризация
        reg_layout = QVBoxLayout()
        reg_layout.addWidget(self.reg_checkbox)
        reg_h_layout = QHBoxLayout()
        reg_h_layout.addWidget(QLabel("Коэффициент:"))
        reg_h_layout.addWidget(self.reg_spinbox)
        reg_layout.addLayout(reg_h_layout)
        
        # Усреднение ковариаций
        avg_layout = QVBoxLayout()
        avg_layout.addWidget(QLabel("Режим ковариаций:"))
        avg_layout.addWidget(self.avg_cov_radio)
        avg_layout.addWidget(self.single_cov_radio)
        
        # Собираем общие настройки
        settings_layout = QVBoxLayout()
        settings_layout.addLayout(cov_layout)
        settings_layout.addLayout(reg_layout)
        settings_layout.addLayout(avg_layout)
        self.settings_group.setLayout(settings_layout)
        
        # Частотные диапазоны
        bands_controls = QHBoxLayout()
        bands_controls.addWidget(self.add_band_button)
        bands_controls.addWidget(self.remove_band_button)
        
        self.bands_group.setLayout(self.bands_layout)
        bands_main_layout = QVBoxLayout()
        bands_main_layout.addWidget(self.bands_group)
        bands_main_layout.addLayout(bands_controls)
        
        # Кнопки действий
        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self.process_file_btn)
        buttons_layout.addWidget(self.calc_csp_btn)
        buttons_layout.addWidget(self.train_classifier_btn)
        buttons_layout.addWidget(self.show_components_btn)
        
        # Собираем правую панель
        right_panel.addWidget(self.settings_group)
        right_panel.addLayout(bands_main_layout)
        right_panel.addLayout(buttons_layout)
        right_panel.addStretch()
        right_panel.addWidget(self.status_label)
        
        # Добавляем панели в главный layout
        main_layout.addLayout(left_panel, stretch=1)
        main_layout.addLayout(right_panel, stretch=2)
    
    def setup_connections(self):
        """Настройка сигналов и слотов"""
        
        # Выбор папки
        self.folder_combo.currentTextChanged.connect(self.on_folder_selected)
        
        # Выбор файла
        self.files_list.itemClicked.connect(self.on_file_selected)
        
        # Настройки
        self.cov_robust_radio.toggled.connect(self.on_settings_changed)
        self.cov_ordinary_radio.toggled.connect(self.on_settings_changed)
        self.reg_checkbox.stateChanged.connect(self.on_regularization_toggled)
        self.reg_spinbox.valueChanged.connect(self.on_settings_changed)
        self.avg_cov_radio.toggled.connect(self.on_settings_changed)
        self.single_cov_radio.toggled.connect(self.on_settings_changed)
        
        # Кнопки
        self.process_file_btn.clicked.connect(self.on_process_file)
        self.calc_csp_btn.clicked.connect(self.on_calc_csp)
        self.train_classifier_btn.clicked.connect(self.on_train_classifier)
        self.show_components_btn.clicked.connect(self.on_show_components)
        
        # Частотные диапазоны
        self.add_band_button.clicked.connect(self.on_add_band)
        self.remove_band_button.clicked.connect(self.on_remove_band)
        
        # Заполнение списка папок
        self.load_folders()
    
    def finalize(self):
        """Завершающие действия"""
        self.status_label.setText("Приложение запущено")
        self.show()
    
    # ==================== ЛОГИКА РАБОТЫ ====================
    
    def load_folders(self):
        """Загружает список папок из folder1"""
        folder1_path = self.settings_handler.get_setting("FOLDER1_PATH")
        
        if os.path.exists(folder1_path):
            self.folder_combo.clear()
            self.folder_combo.addItem("-- Выберите папку --")
            
            for item in os.listdir(folder1_path):
                item_path = os.path.join(folder1_path, item)
                if os.path.isdir(item_path):
                    self.folder_combo.addItem(item)
    
    def on_folder_selected(self, folder_name):
        """При выборе папки загружает список файлов"""
        if folder_name == "-- Выберите папку --" or not folder_name:
            self.files_list.clear()
            self._current_folder = None
            return
        
        self._current_folder = folder_name
        folder_path = os.path.join(
            self.settings_handler.get_setting("FOLDER1_PATH"),
            folder_name
        )
        
        self.files_list.clear()
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    self.files_list.addItem(file)
    
    def on_file_selected(self, item):
        """При выборе файла сохраняет его в self._current_record"""
        self._current_record = item.text()
        self.status_label.setText(f"Выбран файл: {self._current_record}")
    
    def on_settings_changed(self):
        """Обработчик изменения настроек"""
        # Определяем тип ковариации
        cov_type = "robust" if self.cov_robust_radio.isChecked() else "ordinary"
        self.settings_handler.set_setting("COVARIANCE_TYPE", cov_type)
        
        # Регуляризация
        self.settings_handler.set_setting(
            "USE_REGULARIZATION",
            self.reg_checkbox.isChecked()
        )
        self.settings_handler.set_setting(
            "REGULARIZATION_COEFF",
            self.reg_spinbox.value()
        )
        
        # Усреднение ковариаций
        self.settings_handler.set_setting(
            "AVERAGE_COVARIANCES",
            self.avg_cov_radio.isChecked()
        )
        
        # Частотные диапазоны
        bands = []
        for low_input, high_input in self.bands_inputs:
            try:
                low = float(low_input.text())
                high = float(high_input.text())
                bands.append((low, high))
            except ValueError:
                pass
        
        if bands:
            self.settings_handler.set_setting("FREQUENCY_BANDS", bands)
        
        self.status_label.setText("Настройки сохранены")
    
    def on_regularization_toggled(self):
        """Вкл/выкл поля коэффициента регуляризации"""
        self.reg_spinbox.setEnabled(self.reg_checkbox.isChecked())
        self.on_settings_changed()
    
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
        
        low_input.textChanged.connect(self.on_settings_changed)
        high_input.textChanged.connect(self.on_settings_changed)
        
        layout.addWidget(QLabel("От:"))
        layout.addWidget(low_input)
        layout.addWidget(QLabel("До:"))
        layout.addWidget(high_input)
        
        self.bands_layout.addWidget(container)
        self.bands_inputs.append((low_input, high_input))
    
    def on_add_band(self):
        """Добавляет новый частотный диапазон"""
        self.add_band_input(0.0, 0.0)
        self.on_settings_changed()
    
    def on_remove_band(self):
        """Удаляет последний частотный диапазон"""
        if self.bands_inputs:
            last_container = self.bands_layout.takeAt(self.bands_layout.count() - 1)
            if last_container and last_container.widget():
                last_container.widget().deleteLater()
            self.bands_inputs.pop()
            self.on_settings_changed()
    
    # ==================== ОБРАБОТЧИКИ КНОПОК ====================
    
    def on_process_file(self):
        """Обработка выбранного файла"""
        if not self._current_record:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите файл!")
            return
        
        # Здесь должна быть ваша логика обработки файла
        self.status_label.setText(f"Обработка файла: {self._current_record}")
        print(f"Обработка файла: {self._current_record}")
        print(f"Текущие настройки: {self.settings_handler.settings.__dict__}")
    
    def on_calc_csp(self):
        """Расчет CSP"""
        if not self._current_record:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите файл!")
            return
        
        self.status_label.setText("Расчет CSP...")
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
