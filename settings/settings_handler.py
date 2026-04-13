import json
import os
from .settings import Settings

class SettingsHandler:
    """Класс для управления настройками (чтение/запись)"""
    
    def __init__(self, ui, settings,  config_file="settings/config.json"):
        self.ui = ui
        self.config_file = config_file
        self.settings = settings
        # self.load_settings()
        self.setup_connections()
    
    def setup_connections(self):
        # projects and files
        self.ui.project_combo.currentTextChanged.connect(self.update_project_combo)
        self.ui.stage_combo.currentTextChanged.connect(self.update_stage_combo)
        self.ui.session_combo.currentTextChanged.connect(self.update_session_combo)

        # preprocess
        self.ui.spin_box_baseline_ms.valueChanged.connect(self.update_baseline)
        self.ui.spin_box_trial_dur_ms.valueChanged.connect(self.update_trial_dur)
        self.ui.spin_box_start_shift_ms.valueChanged.connect(self.update_start_shift)
        self.ui.spin_box_class1_photo.valueChanged.connect(self.update_class1)
        self.ui.spin_box_class2_photo.valueChanged.connect(self.updata_class2)

        # csp
        self.ui.combo_cov_type.currentTextChanged.connect(self.update_cov_type)
        self.ui.checkbox_regul.stateChanged.connect(self.update_regul)
        self.ui.spin_box_regul_alpha.valueChanged.connect(self.update_regul_alpha)
        self.ui.checkbox_cov.stateChanged.connect(self.update_average_cov)

    # CSP
    def update_cov_type(self, text):
        self.settings.CSP.covariance_type = text
    
    def update_regul(self):
        self.settings.CSP.use_regularization = self.ui.checkbox_regul.isChecked()
    
    def update_regul_alpha(self, value):
        self.settings.CSP.alpha_reg = value
    
    def update_average_cov(self):
        self.settings.CSP.average_cov = self.ui.checkbox_cov.isChecked()
    
    # PREPROCESS
    def update_baseline(self, value):
        self.settings.preprocess.baseline_ms = value

    def update_trial_dur(self, value):
        self.settings.preprocess.trial_dur_ms = value

    def update_start_shift(self, value):
        self.settings.preprocess.start_shift_ms = value
    
    def update_class1(self, value):
        self.settings.preprocess.class1_photo = value
    
    def updata_class2(self, value):
        self.settings.preprocess.class2_photo = value

    # FILES
    def update_project_combo(self, text):
        self.settings.project = text

    def update_stage_combo(self, text):
        self.settings.stage = text

    def update_session_combo(self, text):
        self.settings.session = text 
    
    def load_settings(self):
        """Загружает настройки из JSON файла, если он существует"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                self.settings.COVARIANCE_TYPE = data.get('covariance_type', self.settings.COVARIANCE_TYPE)
                self.settings.USE_REGULARIZATION = data.get('use_regularization', self.settings.USE_REGULARIZATION)
                self.settings.REGULARIZATION_COEFF = data.get('regularization_coeff', self.settings.REGULARIZATION_COEFF)
                self.settings.AVERAGE_COVARIANCES = data.get('average_covariances', self.settings.AVERAGE_COVARIANCES)
                
                # Преобразуем диапазоны обратно в кортежи
                bands = data.get('frequency_bands', self.settings.FREQUENCY_BANDS)
                self.settings.FREQUENCY_BANDS = [tuple(band) for band in bands]
                
                self.settings.FOLDER1_PATH = data.get('folder1_path', self.settings.FOLDER1_PATH)
                
            except Exception as e:
                print(f"Ошибка загрузки настроек: {e}")
    
    def save_settings(self):
        """Сохраняет текущие настройки в JSON файл"""
        try:
            # Создаем директорию, если её нет
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            data = {
                'covariance_type': self.settings.COVARIANCE_TYPE,
                'use_regularization': self.settings.USE_REGULARIZATION,
                'regularization_coeff': self.settings.REGULARIZATION_COEFF,
                'average_covariances': self.settings.AVERAGE_COVARIANCES,
                'frequency_bands': self.settings.FREQUENCY_BANDS,
                'folder1_path': self.settings.FOLDER1_PATH
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"Ошибка сохранения настроек: {e}")
    
    def get_setting(self, name):
        """Получить значение настройки"""
        return getattr(self.settings, name, None)
    
    def set_setting(self, name, value):
        """Установить значение настройки и сохранить"""
        if hasattr(self.settings, name):
            setattr(self.settings, name, value)
            self.save_settings()
            return True
        return False