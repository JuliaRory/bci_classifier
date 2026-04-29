from dataclasses import dataclass, field
from typing import List

@dataclass
class PreprocessingSettings:
    
    Fs: int = 1000
    do_filtering: bool = True
    low_freq: int = 5
    high_freq: int = 35
    baseline_ms: int = 500
    trial_dur_ms: int = 4000   # 4000  #8000
    start_shift_ms: int = 1000    # 1000   #0
    end_shift_ms: int = 0
    
    class1_photo: int = 2
    class2_photo: int = 3

@dataclass
class CSPSettings:
    use_regularization: bool = False

    robust_cov: bool = True
    covariance_type: str = "ohcov"  # "standard"
    alpha_reg: float = 0.01

    average_cov: bool = False 
    freq_bands: List[object] = field(default_factory=lambda: [(8.0, 12.0), (9, 13), (10, 14), (8, 15)])


@dataclass
class Settings:
    use_regularization: bool = False

    robust_cov: bool = True
    covariance_type: str = "ohcov"  # "standard"
    alpha_reg: float = 0.01

    average_cov: bool = False 
    freq_bands: List[object] = field(default_factory=lambda: [(8.0, 12.0), (13.0, 30.0)])

    folder_data: str = r"data"
    project: str = "pr_AstroSync"
    stage: str = "exp"
    session: str = "04_03 Artem"

    CSP: CSPSettings = field(default_factory=CSPSettings)
    preprocess: PreprocessingSettings = field(default_factory=PreprocessingSettings)





# class Settings:
#     """Класс с настройками по умолчанию"""
    
#     # 1) Тип ковариации
#     COVARIANCE_TYPE = "robust"  # "robust" или "ordinary"
    
#     # 2) Использовать регуляризацию
#     USE_REGULARIZATION = True
    
#     # 3) Коэффициент регуляризации
#     REGULARIZATION_COEFF = 0.01
    
#     # 4) Усреднять ковариации или считать одну
#     AVERAGE_COVARIANCES = True  # True - усреднять, False - одну
    
#     # 5) Частотные диапазоны: список кортежей (нижняя_частота, верхняя_частота)
#     FREQUENCY_BANDS = [(8.0, 12.0), (13.0, 30.0)]  # Альфа и бета диапазоны

#     # Дополнительные пути
#     FOLDER1_PATH = "folder1"  # Папка с подпапками