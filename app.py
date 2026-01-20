import sys
import os
import re
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, Any

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

path_to_csv = 'hh.csv'

class PipelineContext:
    """
    Класс контекста, хранящий состояние данных в процессе прохождения по цепочке.
    """
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.df: Optional[pd.DataFrame] = None
        self.x_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None


class Handler(ABC):
    """
    Абстрактный класс обработчика, реализующий паттерн Chain of Responsibility.
    """
    def __init__(self):
        self._next_handler: Optional[Handler] = None

    def set_next(self, handler: 'Handler') -> 'Handler':
        """
        Устанавливает следующий обработчик в цепочке.
        Возвращает установленный обработчик для удобной настройки цепочки.
        """
        self._next_handler = handler
        return handler

    @abstractmethod
    def handle(self, context: PipelineContext) -> None:
        """
        Метод обработки данных.
        """
        if self._next_handler:
            self._next_handler.handle(context)


class DataLoaderHandler(Handler):
    """
    Загружает данные из CSV файла.
    """
    def handle(self, context: PipelineContext) -> None:
        """
        Читает CSV файл, создает DataFrame и передает управление дальше.
        """
        logging.info(f"Loading data from {context.input_path}...")
        try:
            context.df = pd.read_csv(context.input_path)
            logging.info(f"Data loaded successfully. Shape: {context.df.shape}")
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            sys.exit(1)
        
        super().handle(context)


class SalaryParserHandler(Handler):
    """
    Обрабатывает целевую переменную 'ЗП' (Зарплата).
    Очищает данные, конвертирует валюты в рубли.
    """
    
    CURRENCY_RATES = {
        'руб': 1.0, 'rub': 1.0, 'rur': 1.0,
        'usd': 90.0,
        'eur': 100.0,
        'kzt': 0.2,
        'грн': 2.5, 'uah': 2.5,
        'бел.руб': 28.0, 'byn': 28.0, 'bel': 28.0,
        'kgs': 1.0, 'som': 1.0,
        'сум': 0.007, 'uzs': 0.007,
        'azn': 53.0
    }

    def _convert_salary(self, salary_str: str) -> float:
        """
        Парсит строку зарплаты, выделяет число и валюту, конвертирует по курсу.
        Если валюта не найдена, считается рублями.
        """
        if pd.isna(salary_str):
            return np.nan
        
        salary_str = str(salary_str).lower().replace('\xa0', '').replace(' ', '')
        
        match = re.search(r'(\d+)', salary_str)
        if not match:
            return np.nan
        
        value = float(match.group(1))
        
        rate = 1.0
        for currency, currency_rate in self.CURRENCY_RATES.items():
            if currency in salary_str:
                rate = currency_rate
                break
        
        return value * rate

    def handle(self, context: PipelineContext) -> None:
        """
        Применяет парсинг зарплаты и удаляет пропуски в целевой переменной.
        """
        logging.info("Processing target variable (Salary)...")
        df = context.df
        
        df['salary_rub'] = df['ЗП'].apply(self._convert_salary)
        
        initial_count = len(df)
        df.dropna(subset=['salary_rub'], inplace=True)
        dropped_count = initial_count - len(df)
        
        logging.info(f"Salary parsed. Dropped {dropped_count} rows with invalid target.")
        
        super().handle(context)


class DemographicsHandler(Handler):
    """
    Обрабатывает столбец 'Пол, возраст'.
    Извлекает пол (0 - ж, 1 - м) и возраст (число).
    """
    def _parse_gender(self, data_str: str) -> int:
        """
        Возвращает:
        1 - Мужчина
        0 - Женщина
        """
        if pd.isna(data_str):
            return 0
        return 1 if 'мужчина' in str(data_str).lower() else 0
    
    def _parse_age(self, data_str: str) -> int:
        """
        Извлекает возраст используя регулярное выражение.
        Ищет паттерн: число + пробел + (год/года/лет).
        """
        if pd.isna(data_str):
            return 0
        
        data_str = str(data_str).lower()
        match = re.search(r'(\d+)\s+(?:год|лет|года)', data_str)
        
        if match:
            return int(match.group(1))
        return 0

    def handle(self, context: PipelineContext) -> None:
        """
        Создает новые столбцы 'gender' и 'age' из столбца 'Пол, возраст'.
        """
        logging.info("Processing demographics (Gender, Age)...")
        df = context.df
        
        df['gender'] = df['Пол, возраст'].apply(self._parse_gender)
        df['age'] = df['Пол, возраст'].apply(self._parse_age)
        
        super().handle(context)


class ExperienceHandler(Handler):
    """
    Обрабатывает столбец 'Опыт работы'.
    Переводит опыт в общее количество месяцев.
    """
    def _parse_experience(self, exp_str: str) -> int:
        """
        Вычисляет общий опыт работы в месяцах.
        Парсит строку на наличие лет и месяцев.
        Учитывает различные склонения (лет, год, года, month, year и т.д.).
        """
        if pd.isna(exp_str) or 'не указано' in str(exp_str).lower():
            return 0
        
        exp_str = str(exp_str).lower().replace('\xa0', ' ')
        
        years = 0
        months = 0
        
        year_match = re.search(r'(\d+)\s+(?:год|лет|года|year|years)', exp_str)
        if year_match:
            years = int(year_match.group(1))
            
        month_match = re.search(r'(\d+)\s+(?:месяц|месяца|месяцев|month|months)', exp_str)
        if month_match:
            months = int(month_match.group(1))
            
        return years * 12 + months

    def handle(self, context: PipelineContext) -> None:
        """
        Создает столбец 'experience_months'.
        """
        logging.info("Processing work experience...")
        context.df['experience_months'] = context.df['Опыт (двойное нажатие для полной версии)'].apply(self._parse_experience)
        super().handle(context)


class FeatureSelectionHandler(Handler):
    """
    Обрабатывает столбец 'Образование и ВУЗ' и формирует итоговые матрицы.
    Кодирует категориальные признаки.
    """
    def _parse_education_level(self, edu_str: str) -> int:
        """
        Определяет уровень образования на основе ключевых слов в строке.
        Приоритет (от высшего к низшему):
        4 - Кандидат наук / PhD
        3 - Высшее (включая бакалавра, магистра, специалиста)
        2 - Неоконченное высшее
        1 - Среднее специальное (колледж, техникум)
        0 - Среднее / не указано
        
        Важен порядок проверок: сначала ищем 'кандидат', затем 'неоконченное', затем 'высшее'.
        Это необходимо, так как фраза "Неоконченное высшее" содержит слово "высшее".
        """
        if pd.isna(edu_str):
            return 0
            
        edu_str = str(edu_str).lower()
        
        if any(keyword in edu_str for keyword in ['кандидат наук', 'candidate of science', 'phd']):
            return 4
            
        if any(keyword in edu_str for keyword in ['неоконченное высшее', 'incomplete higher']):
            return 2
            
        if any(keyword in edu_str for keyword in ['высшее', 'higher education', 'bachelor', 'master', 'magister', 'specialist', 'бакалавр', 'магистр', 'специалист']):
            return 3
            
        if any(keyword in edu_str for keyword in ['среднее специальное', 'secondary special', 'college', 'колледж', 'техникум']):
            return 1
            
        return 0

    def _parse_auto(self, auto_str: str) -> int:
        """
        Бинарное кодирование наличия авто.
        """
        if pd.isna(auto_str):
            return 0
        return 1 if 'имеется' in str(auto_str).lower() else 0

    def handle(self, context: PipelineContext) -> None:
        """
        Выделяет образование, авто, формирует X и y.
        """
        logging.info("Selecting features and encoding categories...")
        df = context.df
        
        df['education_level'] = df['Образование и ВУЗ'].apply(self._parse_education_level)
        df['has_auto'] = df['Авто'].apply(self._parse_auto)
        
        feature_cols = ['gender', 'age', 'experience_months', 'education_level', 'has_auto']
        
        x_data = df[feature_cols].values.astype(np.float32)
        y_data = df['salary_rub'].values.astype(np.float32)
        
        context.x_data = x_data
        context.y_data = y_data
        
        logging.info(f"Final feature matrix shape: {x_data.shape}")
        
        super().handle(context)


class NumpySaverHandler(Handler):
    """
    Сохраняет обработанные данные в .npy файлы рядом с исходным файлом.
    """
    def handle(self, context: PipelineContext) -> None:
        """
        Сохраняет x_data.npy и y_data.npy.
        """
        base_dir = os.path.dirname(os.path.abspath(context.input_path))
        x_path = os.path.join(base_dir, 'x_data.npy')
        y_path = os.path.join(base_dir, 'y_data.npy')
        
        logging.info(f"Saving data to {x_path} and {y_path}...")
        
        try:
            np.save(x_path, context.x_data)
            np.save(y_path, context.y_data)
            logging.info("Files saved successfully.")
        except Exception as e:
            logging.error(f"Error saving files: {e}")
            sys.exit(1)
            
        super().handle(context)


def build_pipeline() -> Handler:
    """
    Конструирует цепочку обработчиков.
    Порядок: Загрузка -> Зарплата -> Демография -> Опыт -> Фичи -> Сохранение.
    """
    loader = DataLoaderHandler()
    salary = SalaryParserHandler()
    demographics = DemographicsHandler()
    experience = ExperienceHandler()
    features = FeatureSelectionHandler()
    saver = NumpySaverHandler()
    
    loader.set_next(salary) \
          .set_next(demographics) \
          .set_next(experience) \
          .set_next(features) \
          .set_next(saver)
          
    return loader


def main():
    """
    Точка входа. Использует глобальную переменную path_to_csv.
    """
    # Проверка существования файла по глобальному пути
    if not os.path.exists(path_to_csv):
        logging.error(f"File not found: {path_to_csv}")
        sys.exit(1)
        
    # Инициализация контекста с глобальной переменной
    context = PipelineContext(path_to_csv)
    pipeline = build_pipeline()
    
    logging.info(f"Starting pipeline using file: {path_to_csv}...")
    pipeline.handle(context)
    logging.info("Pipeline completed.")


if __name__ == "__main__":
    main()
