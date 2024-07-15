import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import catboost as cb
from catboost import CatBoostClassifier
import re

import warnings 
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def fill_nan_categorical(df, column):
    # Sütundaki değerlerin dağılımını hesapla
    value_counts = df[column].value_counts(normalize=True)
    
    # NaN değerlerini oransal olarak doldur
    def fill_nan(value):
        if pd.isna(value):
            return np.random.choice(value_counts.index, p=value_counts.values)
        return value
    
    # Fonksiyonu sütuna uygula
    return df[column].apply(fill_nan)

def find_non_numeric_characters(value):
    if isinstance(value, str):
        # Sadece sayılar ve nokta dışında kalan karakterleri bul
        non_numeric_chars = re.findall(r'[^0-9.]', value)
        if non_numeric_chars:
            return ''.join(non_numeric_chars)
    return None

def clean_numeric(value):
    if isinstance(value, str):
        # Sadece sayılar ve nokta dışındaki karakterleri kaldır
        value = re.sub(r'[^0-9.]', '', value)
    try:
        return float(value)
    except ValueError:
        return np.nan
    
def clean_and_convert_to_months(value):
    if isinstance(value, str):
        # Yıl ve ay bilgilerini ay cinsine dönüştürme
        parts = value.split(' ')
        years = int(parts[0])
        months = int(parts[3])
        total_months = years * 12 + months
        return total_months
    return None

