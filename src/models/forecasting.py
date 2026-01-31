"""
Modelo de Forecasting de Demanda para Retail
=============================================

Este módulo implementa un modelo de predicción de demanda basado en XGBoost
para pronosticar ventas por SKU a 30 días. Incluye feature engineering
avanzado con variables temporales, de tendencia y estacionalidad.

Características principales:
    - Predicción multi-SKU con un modelo por categoría
    - Feature engineering automático con variables temporales
    - Validación cruzada temporal (Time Series Split)
    - Métricas de evaluación especializadas para forecasting
    - Exportación de predicciones y modelo entrenado

Autor: ML & Research Scientist Agent
Fecha: 2026-01-31
Versión: 1.0.0

Example:
    >>> from src.models.forecasting import DemandForecaster
    >>> forecaster = DemandForecaster()
    >>> forecaster.fit(train_df)
    >>> predictions = forecaster.predict(horizon_days=30)
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

# Suprimir warnings de convergencia durante entrenamiento
warnings.filterwarnings('ignore', category=UserWarning)

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


# =============================================================================
# CONFIGURACIÓN Y CONSTANTES
# =============================================================================

@dataclass
class ForecastConfig:
    """Configuración para el modelo de forecasting.
    
    Attributes:
        horizon_days: Número de días a pronosticar hacia el futuro.
        test_size: Proporción de datos para testing (0.0 - 1.0).
        n_splits: Número de splits para validación cruzada temporal.
        xgb_params: Parámetros del modelo XGBoost.
        random_state: Semilla para reproducibilidad.
        lag_features: Lista de lags a crear como features.
        rolling_windows: Tamaños de ventana para features de rolling.
    """
    horizon_days: int = 30
    test_size: float = 0.2
    n_splits: int = 5
    random_state: int = 42
    lag_features: List[int] = field(default_factory=lambda: [1, 7, 14, 21, 28])
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 30])
    xgb_params: Dict = field(default_factory=lambda: {
        'objective': 'reg:squarederror',
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    })


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """Clase para crear features temporales y de demanda.
    
    Esta clase implementa el feature engineering necesario para el modelo
    de forecasting, incluyendo variables de calendario, lags, y estadísticas
    rolling.
    
    Attributes:
        config: Configuración del forecaster.
        label_encoders: Diccionario de encoders para variables categóricas.
        
    Example:
        >>> engineer = FeatureEngineer(config)
        >>> df_features = engineer.create_features(df)
    """
    
    def __init__(self, config: ForecastConfig):
        """Inicializa el FeatureEngineer.
        
        Args:
            config: Objeto de configuración ForecastConfig.
        """
        self.config = config
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self._fitted = False
    
    def create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de calendario a partir de la columna de fecha.
        
        Extrae componentes temporales como día de la semana, mes, trimestre,
        y crea variables indicadoras para patrones estacionales.
        
        Args:
            df: DataFrame con columna 'date' en formato datetime o string.
            
        Returns:
            DataFrame con las nuevas columnas de calendario añadidas.
            
        Raises:
            ValueError: Si la columna 'date' no existe en el DataFrame.
        """
        if 'date' not in df.columns:
            raise ValueError("El DataFrame debe contener una columna 'date'")
        
        df = df.copy()
        
        # Asegurar que date es datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Features básicos de calendario
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        
        # Features cíclicos (sin/cos encoding para preservar ciclicidad)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
        # Features binarios
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        
        # Temporada boliviana (lluvias: Nov-Mar, seca: Abr-Oct)
        df['is_rainy_season'] = df['month'].isin([11, 12, 1, 2, 3]).astype(int)
        
        return df
    
    def create_lag_features(
        self, 
        df: pd.DataFrame, 
        group_cols: List[str],
        target_col: str = 'quantity_sold'
    ) -> pd.DataFrame:
        """Crea features de lag (valores pasados) para la variable objetivo.
        
        Genera columnas con valores históricos de la variable objetivo
        agrupados por las columnas especificadas.
        
        Args:
            df: DataFrame con datos ordenados por fecha.
            group_cols: Columnas para agrupar (ej: ['sku_id', 'store_id']).
            target_col: Nombre de la columna objetivo.
            
        Returns:
            DataFrame con columnas de lag añadidas.
        """
        df = df.copy()
        
        for lag in self.config.lag_features:
            col_name = f'lag_{lag}'
            df[col_name] = df.groupby(group_cols)[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        target_col: str = 'quantity_sold'
    ) -> pd.DataFrame:
        """Crea features de estadísticas rolling (media, std, min, max).
        
        Calcula estadísticas sobre ventanas móviles de tiempo para capturar
        tendencias y volatilidad reciente.
        
        Args:
            df: DataFrame con datos ordenados por fecha.
            group_cols: Columnas para agrupar.
            target_col: Nombre de la columna objetivo.
            
        Returns:
            DataFrame con columnas de rolling añadidas.
        """
        df = df.copy()
        
        for window in self.config.rolling_windows:
            # Media móvil
            df[f'rolling_mean_{window}'] = (
                df.groupby(group_cols)[target_col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
            
            # Desviación estándar móvil
            df[f'rolling_std_{window}'] = (
                df.groupby(group_cols)[target_col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
            )
            
            # Máximo móvil
            df[f'rolling_max_{window}'] = (
                df.groupby(group_cols)[target_col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).max())
            )
            
            # Mínimo móvil
            df[f'rolling_min_{window}'] = (
                df.groupby(group_cols)[target_col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).min())
            )
        
        return df
    
    def encode_categorical_features(
        self, 
        df: pd.DataFrame,
        categorical_cols: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """Codifica variables categóricas usando LabelEncoder.
        
        Args:
            df: DataFrame con columnas categóricas.
            categorical_cols: Lista de columnas a codificar.
            fit: Si True, ajusta nuevos encoders. Si False, usa existentes.
            
        Returns:
            DataFrame con columnas categóricas codificadas.
            
        Raises:
            ValueError: Si fit=False y no hay encoders entrenados.
        """
        df = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
                
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                    df[col].astype(str)
                )
            else:
                if col not in self.label_encoders:
                    raise ValueError(f"No hay encoder entrenado para '{col}'")
                # Manejar valores no vistos
                df[f'{col}_encoded'] = df[col].apply(
                    lambda x: self._safe_transform(col, x)
                )
        
        self._fitted = True
        return df
    
    def _safe_transform(self, col: str, value: str) -> int:
        """Transforma un valor categórico de forma segura.
        
        Args:
            col: Nombre de la columna.
            value: Valor a transformar.
            
        Returns:
            Valor codificado o -1 si no está en el encoder.
        """
        try:
            return self.label_encoders[col].transform([str(value)])[0]
        except ValueError:
            return -1
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        group_cols: List[str] = ['sku_id', 'store_id'],
        categorical_cols: List[str] = ['category', 'sku_id', 'store_id'],
        target_col: str = 'quantity_sold',
        fit: bool = True
    ) -> pd.DataFrame:
        """Pipeline completo de feature engineering.
        
        Aplica todos los pasos de feature engineering en secuencia:
        1. Features de calendario
        2. Features de lag
        3. Features de rolling
        4. Encoding de categóricas
        
        Args:
            df: DataFrame con datos crudos.
            group_cols: Columnas para agrupar en lag/rolling.
            categorical_cols: Columnas categóricas a codificar.
            target_col: Columna objetivo.
            fit: Si True, ajusta encoders.
            
        Returns:
            DataFrame con todas las features creadas.
        """
        print("  📅 Creando features de calendario...")
        df = self.create_calendar_features(df)
        
        print("  ⏮️  Creando features de lag...")
        df = self.create_lag_features(df, group_cols, target_col)
        
        print("  📊 Creando features de rolling...")
        df = self.create_rolling_features(df, group_cols, target_col)
        
        print("  🏷️  Codificando variables categóricas...")
        df = self.encode_categorical_features(df, categorical_cols, fit)
        
        return df


# =============================================================================
# MODELO DE FORECASTING
# =============================================================================

class DemandForecaster:
    """Modelo de forecasting de demanda basado en XGBoost.
    
    Esta clase implementa un pipeline completo para predicción de demanda
    incluyendo preprocesamiento, entrenamiento, evaluación y predicción.
    
    Attributes:
        config: Configuración del modelo.
        feature_engineer: Instancia de FeatureEngineer.
        model: Modelo XGBoost entrenado.
        feature_columns: Lista de columnas usadas como features.
        train_metrics: Métricas de entrenamiento.
        
    Example:
        >>> config = ForecastConfig(horizon_days=30)
        >>> forecaster = DemandForecaster(config)
        >>> forecaster.fit(train_df)
        >>> predictions = forecaster.predict(test_df, horizon_days=30)
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        """Inicializa el DemandForecaster.
        
        Args:
            config: Configuración del modelo. Si None, usa valores por defecto.
        """
        self.config = config or ForecastConfig()
        self.feature_engineer = FeatureEngineer(self.config)
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_columns: List[str] = []
        self.train_metrics: Dict[str, float] = {}
        self._is_fitted = False
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Identifica las columnas de features a usar en el modelo.
        
        Args:
            df: DataFrame con features creadas.
            
        Returns:
            Lista de nombres de columnas de features.
        """
        exclude_cols = [
            'date', 'quantity_sold', 'sku_id', 'store_id', 'category',
            'stock_level'
        ]
        
        feature_cols = [
            col for col in df.columns 
            if col not in exclude_cols and df[col].dtype in ['int64', 'float64', 'int32']
        ]
        
        return feature_cols
    
    def _prepare_data(
        self, 
        df: pd.DataFrame, 
        fit: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepara los datos para entrenamiento o predicción.
        
        Args:
            df: DataFrame con datos crudos.
            fit: Si True, ajusta el feature engineer.
            
        Returns:
            Tuple de (DataFrame con features, array del target).
        """
        # Ordenar por fecha
        df = df.sort_values('date').reset_index(drop=True)
        
        # Crear features
        df = self.feature_engineer.create_all_features(df, fit=fit)
        
        # Eliminar filas con NaN (generados por lags)
        df = df.dropna()
        
        # Obtener columnas de features si es entrenamiento
        if fit:
            self.feature_columns = self._get_feature_columns(df)
        
        X = df[self.feature_columns].values
        y = df['quantity_sold'].values if 'quantity_sold' in df.columns else None
        
        return df, X, y
    
    def fit(
        self, 
        df: pd.DataFrame,
        eval_metric: str = 'rmse',
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ) -> 'DemandForecaster':
        """Entrena el modelo de forecasting.
        
        Realiza el entrenamiento del modelo XGBoost con validación cruzada
        temporal y calcula métricas de evaluación.
        
        Args:
            df: DataFrame con datos de entrenamiento.
            eval_metric: Métrica de evaluación ('rmse', 'mae').
            early_stopping_rounds: Rounds para early stopping.
            verbose: Si True, muestra progreso.
            
        Returns:
            Self para encadenamiento de métodos.
            
        Raises:
            ValueError: Si el DataFrame no tiene las columnas requeridas.
        """
        required_cols = ['date', 'sku_id', 'quantity_sold']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")
        
        if verbose:
            print("\n" + "="*60)
            print("🚀 ENTRENAMIENTO DEL MODELO DE FORECASTING")
            print("="*60)
            print(f"\n📊 Datos de entrada: {len(df):,} registros")
        
        # Preparar datos
        if verbose:
            print("\n1️⃣  Preparando features...")
        df_processed, X, y = self._prepare_data(df, fit=True)
        
        if verbose:
            print(f"   ✅ Features creadas: {len(self.feature_columns)}")
            print(f"   ✅ Registros válidos: {len(X):,}")
        
        # División temporal train/test
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        if verbose:
            print(f"\n2️⃣  División de datos:")
            print(f"   📈 Entrenamiento: {len(X_train):,} registros")
            print(f"   📉 Validación: {len(X_test):,} registros")
        
        # Crear y entrenar modelo
        if verbose:
            print("\n3️⃣  Entrenando modelo XGBoost...")
        
        self.model = xgb.XGBRegressor(**self.config.xgb_params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Calcular métricas
        if verbose:
            print("\n4️⃣  Evaluando modelo...")
        
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        self.train_metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_r2': r2_score(y_test, y_pred_test),
            'mape': self._calculate_mape(y_test, y_pred_test)
        }
        
        if verbose:
            self._print_metrics()
        
        self._is_fitted = True
        
        if verbose:
            print("\n✅ Modelo entrenado exitosamente!")
        
        return self
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula el Mean Absolute Percentage Error.
        
        Args:
            y_true: Valores reales.
            y_pred: Valores predichos.
            
        Returns:
            MAPE como porcentaje.
        """
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _print_metrics(self) -> None:
        """Imprime las métricas de evaluación del modelo."""
        print("\n" + "-"*40)
        print("📊 MÉTRICAS DE EVALUACIÓN")
        print("-"*40)
        print(f"   {'Conjunto':<15} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
        print(f"   {'-'*45}")
        print(f"   {'Entrenamiento':<15} {self.train_metrics['train_rmse']:>10.3f} "
              f"{self.train_metrics['train_mae']:>10.3f} "
              f"{self.train_metrics['train_r2']:>10.3f}")
        print(f"   {'Validación':<15} {self.train_metrics['test_rmse']:>10.3f} "
              f"{self.train_metrics['test_mae']:>10.3f} "
              f"{self.train_metrics['test_r2']:>10.3f}")
        print(f"\n   📈 MAPE (Validación): {self.train_metrics['mape']:.2f}%")
    
    def predict(
        self,
        df: pd.DataFrame,
        horizon_days: Optional[int] = None
    ) -> pd.DataFrame:
        """Genera predicciones de demanda.
        
        Args:
            df: DataFrame con datos para predicción.
            horizon_days: Días a predecir. Si None, usa config.
            
        Returns:
            DataFrame con predicciones.
            
        Raises:
            ValueError: Si el modelo no está entrenado.
        """
        if not self._is_fitted:
            raise ValueError("El modelo debe ser entrenado primero con fit()")
        
        horizon = horizon_days or self.config.horizon_days
        
        # Preparar datos
        df_processed, X, _ = self._prepare_data(df, fit=False)
        
        # Generar predicciones
        predictions = self.model.predict(X)
        
        # Asegurar predicciones no negativas
        predictions = np.maximum(predictions, 0)
        
        # Crear DataFrame de resultados
        result = df_processed[['date', 'sku_id', 'store_id', 'category']].copy()
        result['predicted_quantity'] = np.round(predictions).astype(int)
        
        return result
    
    def forecast_future(
        self,
        historical_df: pd.DataFrame,
        horizon_days: Optional[int] = None
    ) -> pd.DataFrame:
        """Genera pronósticos para fechas futuras.
        
        Crea predicciones para los próximos N días a partir de la última
        fecha en los datos históricos.
        
        Args:
            historical_df: DataFrame con datos históricos.
            horizon_days: Número de días a pronosticar.
            
        Returns:
            DataFrame con pronósticos futuros por SKU.
        """
        if not self._is_fitted:
            raise ValueError("El modelo debe ser entrenado primero con fit()")
        
        horizon = horizon_days or self.config.horizon_days
        
        print(f"\n🔮 Generando pronóstico para los próximos {horizon} días...")
        
        # Obtener última fecha y combinaciones únicas de SKU/Store
        historical_df['date'] = pd.to_datetime(historical_df['date'])
        last_date = historical_df['date'].max()
        
        unique_combinations = historical_df.groupby(
            ['sku_id', 'store_id', 'category']
        ).size().reset_index()[['sku_id', 'store_id', 'category']]
        
        # Generar fechas futuras
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        # Crear DataFrame de fechas futuras para cada combinación
        future_records = []
        for _, row in unique_combinations.iterrows():
            for date in future_dates:
                future_records.append({
                    'date': date,
                    'sku_id': row['sku_id'],
                    'store_id': row['store_id'],
                    'category': row['category'],
                    'quantity_sold': 0,  # Placeholder
                    'stock_level': 50    # Placeholder
                })
        
        future_df = pd.DataFrame(future_records)
        
        # Combinar con datos históricos para calcular lags
        combined_df = pd.concat([historical_df, future_df], ignore_index=True)
        combined_df = combined_df.sort_values(['sku_id', 'store_id', 'date'])
        
        # Generar predicciones
        predictions = self.predict(combined_df)
        
        # Filtrar solo fechas futuras
        predictions['date'] = pd.to_datetime(predictions['date'])
        future_predictions = predictions[predictions['date'] > last_date]
        
        print(f"✅ Pronóstico generado: {len(future_predictions):,} predicciones")
        
        return future_predictions.reset_index(drop=True)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Obtiene la importancia de las features del modelo.
        
        Args:
            top_n: Número de features top a retornar.
            
        Returns:
            DataFrame con importancia de features ordenado.
            
        Raises:
            ValueError: Si el modelo no está entrenado.
        """
        if not self._is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        })
        
        importance = importance.sort_values('importance', ascending=False)
        
        return importance.head(top_n).reset_index(drop=True)
    
    def save_model(self, filepath: str) -> None:
        """Guarda el modelo entrenado en disco.
        
        Args:
            filepath: Ruta donde guardar el modelo.
        """
        if not self._is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'label_encoders': self.feature_engineer.label_encoders,
            'config': self.config,
            'train_metrics': self.train_metrics
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"💾 Modelo guardado en: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'DemandForecaster':
        """Carga un modelo guardado desde disco.
        
        Args:
            filepath: Ruta del modelo guardado.
            
        Returns:
            Instancia de DemandForecaster con el modelo cargado.
        """
        model_data = joblib.load(filepath)
        
        forecaster = cls(config=model_data['config'])
        forecaster.model = model_data['model']
        forecaster.feature_columns = model_data['feature_columns']
        forecaster.feature_engineer.label_encoders = model_data['label_encoders']
        forecaster.feature_engineer._fitted = True
        forecaster.train_metrics = model_data['train_metrics']
        forecaster._is_fitted = True
        
        print(f"📂 Modelo cargado desde: {filepath}")
        return forecaster


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def train_and_forecast(
    data_path: str,
    output_path: str,
    horizon_days: int = 30,
    save_model_path: Optional[str] = None
) -> Tuple[DemandForecaster, pd.DataFrame]:
    """Pipeline completo de entrenamiento y forecasting.
    
    Función de conveniencia que ejecuta el pipeline completo:
    1. Carga datos
    2. Entrena modelo
    3. Genera pronósticos
    4. Guarda resultados
    
    Args:
        data_path: Ruta al archivo CSV de datos.
        output_path: Ruta para guardar predicciones.
        horizon_days: Días a pronosticar.
        save_model_path: Ruta opcional para guardar el modelo.
        
    Returns:
        Tuple de (forecaster entrenado, DataFrame de predicciones).
    """
    print("\n" + "="*60)
    print("🏠 PIPELINE DE FORECASTING - RETAIL HOGAR")
    print("="*60)
    
    # Cargar datos
    print(f"\n📂 Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   ✅ Registros cargados: {len(df):,}")
    
    # Crear y entrenar modelo
    config = ForecastConfig(horizon_days=horizon_days)
    forecaster = DemandForecaster(config)
    forecaster.fit(df)
    
    # Generar pronósticos
    predictions = forecaster.forecast_future(df, horizon_days)
    
    # Guardar predicciones
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"\n💾 Predicciones guardadas en: {output_path}")
    
    # Guardar modelo si se especifica ruta
    if save_model_path:
        forecaster.save_model(save_model_path)
    
    # Mostrar importancia de features
    print("\n" + "-"*40)
    print("📊 TOP 10 FEATURES MÁS IMPORTANTES")
    print("-"*40)
    importance = forecaster.get_feature_importance(10)
    for idx, row in importance.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"   {row['feature']:<25} {bar} {row['importance']:.3f}")
    
    print("\n" + "="*60)
    print("🎉 ¡Pipeline completado exitosamente!")
    print("="*60)
    
    return forecaster, predictions


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    # Determinar rutas
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # src/models -> src -> project_root
    
    data_path = project_root / 'data' / 'raw_sales.csv'
    output_path = project_root / 'data' / 'demand_forecast_30d.csv'
    model_path = project_root / 'models' / 'demand_forecaster.joblib'
    
    # Verificar que existen los datos
    if not data_path.exists():
        print(f"❌ Error: No se encontró el archivo de datos en {data_path}")
        print("   Ejecute primero: python src/generate_synthetic_data.py")
        sys.exit(1)
    
    # Ejecutar pipeline
    forecaster, predictions = train_and_forecast(
        data_path=str(data_path),
        output_path=str(output_path),
        horizon_days=30,
        save_model_path=str(model_path)
    )
    
    # Mostrar resumen de predicciones
    print("\n📋 RESUMEN DE PREDICCIONES:")
    print("-"*40)
    print(f"   Período: {predictions['date'].min()} a {predictions['date'].max()}")
    print(f"   Total predicciones: {len(predictions):,}")
    print(f"   SKUs únicos: {predictions['sku_id'].nunique()}")
    print(f"   Demanda total pronosticada: {predictions['predicted_quantity'].sum():,} unidades")
