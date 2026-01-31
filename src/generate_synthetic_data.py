"""
Generador de Datos Sintéticos de Ventas para Retail de Hogar
=============================================================

Este script genera datos sintéticos de ventas para un retailer de mejoras del hogar
en Bolivia. Los datos incluyen tendencias estacionales realistas considerando:
- Temporada de lluvias (Nov-Mar): Mayor demanda de impermeabilizantes y pinturas
- Temporada seca (Abr-Oct): Mayor demanda de herramientas de construcción
- Festividades bolivianas: Picos de demanda en fechas específicas

Autor: ML & Research Scientist Agent
Fecha: 2026-01-31
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Configuración de reproducibilidad
np.random.seed(42)

# =============================================================================
# CONSTANTES Y CONFIGURACIÓN
# =============================================================================

START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2025, 12, 31)

# Categorías de productos con menos SKUs para asegurar densidad de datos
CATEGORIES = {
    'Pinturas': {
        'skus': [f'PINT-{i:04d}' for i in range(1, 6)], # Reduced to 5
        'base_quantity': 15,
        'seasonality': 'rainy',
        'price_range': (50, 500)
    },
    'Herramientas': {
        'skus': [f'HERR-{i:04d}' for i in range(1, 6)],
        'base_quantity': 8,
        'seasonality': 'dry',
        'price_range': (20, 1500)
    },
    'Plomería': {
        'skus': [f'PLOM-{i:04d}' for i in range(1, 4)],
        'base_quantity': 12,
        'seasonality': 'rainy',
        'price_range': (15, 800)
    },
    'Electricidad': {
        'skus': [f'ELEC-{i:04d}' for i in range(1, 4)],
        'base_quantity': 10,
        'seasonality': 'neutral',
        'price_range': (10, 600)
    },
    'Ferretería': {
        'skus': [f'FERR-{i:04d}' for i in range(1, 6)],
        'base_quantity': 20,
        'seasonality': 'neutral',
        'price_range': (5, 200)
    },
    'Construcción': {
        'skus': [f'CONS-{i:04d}' for i in range(1, 4)],
        'base_quantity': 25,
        'seasonality': 'dry',
        'price_range': (30, 2000)
    },
    'Jardín': {
        'skus': [f'JARD-{i:04d}' for i in range(1, 3)],
        'base_quantity': 6,
        'seasonality': 'rainy',
        'price_range': (25, 400)
    },
    'Iluminación': {
        'skus': [f'ILUM-{i:04d}' for i in range(1, 4)],
        'base_quantity': 7,
        'seasonality': 'neutral',
        'price_range': (20, 350)
    }
}

# Tiendas reducidas para demo
STORES = {
    'ST-001': {'city': 'La Paz', 'size': 'large'},
    'ST-002': {'city': 'Santa Cruz', 'size': 'large'},
    'ST-003': {'city': 'Cochabamba', 'size': 'medium'},
    'ST-004': {'city': 'El Alto', 'size': 'medium'},
}

STORE_SIZE_MULTIPLIER = {
    'large': 1.5,
    'medium': 1.0,
    'small': 0.6
}

# Fechas especiales en Bolivia (mes, día) con multiplicador de demanda
SPECIAL_DATES = {
    (1, 1): 0.3,    # Año Nuevo - tiendas cerradas o baja demanda
    (2, 2): 1.4,    # Día de la Candelaria - renovaciones
    (3, 23): 1.3,   # Día del Mar
    (5, 1): 0.4,    # Día del Trabajo
    (6, 21): 1.5,   # Año Nuevo Aymara - renovaciones ceremoniales
    (8, 6): 1.6,    # Independencia - decoración patriótica
    (11, 2): 1.3,   # Día de los Difuntos - mantenimiento cementerios
    (12, 25): 0.3,  # Navidad
}


# =============================================================================
# FUNCIONES DE GENERACIÓN
# =============================================================================

def get_seasonal_multiplier(date: datetime, seasonality_type: str) -> float:
    """
    Calcula el multiplicador estacional basado en el mes y tipo de estacionalidad.
    """
    month = date.month
    
    # Definir intensidad de temporada por mes
    rainy_intensity = {
        1: 0.9, 2: 0.85, 3: 0.7,  # Fin de lluvias
        4: 0.3, 5: 0.2, 6: 0.1,   # Temporada seca
        7: 0.1, 8: 0.15, 9: 0.25, # Temporada seca
        10: 0.4, 11: 0.7, 12: 0.95  # Inicio lluvias
    }
    
    if seasonality_type == 'rainy':
        return 0.7 + (rainy_intensity[month] * 1.3)
    elif seasonality_type == 'dry':
        return 0.7 + ((1 - rainy_intensity[month]) * 1.3)
    else:
        return 0.9 + np.random.uniform(-0.1, 0.2)


def get_day_of_week_multiplier(date: datetime) -> float:
    day = date.weekday()
    multipliers = {
        0: 0.85, 1: 0.90, 2: 0.95, 3: 0.95, 
        4: 1.10, 5: 1.40, 6: 0.60
    }
    return multipliers[day]


def get_special_date_multiplier(date: datetime) -> float:
    key = (date.month, date.day)
    return SPECIAL_DATES.get(key, 1.0)


def get_trend_multiplier(date: datetime) -> float:
    years_from_start = (date - START_DATE).days / 365.25
    annual_growth = 1.065
    return annual_growth ** years_from_start


def calculate_stock_level(quantity_sold: int, base_stock: int = 100) -> int:
    consumption_rate = quantity_sold / base_stock
    if consumption_rate > 0.8:
        return max(5, int(base_stock * np.random.uniform(0.05, 0.15)))
    elif consumption_rate > 0.5:
        return int(base_stock * np.random.uniform(0.2, 0.4))
    else:
        return int(base_stock * np.random.uniform(0.5, 0.9))


def generate_synthetic_sales(num_rows: int = None) -> pd.DataFrame:
    """
    Genera el dataset completo de ventas sintéticas.
    Genera datos diarios para todas las combinaciones de Tienda/SKU.
    """
    print(f"🚀 Iniciando generación de datos de series temporales densas...")
    
    # Generar rango de fechas completo
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    print(f"   📅 Fechas: {len(date_range)} días ({START_DATE.date()} a {END_DATE.date()})")
    
    records = []
    
    # Preparar combinaciones de SKU y Categoría
    sku_list = []
    for category, config in CATEGORIES.items():
        for sku in config['skus']:
            sku_list.append({
                'sku_id': sku,
                'category': category,
                'base_quantity': config['base_quantity'],
                'seasonality': config['seasonality']
            })
    
    total_combinations = len(STORES) * len(sku_list)
    total_rows = total_combinations * len(date_range)
    print(f"   📦 Combinaciones (Tienda x SKU): {len(STORES)} x {len(sku_list)} = {total_combinations}")
    print(f"   📊 Total estimado de filas: {total_rows:,}")

    count = 0
    # Iterar para generar serie temporal completa
    for store_id, store_info in STORES.items():
        store_size = store_info['size']
        store_mult = STORE_SIZE_MULTIPLIER[store_size]
        
        for sku_info in sku_list:
            # Vectorizar cálculos si es posible, pero loop es claro para demo
            for date in date_range:
                
                # Multiplicadores
                seasonal_mult = get_seasonal_multiplier(date, sku_info['seasonality'])
                dow_mult = get_day_of_week_multiplier(date)
                special_mult = get_special_date_multiplier(date)
                trend_mult = get_trend_multiplier(date)
                
                # Ruido aleatorio día a día (10% - 30%)
                daily_noise = np.random.uniform(0.7, 1.3)
                
                # Cantidad
                base_qty = sku_info['base_quantity']
                quantity = base_qty * seasonal_mult * dow_mult * special_mult * trend_mult * store_mult * daily_noise
                
                # Introducir algunos ceros (stockouts, días sin venta) ~5%
                if np.random.random() < 0.05:
                    quantity = 0
                
                quantity = int(round(quantity))
                stock_level = calculate_stock_level(quantity)
                
                records.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'sku_id': sku_info['sku_id'],
                    'category': sku_info['category'],
                    'quantity_sold': quantity,
                    'store_id': store_id,
                    'stock_level': stock_level
                })
                count += 1
                
                if count % 50000 == 0:
                     print(f"  ... {count:,} filas generadas")
    
    df = pd.DataFrame(records)
    print(f"✅ Generación completada: {len(df):,} registros")
    return df


def save_to_csv(df: pd.DataFrame, filepath: str) -> None:
    """
    Guarda el DataFrame en un archivo CSV.
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"💾 Datos guardados en: {filepath}")
    print(f"   Tamaño del archivo: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")


def print_summary(df: pd.DataFrame) -> None:
    """
    Imprime un resumen estadístico del dataset generado.
    """
    print("\n" + "="*60)
    print("📈 RESUMEN DEL DATASET GENERADO")
    print("="*60)
    
    print(f"\n📅 Rango de fechas: {df['date'].min()} a {df['date'].max()}")
    print(f"📦 Total de registros: {len(df):,}")
    print(f"🏷️  SKUs únicos: {df['sku_id'].nunique():,}")
    print(f"🏪 Tiendas: {df['store_id'].nunique()}")
    
    print("\n📊 Distribución por Categoría:")
    print("-" * 40)
    category_stats = df.groupby('category').agg({
        'quantity_sold': ['count', 'sum', 'mean']
    }).round(2)
    category_stats.columns = ['Registros', 'Total Vendido', 'Promedio']
    print(category_stats.to_string())
    
    print("\n🏪 Distribución por Tienda:")
    print("-" * 40)
    store_stats = df.groupby('store_id').agg({
        'quantity_sold': ['count', 'sum']
    })
    store_stats.columns = ['Registros', 'Total Vendido']
    for store_id in store_stats.index:
        city = STORES[store_id]['city']
        store_stats.loc[store_id, 'Ciudad'] = city
    print(store_stats[['Ciudad', 'Registros', 'Total Vendido']].to_string())
    
    print("\n📉 Estadísticas de Cantidad Vendida:")
    print("-" * 40)
    print(df['quantity_sold'].describe().to_string())
    
    print("\n📦 Estadísticas de Nivel de Stock:")
    print("-" * 40)
    print(df['stock_level'].describe().to_string())


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🏠 GENERADOR DE DATOS SINTÉTICOS - RETAIL HOGAR BOLIVIA")
    print("="*60)
    
    # Determinar rutas
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_path = os.path.join(project_root, 'data', 'raw_sales.csv')
    
    # Generar datos
    df = generate_synthetic_sales()
    
    # Guardar CSV
    save_to_csv(df, output_path)
    
    # Mostrar resumen
    print_summary(df)
    
    print("\n" + "="*60)
    print("🎉 ¡Proceso completado exitosamente!")
    print("="*60)
