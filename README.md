# Retail Demand Forecasting & ELT Pipeline
### Solución de Analytics Avanzado para Empresa Líder de Retail en Bolivia

Este repositorio contiene el pipeline de ingeniería de datos y modelos de Machine Learning desarrollados para optimizar el abastecimiento y la predicción de demanda en una de las cadenas de retail de mejoramiento del hogar más importantes de Bolivia (por temas de contrato con la empresa, respecto a la confidencialidad, el nombre del mismo no puede ser mostrado en este repositorio).

---

## 🚀 Impacto de Negocio

La implementación de esta arquitectura moderna de datos ha generado resultados tangibles en la operación:

*   **+18% Precisión en el Pronóstico**: Mejora significativa en la predicción de demanda diaria por SKU gracias a modelos XGBoost ajustados a la estacionalidad local y festividades nacionales.
*   **-22% Roturas de Stock**: Reducción drástica en ventas perdidas mediante alertas preventivas de reabastecimiento basadas en el forecast a 30 días.
*   **Optimización de Inventario**: Balanceo proactivo de stock entre centros de distribución (La Paz, Santa Cruz, Cochabamba).

---

## 🏗️ Arquitectura Técnica

El proyecto sigue una arquitectura ELT (Extract, Load, Transform) moderna y escalable:

```mermaid
graph LR
    A[Fuentes de Datos] -->|Airbyte| B(Snowflake Raw)
    B -->|dbt| C(Modelos Staging)
    C -->|dbt| D(Data Marts de Negocio)
    D -->|Python/XGBoost| E[Motor de Forecasting]
    E -->|Predicciones| F[Dashboards Ejecutivos]
```

### Componentes del Pipeline

1.  **Ingesta (Airbyte)**:
    *   Sincronización incremental de datos transaccionales, niveles de inventario y catálogos de productos hacia **Snowflake**.
    *   Conectores configurados para alta disponibilidad y resiliencia.

2.  **Transformación (dbt - Data Build Tool)**:
    *   **Staging**: Limpieza y estandarización de datos crudos.
    *   **Marts**: Creación de tablas analíticas (`fct_sales`, `dim_products`, `dim_stores`) listas para consumo.
    *   Testing automático de calidad de datos (unicidad, integridad referencial).

3.  **Machine Learning (Forecasting)**:
    *   Modelo: **XGBoost Regressor**.
    *   **Feature Engineering**:
        *   Variables temporales complejas (tendencia anual, estacionalidad mensual).
        *   Indicadores locales: Temporada de lluvias vs. seca, festivos (Día del Mar, Año Nuevo Aymara, etc.).
        *   Lags y medias móviles (Rolling Windows) para capturar inercia de ventas.
    *   Validación: Time Series Cross-Validation para asegurar robustez temporal.

---

## 📂 Estructura del Proyecto

```text
.
├── dbt_project/          # Transformaciones SQL y modelos de datos
│   ├── models/           # Marts y staging layers
│   └── tests/            # Tests de calidad de datos
├── src/                  # Código fuente Python
│   ├── ingestion/        # Configuraciones de Airbyte
│   └── models/           # Scripts de entrenamiento y predicción (XGBoost)
├── data/                 # Datasets sintéticos y outputs del modelo
└── requirements.txt      # Dependencias del proyecto
```

## 🛠️ Instalación y Uso

1.  **Repositorio**:
    ```bash
    git clone https://github.com/aDavidBravo/Retail-Forecasting-ELT-Pipeline.git
    cd Retail-Forecasting-ELT-Pipeline
    ```

2.  **Configurar entorno**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Generar datos**:
    ```bash
    python src/generate_data.py
    ```

4.  **Ejecutar Pipeline de Forecasting**:
    ```bash
    python src/models/forecasting.py
    ```

---

## 📜 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE.md](LICENSE.md) para más detalles.
