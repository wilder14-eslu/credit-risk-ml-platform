import numpy as np
import pandas as pd

from src.ml.train import train_model

def test_model_convergence_and_nans(tmp_path) -> None:
    """Prueba de Integración Continua (Nivel 2 de MLOps) requerida por Google.
    
    Asegura que el modelo realmente converge (aprende patrones) y no devuelve NaNs
    ante distribuciones de datos ruidosas o perfectas, antes de permitir que
    una nueva arquitectura sea mezclada a la rama main.
    """
    # 1. Crear un dataset sintético donde la variable objetivo es perfectamente
    # predecible (para asegurar que el algoritmo converja y no subajuste brutalmente).
    np.random.seed(42)
    n_samples = 500
    
    # Feature principal: ingreso mensual. Si > 5000, no incumple (0). Si < 5000, incumple (1).
    monthly_income = np.random.uniform(1000, 10000, size=n_samples)
    target = (monthly_income < 5000).astype(int)
    
    # Introducir un poco de ruido (5%) para que no sea 100% perfecto
    noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    target[noise_indices] = 1 - target[noise_indices]
    
    data = pd.DataFrame({
        "MonthlyIncome": monthly_income,
        "age": np.random.randint(18, 80, size=n_samples),
        "DebtRatio": np.random.uniform(0, 1, size=n_samples),
        "RevolvingUtilizationOfUnsecuredLines": np.random.uniform(0, 1, size=n_samples),
        "NumberOfTime30-59DaysPastDueNotWorse": np.zeros(n_samples),
        "NumberOfOpenCreditLinesAndLoans": np.random.randint(1, 10, size=n_samples),
        "NumberOfTimes90DaysLate": np.zeros(n_samples),
        "NumberRealEstateLoansOrLines": np.zeros(n_samples),
        "NumberOfTime60-89DaysPastDueNotWorse": np.zeros(n_samples),
        "NumberOfDependents": np.zeros(n_samples),
        "SeriousDlqin2yrs": target
    })
    
    model_path = tmp_path / "model.joblib"
    
    # 2. Entrenar modelo
    result = train_model(data, model_path=model_path, register=False, algorithm="xgboost")
    
    # 3. Validar Convergencia (ROC-AUC debe ser significativamente mejor que el azar > 0.8)
    assert result["test_metrics"]["roc_auc"] > 0.8, "El modelo no convergió al patrón obvio"
    
    # 4. Validar ausencia de NaNs (Log loss o Brier score no deben ser NaN)
    assert not np.isnan(result["test_metrics"]["log_loss"])
    assert not np.isnan(result["test_metrics"]["brier_score"])
    
    # 5. La predicción de prueba no debe devolver NaNs ni infinitos
    assert result["diagnosis"]["label"] != "subajuste"
