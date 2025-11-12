import pandas as pd

def feature_engineering(df):
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    # lags
    # сдвигает данные на 1 день назад, то есть значение продаж вчера.
    df["lag_1"] = df["sales"].shift(1).fillna(method="bfill")
    # продажи неделю назад
    df["lag_7"] = df["sales"].shift(7).fillna(method="bfill")
    # cкользящие средние: среднее значение продаж за последние 7 (или 30) дней.
    df["ma7"] = df["sales"].rolling(window=7, min_periods=1).mean().fillna(method="bfill")
    df["ma30"] = df["sales"].rolling(window=30, min_periods=1).mean().fillna(method="bfill")
    return df