import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import pandas as pd
from feature_engineering import feature_engineering

DATA_CSV = "demo_datasets/synthetic_sales.csv"
SKU = "SKU-1"

def load_data(forecast_days, csv=DATA_CSV, sku=SKU, ):
    df = pd.read_csv(csv, parse_dates=["ds"])
    df = df[df["sku"] == sku].sort_values("ds").reset_index(drop=True)
    df = feature_engineering(df)
    train = df.iloc[:-forecast_days].copy()
    test = df.iloc[-forecast_days:].copy()
    return df, train, test

def load_data_prod(uploaded_df, sku=SKU):
    df = uploaded_df.copy()
    df = df[df["sku"] == sku].sort_values("ds").reset_index(drop=True)
    df = feature_engineering(df)
    train_df = df.copy()
    return train_df

# ==========================================================
#                  CybrexXGBoost
# ==========================================================

class CybrexXGBoost():
    def __init__(self):
        self.model = None
        self.best_params = None

    def tune_hyperparameters_xgb(self, df, features, target="sales", n_trials=70, test_size=0.2):
        X = df[features]
        y = df[target]

        X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                          test_size=test_size,
                                                          )
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        def objective_xgb(trial):
            params = {
                "objective": "reg:squarederror",
                "eta": trial.suggest_float("eta", 0.01, 0.1),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
                "alpha": trial.suggest_float("alpha", 0.0, 1.0),
                "lambda": trial.suggest_float("lambda", 0.5, 2.0),
                "eval_metric": "rmse",
                "tree_method": "hist"
            }
            num_boost_round = trial.suggest_int("num_boost_round", 200, 600)
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=[(dval, "validation")],
                verbose_eval=False
            )
            preds = model.predict(dval)
            rmse = mean_squared_error(y_val, preds, squared=False)
            return rmse
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_xgb, n_trials=n_trials)
        self.best_params = study.best_params

    def train_xgboost(self, train_df, features, target="sales"):
        dtrain = xgb.DMatrix(train_df[features], 
                            label=train_df[target])
        
        if not self.best_params:
            # Если гиперпараметры не подбирались — используем стандартные
            params = {
                "objective": "reg:squarederror",
                "eta": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "eval_metric": "rmse",
                "seed": 42
            }
            num_boost_round = 300
        else:
            params = self.best_params.copy()
            num_boost_round = params.pop("num_boost_round", 300)
        
        self.model = xgb.train(params, 
                        dtrain, 
                        num_boost_round=num_boost_round, 
                        verbose_eval=False)
        return self.model

    def predict(self, test_df, features):
        dtest = xgb.DMatrix(test_df[features])
        preds = self.model.predict(dtest)
        return preds
        

# ==========================================================
#                  CybrexProphet
# ==========================================================

class CybrexProphet():
    def __init__(self):
        self.model = None
        self.best_params = None

    def tune_hyperparameters_prophet(self, df, valid_ratio=0.2, n_trials=50):

        dfp = df[["ds", "sales", "promo"]].rename(columns={"sales": "y"})
        split_idx = int(len(dfp) * (1 - valid_ratio))
        train_df, val_df = dfp.iloc[:split_idx], dfp.iloc[split_idx:]

        def objective_prophet(trial, train_df, val_df):
            params = {
                "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
                "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 1.0, 15.0),
                "holidays_prior_scale": trial.suggest_float("holidays_prior_scale", 0.01, 10.0, log=True),
                "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
                "yearly_seasonality": trial.suggest_categorical("yearly_seasonality", [True, False]),
                "weekly_seasonality": trial.suggest_categorical("weekly_seasonality", [True, False]),
                "daily_seasonality": trial.suggest_categorical("daily_seasonality", [False])
            }
            model = Prophet(**params)
            model.add_regressor("promo")
            model.fit(train_df)

            future = val_df[["ds", "promo"]]
            forecast = model.predict(future)

            y_true = val_df["y"].values
            y_pred = forecast["yhat"].values
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective_prophet(trial, train_df, val_df), 
                       n_trials=n_trials)

        self.best_params = study.best_params


    def train_prophet(self, train_df):
        # Prophet требует, чтобы входной DataFrame имел строго определённые названия столбцов
        dfp = train_df[["ds", "sales", "promo"]].rename(columns={"sales":"y"})


        if not self.best_params:
            self.model = Prophet(yearly_seasonality='auto',
                                weekly_seasonality='auto', 
                                daily_seasonality='auto',
                        )
        else:
            self.model = Prophet(**self.best_params)
            
        # Добавление дополнительного фактора (регрессора)
        self.model.add_regressor("promo")
        self.model.fit(dfp)
        return self.model
    
    def predict(self, df, forecast_days):
        future = self.model.make_future_dataframe(periods=forecast_days)
        if "promo" in df.columns:
            # align by date
            promo_map = df.set_index("ds")["promo"].to_dict()
            future["promo"] = future["ds"].map(lambda d: promo_map.get(d, 0))
        prophet_fc = self.model.predict(future)
        prophet_pred = prophet_fc.tail(forecast_days)["yhat"].values
        return prophet_pred
    
    def predict_for_meta(self, df):
        return self.model.predict(df)["yhat"]