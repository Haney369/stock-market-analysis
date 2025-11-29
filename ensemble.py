import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier


# ------------------------------------------------------------
#  Train Ensemble Models (RandomForest + XGBoost)
# ------------------------------------------------------------
def train_ensemble(df: pd.DataFrame, features: list):
    """
    Trains a dual-model ensemble:
    - RandomForestRegressor + XGBRegressor for growth prediction
    - RandomForestClassifier + XGBClassifier for hype detection

    Returns: dictionary of all fitted models
    """
    X = df[features]
    y_reg = df["FwdReturn"]     # regression target (growth)
    y_cls = df["Hype"]          # classification target (hype/no hype)

    # -----------------------------
    # RandomForest Models
    # -----------------------------
    rf_reg = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    rf_clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    rf_reg.fit(X, y_reg)
    rf_clf.fit(X, y_cls)

    # -----------------------------
    # XGBoost Models
    # -----------------------------
    xgb_reg = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist"
    )

    xgb_clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="logloss"
    )

    xgb_reg.fit(X, y_reg)
    xgb_clf.fit(X, y_cls)

    return {
        "rf_reg": rf_reg,
        "rf_clf": rf_clf,
        "xgb_reg": xgb_reg,
        "xgb_clf": xgb_clf,
    }


# ------------------------------------------------------------
#  Ensemble Prediction
# ------------------------------------------------------------
def ensemble_predict(models: dict, latest_row: pd.DataFrame):
    """
    Predicts using both RF + XGB and combines results:
    
    - Regression → Average of both models
    - Classification → Majority (soft) vote

    Returns:
        (final_growth_pred, final_hype_flag)
    """

    # ---- Regression predictions ----
    rf_pred = models["rf_reg"].predict(latest_row)[0]
    xgb_pred = models["xgb_reg"].predict(latest_row)[0]

    # final regression = simple average
    final_growth = (rf_pred + xgb_pred) / 2

    # ---- Classification predictions ----
    rf_cls = models["rf_clf"].predict(latest_row)[0]
    xgb_cls = models["xgb_clf"].predict(latest_row)[0]

    # majority / average vote
    final_hype = int(round((rf_cls + xgb_cls) / 2))

    return final_growth, final_hype
