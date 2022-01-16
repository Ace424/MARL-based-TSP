import xgboost
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score
import numpy as np


def xgb_f1(pred, xgbtrain):
    label = xgbtrain.get_label()
    pred = 1 / (1 + np.exp(-pred))
    y_pred = (pred >= 0.5).astype(float)
    f1 = f1_score(label, y_pred)
    return 'xgb_f1', -f1


def xgb_classify_mix(params):
    model = xgboost.XGBClassifier(**params)
    return model


def xgb_classify():
    model = xgboost.XGBClassifier(max_depth=6, learning_rate="0.1",
                                  n_estimators=600, verbosity=0, subsample=0.8,
                                  colsample_bytree=0.8, use_label_encoder=False, scale_pos_weight=1)
    return model


def xgb_regression():
    model = xgboost.XGBRegressor()
    return model


def rf_classify():
    model = RandomForestClassifier(n_estimators=600, max_depth=8, n_jobs=1, class_weight='balanced', random_state=42)
    return model


def rf_regression():
    model = RandomForestRegressor(n_estimators=600, max_depth=8, n_jobs=1, random_state=42)
    return model


def lr_classify(type=None, penalty='l2'):
    if not type:
        solver = 'liblinear'
        n_jobs = 1
    else:
        solver = 'sag'
        n_jobs = -1
    model = LogisticRegression(class_weight='balanced', n_jobs=n_jobs, tol=0.0005, C=0.1,
                               max_iter=10000, solver=solver, penalty=penalty)
    return model


# def lr_classify():
#     model = LogisticRegression(class_weight='balanced', tol=0.0005, C=0.1, max_iter=10000)
#     return model


def lr_regression():
    model = LogisticRegression(class_weight='balanced', tol=0.0005, C=0.5, max_iter=10000)
    return model
