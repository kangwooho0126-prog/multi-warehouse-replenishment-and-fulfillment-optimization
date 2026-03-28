import lightgbm as lgb


def build_lightgbm_model(random_state: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective='regression',
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
    )
