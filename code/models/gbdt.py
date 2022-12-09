import lightgbm as lgb


class GBDTClassifier:

    def __init__(self,
                 boosting: str = 'gbdt',
                 objective: str = 'multiclass',
                 num_class: int = 10,
                 metric: str = 'multi_logloss',
                 num_iterations: int = 1_000,
                 learning_rate: float = 0.05,
                 num_leaves: int = 2048,
                 max_depth: int = 15,
                 min_data_in_leaf: int = 50,
                 feature_fraction: float = 0.5,
                 device: str = 'cpu',
                 num_threads: int = 24,
                 verbosity: int = -1):
        self.hparams = {
            'boosting': boosting,
            'objective': objective,
            'num_class': num_class,
            'metric': metric,
            'num_iterations': num_iterations,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'min_data_in_leaf': min_data_in_leaf,
            'feature_fraction': feature_fraction,
            'device': device,
            'num_threads': num_threads,
            # 1 means INFO, > 1 means DEBUG, 0 means Error(WARNING), <0 means Fatal
            'verbosity': verbosity
        }

    def train(self, X_train, y_train):
        lgbm_dataset = lgb.Dataset(X_train, y_train)
        lgbm_model = lgb.train(self.hparams, lgbm_dataset)
        return lgbm_model

    def test(self, model_file, X_test):
        lgbm_model = lgb.Booster(model_file=model_file)
        return lgbm_model.predict(X_test)