import pickle
from sklearn.ensemble import RandomForestClassifier


class RandomForest:

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 15,
                 max_leaf_nodes: int = 2048,
                 random_state: int = 0,
                 n_jobs: int = 20):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.n_jobs = n_jobs

    def train(self, X_train, y_train):
        model = RandomForestClassifier(n_estimators=self.n_estimators,
                                       max_depth=self.max_depth,
                                       max_leaf_nodes=self.max_leaf_nodes,
                                       random_state=self.random_state,
                                       n_jobs=self.n_jobs)
        model.fit(X_train, y_train)
        return model

    def test(self, model_file, X_test):
        with open(model_file, 'rb') as f:
            model: RandomForestClassifier = pickle.load(f)
        return model.predict(X_test)