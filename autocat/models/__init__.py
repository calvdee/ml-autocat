import pandas as pd
from sklearn.model_selection import GridSearchCV

class Model(object):
    def __init__(self, name, model, pipeline=None):
        self.name = name
        self.model = model
        self.pipeline = pipeline

    def fit(self, X, y):
        """ Fits the model and builds the full pipeline """
        from sklearn.pipeline import make_pipeline
        X_transformed = self.pipeline.fit_transform(X)
        self.model.fit(X_transformed, y)
        self.model_pipeline = make_pipeline(self.pipeline, self.model)
        return self

    def get_model_pipeline(self):
        """ Useful for cross validation to refit the pipeline on every round """
        from sklearn.base import clone
        full_pipeline = clone(self.pipeline)
        full_pipeline.steps.append((self.name, self.model))
        return full_pipeline

    def load(file_path):
        from sklearn.externals import joblib
        file_model = joblib.load(file_path)
        return file_model

    def save(model, file_path):
        from sklearn.externals import joblib
        joblib.dump(model, file_path)
    
    def score(self, X, y, scorer):
        model_predictions = self.model_pipeline.predict(X)
        score = scorer(self.model_pipeline, X, y)
        return (score, model_predictions)
        
class GridSearchModel(Model):
  """ A class used to explore hyperparameters of machine learning
  model using grid search.

  #:todo: this is more like a "TunedModel" - grid search is the implementation

  Parameters
  ----------
  name: string
      The name of a model
  param_grid: dictionary
      A dictionary of (parameter, values) pairs to optimize
  pipeline: object
      A pipeline to apply to the data before fitting the model
  """

  def __init__(self, param_grid, **kwargs):
      Model.__init__(self, **kwargs)
      self.param_grid = param_grid

  def train(self, X, y, cv_folds, scorer):
      """ Get a pipeline that includes the model """
      model_pipeline = self.get_model_pipeline()

      # Create and execute grid search
      grid_search = GridSearchCV(
          model_pipeline, 
          self.param_grid, 
          cv=cv_folds,
          scoring=scorer, 
          return_train_score=True, 
          n_jobs=-1)

      grid_search.fit(X, y)

      self.model = grid_search.best_estimator_.steps[-1][1]
      self.results = pd.DataFrame(grid_search.cv_results_)

class ModelEvaluation(object):
    """ A class used to save a model and its evaluation results
    
    Parameters
    ----------
    name: string
        The name of a model
    model: object
        The model used to make predictions
    labels: list
        The original labels or target values
    predictions: list
        Predictions that were made by the model
    score:
        The score of 
        
    Returns
    -------
    numpy.array
        An (n x m+1) matrix where n = # of features, m = # of instances
    """
    __slots__ = ['name', 'model', 'labels', 'predictions', 'score']

    def __init__(self, model, labels, predictions, score):
        model = model
        labels = labels
        predictions = predictions
        score = score
    
    def evaluate(self, X, y):
        return ModelEvaluation.evaluate(self, X, y)