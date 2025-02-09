""".
 
"""
__author__ = ("Bernhard Lehner <https://github.com/berni-lehner>")

import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedShuffleSplit

from bench_utils import (
    repeat_experiment,
    regression_benchmark,
    aabb_regression_benchmark)

from scoring_utils import (
    get_synth_regression_scoring,
    SHM_Scoring)

from cv_utils import (
    AnomalyShuffleSplit,
    RepeatedAnomalyShuffleSplit)

def conduct_ab_reg_experiment(X, y,
                              estimators,
                              n_splits=32,
                              test_size=0.1,
                              scoring=None,
                              random_state=None):
    
    n_repeats = 1
    cv = [list(StratifiedShuffleSplit(n_splits=32, test_size=0.1,
                                      random_state=random_state).split(X,y)).copy()
          for i in range(n_repeats)]

    results = aabb_regression_benchmark(X=X, y=y,
                                   models=estimators,
                                   cv=cv,
                                   scoring=scoring,
                                   random_state=random_state)
    
    shm_scoring = SHM_Scoring()

    shm_scoring.add_rel_error(df=results, mse_cols=shm_scoring.SYNTH_MSE_RESULTS)
    shm_scoring.add_abs_error(df=results, mse_cols=shm_scoring.SYNTH_MSE_RESULTS)

    return results


def conduct_ab_mixed_reg_experiment(Xsyn, ysyn,
                                    Xreal, yreal,
                                    estimators,
                                    n_splits=32,
                                    test_size=0.1,
                                    scoring=None,
                                    random_state=None):

    cv = AnomalyShuffleSplit(Xpos=Xsyn, Xneg=Xreal,
                             n_splits=n_splits,
                             test_size=test_size,
                             random_state=random_state,
                             unseen_only=True)
    
    # combine to full data set
    X = np.concatenate([Xsyn, Xreal], axis=0)
    y = np.concatenate([ysyn, yreal], axis=0)
    
    results = aabb_regression_benchmark(X=X, y=y,
                                   models=estimators,
                                   cv=cv,
                                   scoring=scoring,
                                   random_state=random_state)
    
    shm_scoring = SHM_Scoring()

    shm_scoring.add_rel_error(df=results, mse_cols=shm_scoring.REAL_MSE_RESULTS)
    shm_scoring.add_abs_error(df=results, mse_cols=shm_scoring.REAL_MSE_RESULTS)

    return results


def conduct_aa_reg_experiment(X, y,
                              estimator,
                              name,
                              n_repeats=5,
                              n_splits=32,
                              test_size=0.1,
                              scoring=None,
                              random_state=None):
    estimators = repeat_experiment(estimator=estimator, n_repeats=n_repeats, name=name)
    
    cv = [list(StratifiedShuffleSplit(n_splits=n_splits,
                                      test_size=test_size,
                                      random_state=random_state).split(X,y)).copy()
          for i in range(n_repeats)]

    results = regression_benchmark(X=X, y=y,
                                   models=estimators,
                                   cv=cv,
                                   scoring=scoring,
                                   random_state=random_state)
    
    shm_scoring = SHM_Scoring()

    shm_scoring.add_rel_error(df=results, mse_cols=shm_scoring.SYNTH_MSE_RESULTS)
    shm_scoring.add_abs_error(df=results, mse_cols=shm_scoring.SYNTH_MSE_RESULTS)

    return results


def conduct_aa_mixed_reg_experiment(Xsyn, ysyn,
                                 Xreal, yreal,
                                 estimator,
                                 name,
                                 n_repeats=5,
                                 n_splits=32,
                                 test_size=0.1,
                                 scoring=None,
                                 random_state=None):
    estimators = repeat_experiment(estimator=estimator, n_repeats=n_repeats, name=name)
    
    cv = RepeatedAnomalyShuffleSplit(Xpos=Xsyn, Xneg=Xreal,
                                     n_splits=n_splits,
                                     test_size=test_size,
                                     n_repeats=n_repeats,
                                     random_state=random_state,
                                     unseen_only=True)
    
    # combine to full data set
    X = np.concatenate([Xsyn, Xreal], axis=0)
    y = np.concatenate([ysyn, yreal], axis=0)
    
    results = regression_benchmark(X=X, y=y,
                                   models=estimators,
                                   cv=cv,
                                   scoring=scoring,
                                   random_state=random_state)
    
    shm_scoring = SHM_Scoring()

    shm_scoring.add_rel_error(df=results, mse_cols=shm_scoring.REAL_MSE_RESULTS)
    shm_scoring.add_abs_error(df=results, mse_cols=shm_scoring.REAL_MSE_RESULTS)

    return results