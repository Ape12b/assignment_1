
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle


def get_scorer(scoring, pos_label):
    """
    Create a custom scoring function for grid search
    """
    scorers = {"f1": f1_score,
    "precision": precision_score,
    "recall": recall_score}

    if scoring in ["f1", "recall", "precision"]:
      scorer = make_scorer(scorers[scoring], pos_label = pos_label)
    else:
      scorer = 'accuracy'
      
    return scorer

def grid_search_func(estimator, params, param_names,
                     cv=5, scoring='accuracy', 
                     X_train=None, X_test=None, 
                     y_train=None, y_test=None,
                     verbose=0, pos_label = None):
    """
    Perform grid search with cross-validation to find the best hyperparameters for the given estimator.

    Parameters:
    estimator (object): The estimator object to be tuned.
    params (list of dicts): List of dictionaries containing the hyperparameters to tune.
    param_names (list of str): List of hyperparameter names.
    cv (int, cross-validation generator, iterable, default=5): Determines the cross-validation splitting strategy.
    scoring (str or callable, default='accuracy'): Scoring metric to use for evaluation.
    X_train (array-like, optional): Training data.
    X_test (array-like, optional): Testing data.
    y_train (array-like, optional): Training labels.
    y_test (array-like, optional): Testing labels.
    verbose (int, default=0): Controls the verbosity of the grid search.
    pos_label (int, default=0): Controls the verbosity of the grid search.

    Returns:
    dict: Best parameters found during grid search.
    """
    # Define the hyperparameters grid
    param_grid = {param_name: param for param_name, param in zip(param_names, params)}

    # Get the scorer used for scoring the performance of the models      
    scorer = get_scorer(scoring, pos_label)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring=scorer, verbose=verbose,  n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    
    # Train the classifier with the best parameters
    clf = estimator.set_params(**best_params)
    clf.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate score
    score = get_score(y_test, y_pred, scoring, pos_label = pos_label)

    print("Best parameters:", best_params)
    print("Test accuracy:", score)
    return (best_params, grid_search.cv_results_)

def plot_train_test_metrics(train_scores, test_scores, params, param_names, scoring = 'Accuracy'):
    """
    Plot training and test metrics (e.g., accuracy) across different hyperparameter values.

    Parameters:
    train_scores (list of arrays): Training scores for each hyperparameter configuration.
    test_scores (list of arrays): Test scores for each hyperparameter configuration.
    params (list of arrays): Hyperparameter values.
    param_names (list of str): Names of hyperparameters.
    scoring (str): Scoring metric used

    Returns:
    None
    """
    num_params = len(param_names)
    fig, axs = plt.subplots(1, num_params, figsize=(num_params * 5, 5))
    
    
    for i in range(num_params):
        contains_strings = any(isinstance(element, str) for element in params[i])
        if not contains_strings:
            axs[i].plot(params[i], train_scores[i], marker="o", drawstyle="steps-post", label='Training Score')
            axs[i].plot(params[i], test_scores[i], marker="o", drawstyle="steps-post", label='Validation Score')
            axs[i].legend(loc='best')
            axs[i].set_title(f"Training and Test {scoring}: {param_names[i]}")
            axs[i].set_xlabel(param_names[i])
            axs[i].set_ylabel(scoring)
        else:
            # Set the width of the bars
            bar_width = 0.35

            # Set the positions of the bars on the x-axis
            x = np.arange(len(params[i]))
            
            plt.legend()
            axs[i].bar(x - bar_width/2, train_scores[i], bar_width, label='Training Score', )
            axs[i].bar(x + bar_width/2, test_scores[i],  bar_width, label='Validation Score')
            axs[i].legend(loc='best')
            axs[i].set_title(f"Training and Test {scoring}: {param_names[i]}")
            axs[i].set_xlabel(param_names[i])
            axs[i].set_ylabel(scoring)
            axs[i].set_xticks(x, params[i])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plot
    plt.show()

def plot_learning_curves(X_train, X_test, y_train, y_test, 
                     clfs,
                     names,
                     plot_train=False,
                     cv=10, 
                     scoring = "f1",
                     pos_label = "M"):
    """
    Plot learning curves for multiple classifiers.

    Parameters:
    X_train (array-like): Training data.
    X_test (array-like): Testing data.
    y_train (array-like): Training labels.
    y_test (array-like): Testing labels.
    clfs (list of classifiers): List of classifier instances.
    plot_train (bool, default=True): Whether to plot training scores.
    cv (int): Determines the cross-validation splitting strategy.
    names (list of str, optional): List of names for each classifier.

    Returns:
    predict_times: Time taken to predict on the whole train data.
    fit_times: Time taken to fit on the whole train data.
    train_scores: Training scores.
    test_scores: Testing scores.
    """
    i = 0
    num_colors = len(clfs) * 2
    cmap = plt.get_cmap('viridis', num_colors)
    rgb_values = [cmap(k)[:3] for k in np.linspace(0, 1, num_colors)]
       
    fig, axs = plt.subplots(1, len(clfs), figsize=(5 * len(clfs), 5))
    
    # Adjust layout to make room for the title
    plt.subplots_adjust(top=0.85)  # Increase top margin
    
    # Setting a common y_lim
    for ax in axs:
        ax.set_ylim(0.5, 1)
    for j, clf in enumerate(clfs):
        
        # Calculating the train_score, test_scores and times for the classifiers used
        train_scores = []
        test_scores = []
        fit_times = []
        predict_times = []
        train_sizes = [int(k * len(X_train)) for k in np.linspace(0.1, 1.0, 10)]
        
        for size in train_sizes:
            train_scores_cv = []
            fit_times_cv = []
            predict_times_cv = []
            test_scores_cv = []
            
            for _ in range(cv):
                train_data, train_label = shuffle(X_train, y_train, random_state=0)
                start_time = time.time()
                clf.fit(train_data[:size], train_label[:size])
                end_time = time.time()
                
                fit_times_cv.append(end_time - start_time)
                
                start_time = time.time()
                y_pred = clf.predict(X_train)
                end_time = time.time()
                
                predict_times_cv.append(end_time - start_time)
                
                train_scores_cv.append(get_score(y_train, y_pred, scoring, pos_label))
                
                y_pred = clf.predict(X_test)
                test_scores_cv.append(get_score(y_test, y_pred, scoring, pos_label))
            train_scores.append(train_scores_cv)
            test_scores.append(test_scores_cv)
            fit_times.append(fit_times_cv)
            predict_times.append(predict_times_cv)
            
        # Calculating mean and standard error of mean for training and validation scores
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_sem = np.std(train_scores, axis=1) / np.sqrt(len(train_scores))
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_sem = np.std(test_scores, axis=1) / np.sqrt(len(test_scores))
        
        if names is None:
            clf_name = set([clf.__class__.__name__ for clf in clfs])
            title = " vs. \n".join([name for name in clf_name])
            label_train = f'Training Score {clf.__class__.__name__}'
            label_val = f'Validation Score {clf.__class__.__name__}'
        else:
            title = " vs. \n".join([name for name in names])
            label_train = f'Training Score: {names[j].split(":")[0]}'
            label_val = f'Validation Score: {names[j].split(":")[0]}'
        

        if plot_train:
            axs[j].plot(train_sizes, train_scores_mean, label=label_train, color=rgb_values[i], marker='o')
            axs[j].fill_between(train_sizes, train_scores_mean - train_scores_sem, train_scores_mean + train_scores_sem, alpha=0.2, color=rgb_values[i])

        axs[j].plot(train_sizes, test_scores_mean, label=label_val, color=rgb_values[i + 1], marker='o')
        axs[j].fill_between(train_sizes, test_scores_mean - test_scores_sem, test_scores_mean + test_scores_sem, alpha=0.2, color=rgb_values[i + 1])
        
        axs[j].set_xlabel('Number of Training Samples')
        axs[j].set_ylabel(scoring)
        axs[j].legend(loc='best')
        axs[j].grid(True)
        
        i += 2
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{clf.__class__.__name__} time: {elapsed_time:.4f} seconds")
        
    fig.suptitle(title)
    plt.show()
    
    return(predict_times, fit_times, train_scores, test_scores)
    
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_score(y_true, y_pred, score_name, pos_label = 'M'):
    """
    Calculate and return different scores based on the passed string name of the score.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    score_name (str): Name of the score ('accuracy', 'precision', 'recall', 'f1').

    Returns:
    float: Calculated score.
    """
    if score_name == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif score_name == 'precision':
        return precision_score(y_true, y_pred, pos_label = pos_label)
    elif score_name == 'recall':
        return recall_score(y_true, y_pred, pos_label = pos_label)
    elif score_name == 'f1':
        return f1_score(y_true, y_pred, pos_label = pos_label)
    else:
        raise ValueError("Invalid score name. Please choose from 'accuracy', 'precision', 'recall', or 'f1'.")

def train_clfs_with_hyperparameters(model, param_name, param_values, X_train, y_train):
    """
    Train classifiers with different values of a hyperparameter.

    Parameters:
    model (class): The classifier model to be tuned.
    param_name (str): Name of the hyperparameter.
    param_values (list): List of values for the hyperparameter.
    X_train (array-like): Training data.
    y_train (array-like): Training labels.

    Returns:
    list: List of trained classifiers with different hyperparameter values.
    """
    clfs = []
  
    # Loop through values of the hyperparameter
    for value in param_values:
        params = {param_name: value}
        clf = model(**params)
        clf.fit(X_train, y_train)
        clfs.append(clf)
        
    return clfs


# def plot_learning_curves(X_train, X_test, y_train, y_test, clfs, plot_train=True, cv=5, names=None, scoring="accuracy"):
#     """
#     Plot learning curves for multiple classifiers.

#     Parameters:
#     X_train (array-like): Training data.
#     X_test (array-like): Testing data.
#     y_train (array-like): Training labels.
#     y_test (array-like): Testing labels.
#     clfs (list of classifiers): List of classifier instances.
#     plot_train (bool, default=True): Whether to plot training scores.
#     cv (int, cross-validation generator, iterable, optional, default=5): Determines the cross-validation splitting strategy.
#     names (list of str, optional): List of names for each classifier.

#     Returns:
#     None
#     """
#     i = 0
#     j = 0
#     num_colors = len(clfs) * 2
#     cmap = plt.get_cmap('viridis', num_colors)
#     rgb_values = [cmap(i)[:3] for i in np.linspace(0, 1, num_colors)]

#     plt.figure(figsize=(15, 9))
#     for clf in clfs:
#         start_time = time.time()
#         if len(set(y_train)) > 2: # For multi-class classifications using 'neg_log_loss
#             train_sizes, train_scores, val_scores = learning_curve(clf, X_train, y_train, cv=cv,
#                                                                     scoring='neg_log_loss', train_sizes=np.linspace(0.1, 1.0, 10))
#             train_scores = -train_scores
#             val_scores = -val_scores
#         else:
#             train_sizes, train_scores, val_scores = learning_curve(clf, X_train, y_train, cv=cv, 
#                                                                    scoring = scoring, train_sizes=np.linspace(0.1, 1.0, 10))

#         # Calculating mean and standard error of mean
#         train_scores_mean = np.mean(train_scores, axis=1)
#         train_scores_sem = np.std(train_scores, axis=1) / np.sqrt(len(train_scores))
#         val_scores_mean = np.mean(val_scores, axis=1)
#         val_scores_sem = np.std(val_scores, axis=1) / np.sqrt(len(val_scores))

#         if names is None:
#             clf_name = set([clf.__class__.__name__ for clf in clfs])
#             title = " vs. \n".join([name for name in clf_name])
#             label_train = f'Training Score {clf.__class__.__name__}'
#             label_val = f'Validation Score {clf.__class__.__name__}'
#         else:
#             title = " vs. \n".join([name for name in names])
#             label_train = f'Training Score: {names[j]}'
#             label_val = f'Validation Score: {names[j]}'
#             j += 1

#         if plot_train:
#             plt.plot(train_sizes, train_scores_mean, label=label_train, color=rgb_values[i], marker='o')
#             plt.fill_between(train_sizes, train_scores_mean - train_scores_sem, train_scores_mean + train_scores_sem, alpha=0.2, color=rgb_values[i])

#         plt.plot(train_sizes, val_scores_mean, label=label_val, color=rgb_values[i + 1], marker='o')
#         plt.fill_between(train_sizes, val_scores_mean - val_scores_sem, val_scores_mean + val_scores_sem, alpha=0.2, color=rgb_values[i + 1])
#         i += 2

#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         print(f"{clf.__class__.__name__} time: {elapsed_time:.4f} seconds")

#     plt.title(title)
#     plt.xlabel('Number of Training Samples')
#     plt.ylabel('Accuracy' if len(set(y_train)) == 2 else 'Negative Log Loss')
#     plt.legend(loc='best')
#     plt.grid(True)
#     plt.show()