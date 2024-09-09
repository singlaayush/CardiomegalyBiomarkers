import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, callback
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, matthews_corrcoef, average_precision_score
from sklearn.impute import SimpleImputer
import numpy as np

def SplitData(df, splitFracs, class_col = 'class'):
    '''Split dataframe into smaller subsets which keep constant ratio of postive/negative samples. 
    Function retuns a list of DataFrames, where each new subset an element of list.  

    :param df: Dataframe to split
    :type df: pd.DataFrame
    :param splitFracs: list of floats which sum to 1 and relative size of subsets (splits) 
    :type splitFracs: list
    :param class_col: string to indicate which column defines binary classification, default - 'class'
    :type class_col: string

    :return: list of DataFrames of subsets
    :rtype: list of length len(splitFracs)
    '''

    # suffle df into random order
    df.sample(frac=1).reset_index(drop=True)

    # split df by class
    class1 = df.loc[df[class_col]==1]
    class0 = df.loc[df[class_col]==0]

    # initiate output list
    split_list = []

    # cycle through all split fractions and 
    for i in range(len(splitFracs)):
        
        # define split indexes for pos (class 1) samples
        start_class1 = round(sum(splitFracs[:i])*len(class1))
        end_class1 = round(sum(splitFracs[:(i+1)])*len(class1))

        # define split indexes for neg (class 0) samples
        start_class0 = round(sum(splitFracs[:i])*len(class0))
        end_class0 = round(sum(splitFracs[:(i+1)])*len(class0))        

        # collect pos and neg samples of split from class1 and class0 dataframes
        split_class1 = class1.iloc[start_class1:end_class1]
        split_class0 = class0.iloc[start_class0:end_class0]

        # concatenate pos and neg samples of split into new DataFrame
        split = pd.concat([split_class1, split_class0]).sample(frac=1).reset_index(drop=True)

        split_list.append(split)

    return split_list

def SplitDataCorrectly(df, splitFracs, class_col='class', subject_col='subject_id', random_seed=42):
    '''Split dataframe into smaller subsets which keep constant ratio of positive/negative samples,
    while ensuring that all samples with the same subject_id are in the same subset.
    
    :param df: DataFrame to split
    :type df: pd.DataFrame
    :param splitFracs: List of floats which sum to 1, representing the relative size of subsets (splits)
    :type splitFracs: list
    :param class_col: String to indicate which column defines binary classification, default - 'class'
    :type class_col: string
    :param subject_col: String to indicate which column defines the subject ID, default - 'subject_id'
    :type subject_col: string
    :param random_seed: Integer to set the random seed for reproducibility, default is None
    :type random_seed: int, optional
    
    :return: List of DataFrames of subsets
    :rtype: list of length len(splitFracs)
    '''

    # Set the random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Shuffle the dataframe randomly
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Calculate the target number of samples for each split
    total_samples = len(df)
    target_sizes = [int(frac * total_samples) for frac in splitFracs]

    # Group by subject_id
    groups = df.groupby(subject_col)

    # Initialize lists to store the splits
    splits = [[] for _ in range(len(splitFracs))]
    split_class_counts = np.zeros((len(splitFracs), 2))  # Track (class0, class1) counts per split
    split_sizes = np.zeros(len(splitFracs))  # Track the number of samples in each split

    # Iterate through each group and assign to a split
    for _, group in groups:
        # Count the number of positive and negative samples in the group
        num_class1 = group[class_col].sum()
        num_class0 = len(group) - num_class1

        # Calculate the best split for this group based on target size and class balance
        split_scores = [
            (target_sizes[i] - split_sizes[i]) - abs(
                (split_class_counts[i, 1] + num_class1) / (split_sizes[i] + len(group) + 1e-10) - sum(splitFracs)
            )
            for i in range(len(splitFracs))
        ]
        target_split = np.argmax(split_scores)

        # Assign group to the chosen split
        splits[target_split].append(group)
        split_class_counts[target_split, 0] += num_class0
        split_class_counts[target_split, 1] += num_class1
        split_sizes[target_split] += len(group)

    # Concatenate groups within each split and shuffle
    final_splits = [pd.concat(split).sample(frac=1, random_state=random_seed).reset_index(drop=True) for split in splits]

    return final_splits

def get_xgboost(feature_type, model_params):
    '''retreive model skeleton in form of xgboost.XGBClassifier object with model_param features 
    encoded. If using only biomarker features, few features are present reuireing to change max tree depth.

    :param features: string scribding types of features used to train model
    :type features: str
    :param model_params: dict of model parameters
    :type model_params: dict

    :return: model sekeleton with parameters encoded
    :rtype: xgboost.XGBClassifier object
    '''

    if feature_type == 'BMRK':
        # get model -> since using few features (2), use shallow max_depth
        model = XGBClassifier(eval_metric = model_params['eval_metric'], scale_pos_weight = model_params['scale_pos_weight'], 
                              colsample_bytree = model_params['colsample_bytree'], gamma = model_params['gamma'], subsample = model_params['subsample'], 
                              max_depth = model_params['max_depth_shallow'], learning_rate = model_params['lr'], 
                              callbacks=[callback.EvaluationMonitor(show_stdv=False), callback.EarlyStopping(model_params['early_stopping'])])

    else:
        # get model -> since using more features, use deeper max_depth
        model = XGBClassifier(eval_metric = model_params['eval_metric'], scale_pos_weight = model_params['scale_pos_weight'], 
                              colsample_bytree = model_params['colsample_bytree'], gamma = model_params['gamma'], subsample = model_params['subsample'], 
                              max_depth = model_params['max_depth_deep'], learning_rate = model_params['lr'], 
                              callbacks=[callback.EvaluationMonitor(show_stdv=False),callback.EarlyStopping(model_params['early_stopping'])])

    return model




def test_xgboost(model, test, features):
    '''Test xgboost model on 'test' and return preformance scores of accuracy, 
    AUC ROC, F1 score, and confusion matrix

    :param model: trained XGBoost model to be tested
    :type model: xgboost.XGBClassifier object
    :param test: DataFrame of test samples with features and ground truth classes
    :type test: pd.DataFrame
    :param features: list of features used in this model
    :type features: list

    :return accuracy: accuracy of predictions
    :rtype accuracy: float
    :return auc_roc: AUC ROC score of predictions
    :rtype auc_roc: float
    :return f1: F1 score of predictions
    :rtype f1: float
    :return cf: confusion matrix of predictions
    :rtype cf: ndarray of shape (2, 2)
    '''
    
    # make predictions on test set and make binary
    preds_raw = model.predict(test[features])
    preds = [round(value) for value in preds_raw]

    auc_roc = roc_auc_score(list(test['class']), preds_raw)
    avg_precision = average_precision_score(list(test['class']), preds_raw)

    # evaluate preductions
    accuracy = accuracy_score(list(test['class']), preds)
    cm = confusion_matrix(list(test['class']), preds)
    f1 = f1_score(list(test['class']), preds)
    tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0  # Sensitivity, Recall
    tnr = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0  # Specificity
    ppv = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0  # Precision
    npv = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0  # Negative Predictive Value
    mcc = matthews_corrcoef(list(test['class']), preds)

    return accuracy, auc_roc, f1, cm, avg_precision, tpr, tnr, ppv, npv, mcc




def train_test_xgboost(train_folds, val, valFoldNum, test, modalities_combinations, model_params, model_path, lossFigure, exportModels, select_col_groups=None, impute=False):
    '''Train xgboost models on train_folds with validaiotn set val, for differnt modality combinations. Then, if requested, 
    export loss figure and models to model_path folder. Subsequentally test model on test set and return DataFrame of summary
    performance scores.

    :param train_folds: list of k-1 DataFrame (each is one fold of data), in case where there are k folds which are combined into training set
    :type train_folds: list of pd.DataFrame's
    :param val: DataFrame of fold used for validation set
    :type val: pd.DataFrame
    :param valFoldNum: indicator of which fold is used for validation
    :type valFoldNum: int
    :param test: Dataframe of test set
    :type test: pd.DataFrame
    :param modalities_combinations: sublists of 2 elements containing: [list of features used in modality combination corresponding to second element, str of modality combiation]
    :type modalities_combinations: list of 2 element lists
    :param model_params: dictinoary descirbing xgboost model parameters
    :type model_params: dict
    :param model_path: file path to model and figure save folder
    :type model_path: str
    :param lossFigure: bool value to determine is training/validation loss figure is saved to model_path folder
    :type lossFigure: bool
    :param exportModels: bool value to determine if model is exported to model_path folder
    :type exportModels: bool

    :return results_df: DataFrame of preformance scores on test set for different modality combinations 
    :rtype results_df: pd.DataFrame 
    '''

    # combine training folds into single training set
    train = pd.concat(train_folds, axis=0).reset_index(drop=True)

    # Mean Imputation for NaNs in train set if desired
    if impute:
        imputer = SimpleImputer(strategy='mean')
        imputed_df = pd.DataFrame(imputer.fit_transform(train[select_col_groups]), columns=select_col_groups)
        train[select_col_groups] = imputed_df[select_col_groups].values
        test_imputed_df = pd.DataFrame(imputer.transform(test[select_col_groups]), columns=select_col_groups)
        test[select_col_groups] = test_imputed_df[select_col_groups].values

    results = []
    for _, combination in enumerate(modalities_combinations):

        # match features to targetoutputs
        eval_set = [(train[combination[0]], train['class']), (val[combination[0]], val['class'])]

        # get model skeleton
        model = get_xgboost(combination[1], model_params)

        # train model using validation set and call back methods to get best model
        model.fit(train[combination[0]], train['class'], eval_set = eval_set)
        
        # if lossFigure, save figure with trianing and validation losses to model_path
        if lossFigure:
            # retrieve performance metrics
            trainval_error = model.evals_result()
            epochs = len(trainval_error['validation_0']['logloss'])
            x_axis = range(0, epochs)

            # plot log loss
            fig, ax = plt.subplots()
            ax.plot(x_axis, trainval_error['validation_0']['logloss'], label='Train Error')
            ax.plot(x_axis, trainval_error['validation_1']['logloss'], label='Val Error')
            ax.legend()
            plt.ylabel('Log Loss')
            plt.title('XGBoost Log Loss for ' + combination[1])
            plt.savefig(model_path + combination[1] + '_fold' + str(valFoldNum) + '_loss.jpg')

        # if exportModels, export models to model_path 
        if exportModels:
            model.save_model(model_path + combination[1] + '_fold' + str(valFoldNum) + '_model.json')


        # get predictions and evaluate
        [val_accuracy, val_auc_roc, val_f1, val_cf, val_avg_precision, val_tpr, val_tnr, val_ppv, val_npv, val_mcc] = test_xgboost(model, val, combination[0])
        print(f'Val MCC for {combination[1]}: {val_mcc}')
        [accuracy, auc_roc, f1, cf, avg_precision, tpr, tnr, ppv, npv, mcc] = test_xgboost(model, test, combination[0])
        results.append([combination[1], accuracy, auc_roc, f1, cf, avg_precision, tpr, tnr, ppv, npv, mcc])

    # return results of all modality combinations tested
    results_df = pd.DataFrame(results, columns=['Modalities', 'Accuracy', 'ROC AUC', 'F1 score','Confusion Matrix', 'Average Precision', 'TPR', 'TNR', 'PPV', 'NPV', 'MCC']).set_index('Modalities')

    return results_df