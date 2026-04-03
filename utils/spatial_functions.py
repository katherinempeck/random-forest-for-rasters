import pandas as pd

import rasterio

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import numpy as np

import time
from datetime import timedelta

def get_bands(in_raster, num_bands):
    """
    Reads in multi-band analysis raster and returns a list of all bands (as numpy arrays)

    Parameters
    ----------
    in_raster : string
        analysis raster file path
    num_bands : int
        number of bands in the raster

    Returns
    -------
    list
        list of opened raster bands (numpy arrays)
    """
    out_list = []
    for i in range(num_bands):
        bnum = i + 1
        with rasterio.open(in_raster) as src:
            out_list.append(src.read(bnum))
    return out_list

def get_raster_vals(pa_raster, data_raster):
    """
    For a binary presence or absence raster, get the values from the data_raster associated with each of those points;
    called by build_pa_df() to create analysis DataFrame from data raster(s)

    Parameters
    ----------
    pa_raster : numpy array
        binary array representing presence or absence data for model training
    data_raster : numpy array
        array representing a given feature (e.g., slope)

    Returns
    -------
    list
        list of data_raster values associated with presence/absence values
    """
    return data_raster[np.where(pa_raster == 1)].tolist()

def build_pa_df(feature_list, presence_array, absence_array, feature_columns):
    """
    Creates analysis dataframe from the presence raster, absence raster, and feature rasters

    Parameters
    ----------
    feature_list : list
        list of all features (opened rasters) that will be used to train model
        output of get_bands(), excluding presence/absence bands, if using multiband raster
        otherwise provide list of individually opened rasters, e.g.,
            slope = rasterio.open('data/slope.tif').read(1)
            aspect = rasterio.open('data/aspect.tif').read(1)
            feature_list = [slope, aspect]
    presence_array : array
        opened raster representing presence data
    absence_array : array
        opened raster representing absence data
    feature_columns : list
        list of strings representing the desired names of each feature in the output array, e.g.,
            feature_list = ['slope', 'aspect']
    Returns
    -------
    pd.DataFrame
        pandas dataframe with presence/absence column and feature data for each observation
    """
    #Start with presence data
    feature_columns.insert(0, 'PA')
    presence_data_list = [[1]*len(presence_array)]
    for f in feature_list:
        presence_data_list.append(get_raster_vals(presence_array, f))
    presence_df = pd.DataFrame(zip(*presence_data_list))
    presence_df = presence_df.rename(columns = dict(zip(presence_df.columns, feature_columns)))
    absence_data_list = [[0]*len(absence_array)]
    for f in feature_list:
        absence_data_list.append(get_raster_vals(absence_array, f))
    absence_df = pd.DataFrame(zip(*absence_data_list))
    absence_df = absence_df.rename(columns = dict(zip(absence_df.columns, feature_columns)))
    return pd.concat([presence_df, absence_df])

def split_dataset(df, feature_name_list, PA_col = 'PA', test_size = 0.2):
    """
    Splits the final analysis dataframe into training and testing datasets (called by train_predict_results())

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all feature data and a column representing whether a given observation (row) represents presence or absence
    feature_name_list : list
        list of feature names in the DataFrame
    PA_col : str
        name of the column holding the presence/absence data ('PA' by default based on output of build_pa_df())
    test_size: float
        percentage of dataset to hold back to form the test dataset (0.2 by default)
    Returns
    -------
    list
        training data independent variables, testing data independent variables, training data dependent variables (i.e., presence/absence), testing data dependent variables
    """
    X = df[feature_name_list].values
    Y = df[[PA_col]].values
    return train_test_split(X, Y, test_size = test_size)

def train_predict_results(df, feature_name_list, PA_col = 'PA', test_size = 0.2, training_metrics = False, testing_metrics = True, **kwargs):
    """
    Train a random forest classifier on the input data and print performance metrics. Can also take any additional arguments associated with the sklearn RandomForestClassifier().

    Parameters
    ----------
    df : pd.DataFrame
        analysis dataframe with feature columns and a presence/absence column
    feature_name_list : list
        list of names (as strings) of columns in the dataframe that represent the data features
    PA_col : string
        name of presence/absence column ('PA' by default based on output of build_pa_df())
    test_size : float
        percentage of data to hold back for testing raster
    training_metrics : bool
        if True, print accuracy and classification report for training dataset
    testing_metrics : bool
        if True, print classification report and confusion matrix results for testing dataset
    Returns
    -------
    list
        trained random forest classifier, testing dataset x, testing dataset y
    """
    X_train, x_test, Y_train, y_test = split_dataset(df, feature_name_list, PA_col, test_size)
    rforest = RandomForestClassifier(**kwargs)
    print(f'Training on {len(X_train)} observations')
    rforest.fit(X_train, Y_train.ravel())
    print('Model training complete!')
    if training_metrics == True:
        print(f"Model Training Accuracy: {rforest.score(X_train, Y_train):.3f}")
        y_train_pred = rforest.predict(X_train)
        print('Training Dataset Classification Report:')
        print(classification_report(Y_train, y_train_pred))
    if testing_metrics == True:
        y_pred = rforest.predict(x_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel().tolist()
        print(f'\nResults on test dataset:\nTrue Positives = {tp}, False Positives = {fp}, True Negatives = {tn}, False Negatives = {fn}\n')
        print('Test Dataset Classification Report:')
        print(classification_report(y_test, y_pred))
    return rforest, x_test, y_test

def evaluate_on_array(data, classifier):
    """
    Apply random forest classifier to array;
    applied in vectorized operation to stacked raster data (see predict_on_mb_raster() and predict_on_rasters())

    Parameters
    ----------
    data : array
        one 2d array within stacked arrays representing feature data
    classifier : sklearn.ensemble.RandomForestClassifier
        trained model (output of train_predict_results())

    Returns
    -------
    list
        trained random forest classifier, testing dataset x, testing dataset y
    """
    data = [data]
    return classifier.predict(data)

def predict_on_mb_raster(prediction_raster_path, num_bands, classifier):
    """
    Apply random forest classifier to new raster data

    Parameters
    ----------
    prediction_raster_path : string
        file path to multi-band raster with the same feature bands as the training data (but no presence/absence rasters)
    num_bands : integer
        number of bands in raster
    classifier : sklearn.ensemble.RandomForestClassifier
        trained model (output of train_predict_results())

    Returns
    -------
    array
        numpy array representing predictions for every pixel in the input prediction raster
    """
    feature_list = get_bands(prediction_raster_path, num_bands)
    stacked = np.stack(feature_list)
    print(f'Predicting on {stacked.shape[1]*stacked.shape[2]} observations')
    start = time.time()
    out = np.apply_along_axis(lambda x: evaluate_on_array(x, classifier=classifier), 0, stacked)
    end = time.time()
    print('Complete!')
    print(f'{str(timedelta(seconds = end - start))} elapsed')
    return out

def predict_on_rasters(read_raster_list_ordered, classifier):
    """
    Apply random forest classifier to new raster data

    Parameters
    ----------
    read_raster_list_ordered : list
        list of aligned rasters with the same resolution (for situations where data are not all in a single multi-band raster)
    classifier : sklearn.ensemble.RandomForestClassifier
        trained model (output of train_predict_results())

    Returns
    -------
    array
        numpy array representing predictions for every pixel in the input prediction raster
    """
    stacked = np.stack(read_raster_list_ordered)
    print(f'Predicting on {stacked.shape[1]*stacked.shape[2]} observations')
    start = time.time()
    out = np.apply_along_axis(lambda x: evaluate_on_array(x, classifier=classifier), 0, stacked)
    end = time.time()
    print('Complete!')
    print(f'{str(timedelta(seconds = end - start))} elapsed')
    return out

def save_raster_prediction(prediction_raster_path, prediction_out_path, prediction):
    """
    Writes prediction (output of predict_on_mb_raster()/predict_on_rasters()) to single-band, binary raster with same resolution/crs/extent as original

    Parameters
    ----------
    prediction_raster_path : string
        file path of original raster to which classifier as applied
    prediction_out_path : string
        file path at which new prediction raster will be saved
    """
    original = rasterio.open(prediction_raster_path)
    out_raster = rasterio.open(prediction_out_path, 'w', driver='GTiff',
                                height = original.shape[0], width = original.shape[1],
                                count=1, 
                                dtype=str(original.dtypes[0]),
                                crs = original.crs,
                                transform=original.transform)
    out_raster.write(prediction)
    out_raster.close()