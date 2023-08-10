#!/usr/bin/env python

import numpy as np
import pickle
from redshift_utils import *
from tqdm import tqdm, trange
import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
from astropy.table import Table
import forestci as fci


def main():
    # Read files

    atlas_file = "../Data/ATLAS_Corrected.fits"
    stripe_file = "../Data/Stripe_Corrected.fits"
    sdss_file = "../Data/RGZ_Corrected.fits"
    bootes_file = "../Data/Bootes_Corrected.fits"
    cosmos_file = "../Data/COSMOS_Corrected.fits"
    en1_file = "../Data/EN1_Corrected.fits"
    pred_file = "../Data/EMU_PS_Clean.fits"
    pred_file_allwise = "../Data/EMU_PS_Clean_AllWISE_2.fits"

    sdss_cols_allwise = [
        "zspec",
        "g_corrected",
        "r_corrected",
        "i_corrected",
        "z_corrected",
        "W1Mag",
        "W2Mag",
        "W3Mag",
        "W4Mag",
    ]
    sdss_cols_catwise = [
        "zspec",
        "g_corrected",
        "r_corrected",
        "i_corrected",
        "z_corrected",
        "W1Mag",
        "W2Mag",
    ]
    des_cols_catwise = [
        "zspec",
        "mag_auto_g",
        "mag_auto_r",
        "mag_auto_i",
        "mag_auto_z",
        "W1Mag",
        "W2Mag",
    ]
    des_cols_allwise = [
        "zspec",
        "mag_auto_g",
        "mag_auto_r",
        "mag_auto_i",
        "mag_auto_z",
        "W1Mag",
        "W2Mag",
        "W3Mag",
        "W4Mag",
    ]

    pred_cols = [
        "mag_auto_g",
        "mag_auto_r",
        "mag_auto_i",
        "mag_auto_z",
        "w1mag",
        "w2mag",
    ]
    pred_allwise_cols = [
        "mag_auto_g",
        "mag_auto_r",
        "mag_auto_i",
        "mag_auto_z",
        "W1mag_x",
        "W2mag_x",
        "W3mag",
        "W4mag",
    ]

    full_table_catwise = Table.read(pred_file).to_pandas()
    full_table_allwise = Table.read(pred_file_allwise).to_pandas()

    atlas_data_catwise = read_fits(atlas_file, des_cols_catwise)
    stripe_data_catwise = read_fits(stripe_file, des_cols_catwise)
    sdss_data_catwise = read_fits(sdss_file, sdss_cols_catwise)
    cosmos_data_catwise = read_fits(cosmos_file, sdss_cols_catwise)
    bootes_data_catwise = read_fits(bootes_file, sdss_cols_catwise)
    en1_data_catwise = read_fits(en1_file, sdss_cols_catwise)

    atlas_data_allwise = read_fits(atlas_file, des_cols_allwise)
    stripe_data_allwise = read_fits(stripe_file, des_cols_allwise)
    sdss_data_allwise = read_fits(sdss_file, sdss_cols_allwise)
    cosmos_data_allwise = read_fits(cosmos_file, sdss_cols_allwise)
    bootes_data_allwise = read_fits(bootes_file, sdss_cols_allwise)
    en1_data_allwise = read_fits(en1_file, sdss_cols_allwise)

    combined_train_data_catwise = np.vstack(
        (
            atlas_data_catwise,
            stripe_data_catwise,
            sdss_data_catwise,
            cosmos_data_catwise,
            bootes_data_catwise,
            en1_data_catwise,
        )
    )
    combined_train_data_allwise = np.vstack(
        (
            atlas_data_allwise,
            stripe_data_allwise,
            sdss_data_allwise,
            cosmos_data_allwise,
            bootes_data_allwise,
            en1_data_allwise,
        )
    )

    x_vals_emu_catwise = read_fits(pred_file, pred_cols)
    x_vals_emu_allwise = read_fits(pred_file_allwise, pred_allwise_cols)

    x_vals_catwise = combined_train_data_catwise[:, 1:]
    y_vals_catwise = combined_train_data_catwise[:, 0]

    x_vals_allwise = combined_train_data_allwise[:, 1:]
    y_vals_allwise = combined_train_data_allwise[:, 0]

    overall_start = datetime.now()
    # Regression tests
    ## Initial run - finds value of "k" to use, and generates plots.
    # folder_name = "kNN_Regress"

    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)
    # os.chdir(folder_name)

    ## Find the best value of k to use
    k_range = range(3, 31, 2)
    k_fold_val = 5
    random_seed_kfold = 42
    out_rates_catwise = []
    out_rates_allwise = []
    for k_val in tqdm(k_range):
        out_rates_catwise.append(
            kNN_cross_val(
                k_val, k_fold_val, random_seed_kfold, x_vals_catwise, y_vals_catwise
            )
        )
        out_rates_allwise.append(
            kNN_cross_val(
                k_val, k_fold_val, random_seed_kfold, x_vals_allwise, y_vals_allwise
            )
        )

    best_k_catwise = k_range[np.argmin(out_rates_catwise)]
    best_k_allwise = k_range[np.argmin(out_rates_allwise)]

    # best_k_catwise = 3
    # best_k_allwise = 3

    print("Outlier rates for the cross-validation - CatWISE")
    print(out_rates_catwise)
    print("Outlier rates for the cross-validation - AllWISE")
    print(out_rates_allwise)

    ## Use best k for rest.

    start_time = datetime.now()
    x_vals_norm_catwise, x_vals_emu_norm_catwise, _, _ = norm_x_vals(
        x_vals_catwise, x_vals_emu_catwise
    )
    x_vals_norm_allwise, x_vals_emu_norm_allwise, _, _ = norm_x_vals(
        x_vals_allwise, x_vals_emu_allwise
    )

    pred_catwise, model_catwise = kNN_pred(
        best_k_catwise, x_vals_norm_catwise, x_vals_emu_norm_catwise, y_vals_catwise
    )
    pred_allwise, model_allwise = kNN_pred(
        best_k_allwise, x_vals_norm_allwise, x_vals_emu_norm_allwise, y_vals_allwise
    )

    catwise_distances, catwise_indices = model_catwise.kneighbors(
        x_vals_emu_norm_catwise
    )
    allwise_distances, allwise_indices = model_allwise.kneighbors(
        x_vals_emu_norm_allwise
    )

    variances_catwise = []
    for row in tqdm(catwise_indices):
        train_predictions = model_catwise.predict(x_vals_norm_catwise[row, :])
        variances_catwise.append(np.var(train_predictions))

    variances_allwise = []
    for row in tqdm(allwise_indices):
        train_predictions = model_allwise.predict(x_vals_norm_allwise[row, :])
        variances_allwise.append(np.var(train_predictions))

    prediction_filename_catwise = "predictions_catwise.csv"
    df = pd.DataFrame(
        {
            "EMU_island_id": full_table_catwise["island_id"],
            "EMU_component_id": full_table_catwise["component_id"],
            "EMU_component_name": full_table_catwise["component_name"],
            "Pred_z": pred_catwise,
            "Uncertainty": variances_catwise,
        }
    )
    df.to_csv(prediction_filename_catwise, index=False)
    model_filename = "model_catwise.pickle"
    with open(model_filename, "wb") as pickle_file:
        pickle.dump(model_catwise, pickle_file)

    prediction_filename = "predictions_allwise.csv"
    df = pd.DataFrame(
        {
            "EMU_island_id": full_table_allwise["island_id"],
            "EMU_component_id": full_table_allwise["component_id"],
            "EMU_component_name": full_table_allwise["component_name"],
            "Pred_z": pred_allwise,
            "Uncertainty": variances_allwise,
        }
    )
    df.to_csv(prediction_filename, index=False)
    model_filename = "model_allwise.pickle"
    with open(model_filename, "wb") as pickle_file:
        pickle.dump(model_allwise, pickle_file)

    print("##################")
    os.chdir("../")
    folder_name = "rf_Regress"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    os.chdir(folder_name)

    # Regression tests
    ## Initial run - finds value of "k" to use, and generates plots.
    overall_start = datetime.now()
    # x_vals_train, x_vals_test, _, _ = norm_x_vals(x_vals_train, x_vals_test)

    ## Find the best value of k to use
    tree_range = range(2, 201, 1)
    k_fold_val = 5
    random_seed_kfold = 42
    out_rates_catwise = []
    out_rates_allwise = []

    for tree_val in tqdm(tree_range):
        out_rates_catwise.append(
            rf_cross_val(
                tree_val, k_fold_val, x_vals_catwise, y_vals_catwise, random_seed_kfold
            )
        )
        out_rates_allwise.append(
            rf_cross_val(
                tree_val, k_fold_val, x_vals_allwise, y_vals_allwise, random_seed_kfold
            )
        )
    best_tree_catwise = tree_range[np.argmin(out_rates_catwise)]
    best_tree_allwise = tree_range[np.argmin(out_rates_allwise)]
    # best_tree_catwise = 170
    # best_tree_allwise = 173
    print("Best Tree Value | Best Outlier Rate -- CatWISE")
    print(f"{best_tree_catwise} | {np.round(np.min(out_rates_catwise),2)}")
    # 170 | 9.64%
    print("Best Tree Value | Best Outlier Rate -- AllWISE")
    print(f"{best_tree_allwise} | {np.round(np.min(out_rates_allwise),2)}")
    # 173 | 9.89%

    x_vals_norm_catwise, x_vals_emu_norm_catwise, _, _ = norm_x_vals(
        x_vals_catwise, x_vals_emu_catwise
    )
    x_vals_norm_allwise, x_vals_emu_norm_allwise, _, _ = norm_x_vals(
        x_vals_allwise, x_vals_emu_allwise
    )

    pred_catwise, model_catwise = rf_pred(
        best_tree_catwise, x_vals_norm_catwise, x_vals_emu_norm_catwise, y_vals_catwise
    )
    pred_allwise, model_allwise = rf_pred(
        best_tree_allwise, x_vals_norm_allwise, x_vals_emu_norm_allwise, y_vals_allwise
    )

    uncert_catwise = fci.random_forest_error(
        model_catwise, x_vals_norm_catwise, x_vals_emu_norm_catwise
    )
    uncert_allwise = fci.random_forest_error(
        model_allwise, x_vals_norm_allwise, x_vals_emu_norm_allwise
    )

    uncert_allwise = np.sqrt(uncert_allwise)
    uncert_catwise = np.sqrt(uncert_catwise)

    prediction_filename_catwise = "predictions_catwise.csv"
    df = pd.DataFrame(
        {
            "EMU_island_id": full_table_catwise["island_id"],
            "EMU_component_id": full_table_catwise["component_id"],
            "EMU_component_name": full_table_catwise["component_name"],
            "Pred_z": pred_catwise,
            "Uncertainty": uncert_catwise,
        }
    )

    # df = pd.DataFrame({"Pred_z" : pred_catwise})
    df.to_csv(prediction_filename_catwise, index=False)
    model_filename = "model_catwise.pickle"
    with open(model_filename, "wb") as pickle_file:
        pickle.dump(model_catwise, pickle_file)

    prediction_filename = "predictions_allwise.csv"
    # df = pd.DataFrame({"Pred_z" : pred_allwise})
    df = pd.DataFrame(
        {
            "EMU_island_id": full_table_allwise["island_id"],
            "EMU_component_id": full_table_allwise["component_id"],
            "EMU_component_name": full_table_allwise["component_name"],
            "Pred_z": pred_allwise,
            "Uncertainty": uncert_allwise,
        }
    )
    df.to_csv(prediction_filename, index=False)
    model_filename = "model_allwise.pickle"
    with open(model_filename, "wb") as pickle_file:
        pickle.dump(model_allwise, pickle_file)


if __name__ == "__main__":
    main()
