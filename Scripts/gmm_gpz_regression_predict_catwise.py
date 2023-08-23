from datetime import datetime

from matplotlib.pyplot import legend
from redshift_utils import *
import subprocess, argparse, os
import numpy as np
import pandas as pd
from astropy.table import Table, vstack
from sklearn.mixture import GaussianMixture
from collections import OrderedDict
import subprocess


def main():
    # Default GPz options:
    gpz_opts = OrderedDict()
    gpz_opts["VERBOSE"] = 1
    gpz_opts["N_THREAD"] = 4
    gpz_opts["TRAINING_CATALOG"] = "dummy"
    gpz_opts["PREDICTION_CATALOG"] = "dummy"
    gpz_opts["BANDS"] = "[mag]"
    gpz_opts["FLUX_COLUMN_PREFIX"] = "mag_"
    gpz_opts["ERROR_COLUMN_PREFIX"] = "magerr_"
    gpz_opts["OUTPUT_COLUMN"] = "z_spec"
    gpz_opts["WEIGHT_COLUMN"] = ""
    gpz_opts["WEIGHTING_SCHEME"] = "uniform"
    gpz_opts["OUTPUT_MIN"] = 0
    gpz_opts["OUTPUT_MAX"] = 7
    gpz_opts["USE_ERRORS"] = 1
    gpz_opts["TRANSFORM_INPUTS"] = "no"
    gpz_opts["NORMALIZATION_SCHEME"] = "whiten"
    gpz_opts["VALID_SAMPLE_METHOD"] = "random"
    gpz_opts["TRAIN_VALID_RATIO"] = 0.7
    gpz_opts["VALID_SAMPLE_SEED"] = 42
    gpz_opts["OUTPUT_CATALOG"] = "dummy"
    gpz_opts["MODEL_FILE"] = "dummy"
    gpz_opts["SAVE_MODEL"] = 1
    gpz_opts["REUSE_MODEL"] = 0
    gpz_opts["USE_MODEL_AS_HINT"] = 0
    gpz_opts["PREDICT_ERROR"] = 1
    gpz_opts["NUM_BF"] = 50
    gpz_opts["COVARIANCE"] = "gpvd"
    gpz_opts["PRIOR_MEAN"] = "constant"
    gpz_opts["OUTPUT_ERROR_TYPE"] = "input_dependent"
    gpz_opts["BF_POSITION_SEED"] = 55
    gpz_opts["FUZZING"] = 0
    gpz_opts["FUZZING_SEED"] = 42
    gpz_opts["MAX_ITER"] = 500
    gpz_opts["TOLERANCE"] = 1e-9
    gpz_opts["GRAD_TOLERANCE"] = 1e-5

    atlas_file = "../../Data/ATLAS_Corrected.fits"
    stripe_file = "../../Data/Stripe_Corrected.fits"
    sdss_file = "../../Data/RGZ_Corrected.fits"
    bootes_file = "../../Data/Bootes_Corrected.fits"
    cosmos_file = "../../Data/COSMOS_Corrected.fits"
    en1_file = "../../Data/EN1_Corrected.fits"
    pred_file = "../../Data/EMU_PS_Clean.fits"

    sdss_cols_catwise = [
        "zspec",
        "g_corrected",
        "r_corrected",
        "i_corrected",
        "z_corrected",
        "g_corrected_err",
        "r_corrected_err",
        "i_corrected_err",
        "z_corrected_err",
        "W1Mag",
        "W2Mag",
        "e_W1Mag",
        "e_W2Mag",
    ]
    des_cols_catwise = [
        "zspec",
        "mag_auto_g",
        "mag_auto_r",
        "mag_auto_i",
        "mag_auto_z",
        "magerr_auto_g",
        "magerr_auto_r",
        "magerr_auto_i",
        "magerr_auto_z",
        "W1Mag",
        "W2Mag",
        "e_W1Mag",
        "e_W2Mag",
    ]

    pred_cols = [
        "mag_auto_g",
        "mag_auto_r",
        "mag_auto_i",
        "mag_auto_z",
        "magerr_auto_g",
        "magerr_auto_r",
        "magerr_auto_i",
        "magerr_auto_z",
        "w1mag",
        "w2mag",
        "w2sigm",
        "w2sigm",
    ]

    full_table_catwise = Table.read(pred_file).to_pandas()

    atlas_data_catwise = read_fits(atlas_file, des_cols_catwise)
    stripe_data_catwise = read_fits(stripe_file, des_cols_catwise)
    sdss_data_catwise = read_fits(sdss_file, sdss_cols_catwise)
    cosmos_data_catwise = read_fits(cosmos_file, sdss_cols_catwise)
    bootes_data_catwise = read_fits(bootes_file, sdss_cols_catwise)
    en1_data_catwise = read_fits(en1_file, sdss_cols_catwise)

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

    x_vals_emu_catwise = read_fits(pred_file, pred_cols)

    combined_train_data_catwise[
        np.where(combined_train_data_catwise[:, -2] == -99), -2
    ] = np.nanmax(combined_train_data_catwise[:, -2])
    combined_train_data_catwise[
        np.where(combined_train_data_catwise[:, -1] == -99), -1
    ] = np.max(combined_train_data_catwise[:, -1])

    combined_train_data_catwise[
        np.where(np.isnan(combined_train_data_catwise[:, -2])), -2
    ] = np.nanmax(combined_train_data_catwise[:, -2])
    combined_train_data_catwise[
        np.where(np.isnan(combined_train_data_catwise[:, -1])), -1
    ] = np.nanmax(combined_train_data_catwise[:, -1])

    x_vals_emu_catwise[np.where(x_vals_emu_catwise[:, -2] == -99), -2] = np.nanmax(
        x_vals_emu_catwise[:, -2]
    )
    x_vals_emu_catwise[np.where(x_vals_emu_catwise[:, -1] == -99), -1] = np.max(
        x_vals_emu_catwise[:, -1]
    )

    x_vals_emu_catwise[np.where(np.isnan(x_vals_emu_catwise[:, -2])), -2] = np.nanmax(
        x_vals_emu_catwise[:, -2]
    )
    x_vals_emu_catwise[np.where(np.isnan(x_vals_emu_catwise[:, -1])), -1] = np.nanmax(
        x_vals_emu_catwise[:, -1]
    )

    x_vals_cat = combined_train_data_catwise[:, 1:]
    y_vals_cat = combined_train_data_catwise[:, 0]

    gmm_comp = 25

    # for gmm_comp in gmm_components_list:
    gmm_model_cat = GaussianMixture(n_components=gmm_comp, max_iter=500).fit(x_vals_cat)

    clusters_train_cat = gmm_model_cat.predict(x_vals_cat)
    clusters_test_cat = gmm_model_cat.predict(x_vals_emu_catwise)

    for component in np.arange(gmm_model_cat.n_components):
        cat_file_name = "pred_catwise_comp_" + str(component) + ".txt"
        if not os.path.isfile(cat_file_name):
            print("\n#############################\n Component: ", str(component))

            comp_y_train_cat = y_vals_cat[np.where(clusters_train_cat == component)]
            comp_x_train_cat = x_vals_cat[np.where(clusters_train_cat == component)]

            comp_x_cat = x_vals_emu_catwise[np.where(clusters_test_cat == component)]

            train_data_cat = np.hstack(
                (np.expand_dims(comp_y_train_cat, axis=1), comp_x_train_cat)
            )

            test_data_cat = np.array(comp_x_cat)

            col_names_cat = [
                "z_spec",
                "mag_g",
                "mag_r",
                "mag_i",
                "mag_z",
                "magerr_g",
                "magerr_r",
                "magerr_i",
                "magerr_z",
                "mag_w1",
                "mag_w2",
                "magerr_w1",
                "magerr_w2",
            ]
            col_names_pred_cat = [
                "mag_g",
                "mag_r",
                "mag_i",
                "mag_z",
                "magerr_g",
                "magerr_r",
                "magerr_i",
                "magerr_z",
                "mag_w1",
                "mag_w2",
                "magerr_w1",
                "magerr_w2",
            ]

            train_table_cat = Table(train_data_cat, names=col_names_cat)
            test_table_cat = Table(test_data_cat, names=col_names_pred_cat)

            train_file_cat = "train_catwise_comp_" + str(component) + ".cat"
            test_file_cat = "test_catwise_comp_" + str(component) + ".cat"

            temp_table_catwise = full_table_catwise[clusters_test_cat == component]

            test_table_cat["EMU_island_id"] = temp_table_catwise["island_id"]
            test_table_cat["EMU_component_id"] = temp_table_catwise["component_id"]
            test_table_cat["EMU_component_name"] = temp_table_catwise["component_name"]

            train_table_cat.write(
                train_file_cat, format="ascii.commented_header", overwrite=True
            )
            test_table_cat.write(
                test_file_cat, format="ascii.commented_header", overwrite=True
            )

            gpz_opts["TRAINING_CATALOG"] = (
                "train_catwise_comp_" + str(component) + ".cat"
            )
            gpz_opts["PREDICTION_CATALOG"] = (
                "test_catwise_comp_" + str(component) + ".cat"
            )
            gpz_opts["OUTPUT_CATALOG"] = "pred_catwise_comp_" + str(component) + ".txt"
            gpz_opts["MODEL_FILE"] = "model_catwise_comp_" + str(component) + ".cat"

            gpz_file = "gpz_params_catwise_comp_" + str(component) + ".param"

            out_file = open(gpz_file, "w")
            for param in gpz_opts:
                out_file.write("{0:30s} = {1}\n".format(param, gpz_opts[param]))
            out_file.close()

            cmd = ["gpz++", gpz_file]
            subprocess.Popen(cmd).wait()
            print("\n#############################\n")

    pred_cat_file_name = "pred_catwise_comp_0.txt"
    train_cat_file_name = "train_catwise_comp_0.cat"
    test_cat_file_name = "test_catwise_comp_0.cat"
    pred_cat_table_stack = Table.read(
        pred_cat_file_name, format="ascii.commented_header", header_start=10
    )
    train_cat_table_stack = Table.read(
        train_cat_file_name, format="ascii.commented_header"
    )
    test_cat_table_stack = Table.read(
        test_cat_file_name, format="ascii.commented_header"
    )

    test_cat_table_stack["prediction"] = pred_cat_table_stack["value"]
    test_cat_table_stack["prediction_uncert"] = pred_cat_table_stack["uncertainty"]
    test_cat_table_stack["comp_no"] = 0
    train_cat_table_stack["comp_no"] = 0

    for component in np.arange(1, gmm_model_cat.n_components):
        pred_cat_file_name = "pred_catwise_comp_" + str(component) + ".txt"
        test_cat_file_name = "test_catwise_comp_" + str(component) + ".cat"
        train_cat_file_name = "train_catwise_comp_" + str(component) + ".cat"
        test_cat_table = Table.read(test_cat_file_name, format="ascii.commented_header")
        train_cat_table = Table.read(
            train_cat_file_name, format="ascii.commented_header"
        )
        if len(test_cat_table) > 0:
            try:
                pred_cat_table = Table.read(
                    pred_cat_file_name, format="ascii.commented_header", header_start=10
                )
                test_cat_table["prediction"] = pred_cat_table["value"]
                test_cat_table["prediction_uncert"] = pred_cat_table["uncertainty"]
                test_cat_table["comp_no"] = component
                train_cat_table["comp_no"] = component
                test_cat_table_stack = vstack([test_cat_table_stack, test_cat_table])
            except:
                print(f"No EMU CatWISE sources in Component {component}")
        else:
            print(f"No EMU CatWISE sources in Component {component}")

        train_cat_table_stack = vstack([train_cat_table_stack, train_cat_table])

    test_cat_table_stack.write(
        "predictions_catwise_comp_" + str(gmm_comp) + ".csv",
        format="csv",
        overwrite=True,
    )
    train_cat_table_stack.write(
        "train_catwise_comp_" + str(gmm_comp) + ".csv",
        format="csv",
        overwrite=True,
    )


if __name__ == "__main__":
    main()
