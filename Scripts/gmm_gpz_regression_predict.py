from datetime import datetime

from matplotlib.pyplot import legend
from redshift_utils import *
import subprocess,argparse, os
import numpy as np
import pandas as pd
from astropy.table import Table, vstack
from sklearn.mixture import GaussianMixture
from collections import OrderedDict
import subprocess

def main():
    parser = argparse.ArgumentParser(description="This script runs the GPz algorithm, after reading, splitting and creating datasets")
    parser.add_argument("-s", "--seed", nargs=1, required=True, type=int, help="Index of random seed to use") 
    args = vars(parser.parse_args())
    seed_arg = args["seed"][0]
    seed_list = [
            28038,243145,205639,137877,136035,269736,27000,219084,63293,29586,165398,
            306500,231143,239118,225401,246949,161234,40248,263814,141492,157190,116489,
            57349,291151,245536,202276,126422,258477,171351,139302,141515,71389,28945,
            174227,278938,20048,269639,260007,86946,198443,51908,238160,220075,111377,
            21337,304953,140016,280582,212974,244536,238729,61147,114324,146624,156385,
            13761,171709,48471,233538,214585,289820,233973,115184,303951,129072,102360,
            284482,116383,23983,147515,249970,59524,145436,40816,215663,149446,103734,
            71285,177342,210428,295405,137335,50481,261593,197846,219994,30544,98132,
            241225,261461,136727,252823,264425,121729,282142,90580,75259,214412,200044,43904
            ]
    seed = seed_list[seed_arg]
    test_split_rate = 0.3 # 30% of total is taken for test set

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


    atlas_file = "../Data/ATLAS_Corrected.fits"
    stripe_file = "../Data/Stripe_Corrected.fits"
    sdss_file = "../Data/RGZ_Corrected.fits"
    pred_file = "../Data/EMU_PS_Clean.fits"
    pred_file_allwise = "../Data/EMU_PS_Clean_AllWISE_2.fits"

    sdss_cols_allwise = ["z", "g_corrected", "r_corrected", "i_corrected", "z_corrected",
                        "g_corrected_err", "r_corrected_err", "i_corrected_err", "z_corrected_err",
                        "W1Mag", "W2Mag", "W3Mag", "W4Mag",
                        "e_W1Mag", "e_W2Mag", "e_W3Mag", "e_W4Mag"]
    sdss_cols_catwise = ["z", "g_corrected", "r_corrected", "i_corrected", "z_corrected",
                        "g_corrected_err", "r_corrected_err", "i_corrected_err", "z_corrected_err",
                        "W1Mag", "W2Mag",
                        "e_W1Mag", "e_W2Mag"]
    des_cols_catwise = ["z", "mag_auto_g", "mag_auto_r", "mag_auto_i", "mag_auto_z",
                        "magerr_auto_g", "magerr_auto_r", "magerr_auto_i", "magerr_auto_z",
                        "W1Mag", "W2Mag",
                        "e_W1Mag", "e_W2Mag"]
    des_cols_allwise = ["z", "mag_auto_g", "mag_auto_r", "mag_auto_i", "mag_auto_z",
                        "magerr_auto_g", "magerr_auto_r", "magerr_auto_i", "magerr_auto_z",
                        "W1Mag", "W2Mag", "W3Mag", "W4Mag",
                        "e_W1Mag", "e_W2Mag", "e_W3Mag", "e_W4Mag"]

    pred_cols = ["mag_auto_g", "mag_auto_r", "mag_auto_i", "mag_auto_z",
                        "magerr_auto_g", "magerr_auto_r", "magerr_auto_i", "magerr_auto_z",
                        "w1mag", "w2mag",
                        "w2sigm", "w2sigm"]
    pred_allwise_cols = ["mag_auto_g", "mag_auto_r", "mag_auto_i", "mag_auto_z",
                        "magerr_auto_g", "magerr_auto_r", "magerr_auto_i", "magerr_auto_z",
                        "W1mag_x", "W2mag_x", "W3mag","W4mag",
                        "e_W1mag_x", "e_W2mag_x", "e_W3mag","e_W4mag"]

    full_table_catwise = Table.read(pred_file).to_pandas()
    full_table_allwise = Table.read(pred_file_allwise).to_pandas()

    atlas_data_catwise = read_fits(atlas_file, des_cols_catwise)
    stripe_data_catwise = read_fits(stripe_file, des_cols_catwise)
    sdss_data_catwise = read_fits(sdss_file, sdss_cols_catwise)

    atlas_data_allwise = read_fits(atlas_file, des_cols_allwise)
    stripe_data_allwise = read_fits(stripe_file, des_cols_allwise)
    sdss_data_allwise = read_fits(sdss_file, sdss_cols_allwise)

    combined_train_data_catwise = np.vstack((atlas_data_catwise, stripe_data_catwise, sdss_data_catwise))
    combined_train_data_allwise = np.vstack((atlas_data_allwise, stripe_data_allwise, sdss_data_allwise))

    x_vals_emu_catwise = read_fits(pred_file, pred_cols)
    x_vals_emu_allwise = read_fits(pred_file_allwise, pred_allwise_cols)

    x_vals_catwise = combined_train_data_catwise[:, 1:]
    y_vals_catwise = combined_train_data_catwise[:,0]

    x_vals_allwise = combined_train_data_allwise[:, 1:]
    y_vals_allwise = combined_train_data_allwise[:,0]


    combined_train_data_allwise[np.where(combined_train_data_allwise[:,-4] == -99), -4] = np.nanmax(combined_train_data_allwise[:,-4])
    combined_train_data_allwise[np.where(combined_train_data_allwise[:,-3] == -99), -3] = np.nanmax(combined_train_data_allwise[:,-3])
    combined_train_data_allwise[np.where(combined_train_data_allwise[:,-2] == -99), -2] = np.nanmax(combined_train_data_allwise[:,-2])
    combined_train_data_allwise[np.where(combined_train_data_allwise[:,-1] == -99), -1] = np.max(combined_train_data_allwise[:,-1])

    combined_train_data_allwise[np.where(np.isnan(combined_train_data_allwise[:,-4])), -4] = np.nanmax(combined_train_data_allwise[:,-4])
    combined_train_data_allwise[np.where(np.isnan(combined_train_data_allwise[:,-3])), -3] = np.nanmax(combined_train_data_allwise[:,-3])
    combined_train_data_allwise[np.where(np.isnan(combined_train_data_allwise[:,-2])), -2] = np.nanmax(combined_train_data_allwise[:,-2])
    combined_train_data_allwise[np.where(np.isnan(combined_train_data_allwise[:,-1])), -1] = np.nanmax(combined_train_data_allwise[:,-1])

    x_vals_all = combined_train_data_allwise[:, 1:]
    y_vals_all = combined_train_data_allwise[:,0]
    x_vals_cat = combined_train_data_catwise[:, 1:]
    y_vals_cat = combined_train_data_catwise[:,0]


    gmm_comp = 30



    # for gmm_comp in gmm_components_list:
    gmm_model_all = GaussianMixture(n_components=gmm_comp,
                        max_iter=500).fit(x_vals_all)
    gmm_model_cat = GaussianMixture(n_components=gmm_comp,
                        max_iter=500).fit(x_vals_cat)

    clusters_train_cat = gmm_model_cat.predict(x_vals_cat)
    clusters_test_cat = gmm_model_cat.predict(x_vals_emu_catwise)
    clusters_train_all = gmm_model_all.predict(x_vals_all)
    clusters_test_all = gmm_model_all.predict(x_vals_emu_allwise)
    

    for component in np.arange(gmm_model_cat.n_components):
        all_file_name = "pred_allwise_comp_" + str(component) + ".txt"
        cat_file_name = "pred_catwise_comp_" + str(component) + ".txt"
        if not os.path.isfile(all_file_name):
            print("\n#############################\n Component: ", str(component))
            
            comp_y_train_all = y_vals_all[np.where(clusters_train_all == component)]
            comp_x_train_all = x_vals_all[np.where(clusters_train_all == component)]
            comp_y_train_cat = y_vals_cat[np.where(clusters_train_cat == component)]
            comp_x_train_cat = x_vals_cat[np.where(clusters_train_cat == component)]

            # comp_y_test = y_vals_test[np.where(clusters_test == component)]
            comp_x_all = x_vals_emu_allwise[np.where(clusters_test_all == component)]
            # comp_y_test = y_vals_test[np.where(clusters_test == component)]
            comp_x_cat = x_vals_emu_catwise[np.where(clusters_test_cat == component)]

            train_data_all = np.hstack((np.expand_dims(comp_y_train_all, axis=1), comp_x_train_all))
            train_data_cat = np.hstack((np.expand_dims(comp_y_train_cat, axis=1), comp_x_train_cat))

            # test_data_all = np.hstack((np.expand_dims(comp_y_test, axis=1), comp_x_all))
            # test_data_cat = np.hstack((np.expand_dims(comp_y_test, axis=1), comp_x_cat))

            test_data_all = np.array(comp_x_all)
            test_data_cat = np.array(comp_x_cat)


            col_names = ["z_spec", "mag_g", "mag_r", "mag_i", "mag_z",
                "magerr_g", "magerr_r", "magerr_i", "magerr_z",
                "mag_w1", "mag_w2", "mag_w3", "mag_w4",
                "magerr_w1", "magerr_w2", "magerr_w3", "magerr_w4"]
            col_names_pred = ["mag_g", "mag_r", "mag_i", "mag_z",
                "magerr_g", "magerr_r", "magerr_i", "magerr_z",
                "mag_w1", "mag_w2", "mag_w3", "mag_w4",
                "magerr_w1", "magerr_w2", "magerr_w3", "magerr_w4"]
            
            train_table_all = Table(train_data_all, names=col_names)
            test_table_all = Table(test_data_all, names=col_names_pred)
            train_table_cat = Table(train_data_cat, names=col_names)
            test_table_cat = Table(test_data_cat, names=col_names_pred)
            
            train_file_all = "train_allwise_comp_" + str(component) + ".cat"
            test_file_all = "test_allwise_comp_" + str(component) + ".cat"
            train_file_cat = "train_catwise_test_comp_" + str(component) + ".cat"
            test_file_cat = "test_catwise_test_comp_" + str(component) + ".cat"
            
            train_table_all.write(train_file_all, format='ascii.commented_header', overwrite=True)
            test_table_all.write(test_file_all, format='ascii.commented_header', overwrite=True)
            train_table_cat.write(train_file_cat, format='ascii.commented_header', overwrite=True)
            test_table_cat.write(test_file_cat, format='ascii.commented_header', overwrite=True)

            gpz_opts["TRAINING_CATALOG"] = "train_allwise_comp_" + str(component) + ".cat"
            gpz_opts["PREDICTION_CATALOG"] = "test_allwise_comp_" + str(component) + ".cat"
            gpz_opts["OUTPUT_CATALOG"] = "pred_allwise_comp_" + str(component) + ".txt"
            gpz_opts["MODEL_FILE"] = "model_allwise_comp_" + str(component) + ".cat"

            gpz_file = "gpz_params_allwise_comp_" + str(component) + ".param"

            out_file = open(gpz_file, "w")
            for param in gpz_opts:
                out_file.write('{0:30s} = {1}\n'.format(param, gpz_opts[param]))
            out_file.close()

            cmd = ['gpz++', gpz_file]
            subprocess.Popen(cmd).wait()

            gpz_opts["TRAINING_CATALOG"] = "train_catwise_comp_" + str(component) + ".cat"
            gpz_opts["PREDICTION_CATALOG"] = "test_catwise_comp_" + str(component) + ".cat"
            gpz_opts["OUTPUT_CATALOG"] = "pred_catwise_comp_" + str(component) + ".txt"
            gpz_opts["MODEL_FILE"] = "model_catwise_comp_" + str(component) + ".cat"

            gpz_file = "gpz_params_catwise_comp_" + str(component) + ".param"

            out_file = open(gpz_file, "w")
            for param in gpz_opts:
                out_file.write('{0:30s} = {1}\n'.format(param, gpz_opts[param]))
            out_file.close()


            cmd = ['gpz++', gpz_file]
            subprocess.Popen(cmd).wait()
            print("\n#############################\n")
            
        
        
    pred_all_file_name = "pred_all_comp_0.txt"
    train_all_file_name = "train_all_comp_0.cat"
    test_all_file_name = "test_all_comp_0.cat"
    pred_all_table_stack = Table.read(pred_all_file_name, format="ascii.commented_header", header_start=10)
    test_all_table_stack = Table.read(train_all_file_name, format="ascii.commented_header")
    train_all_table_stack = Table.read(test_all_file_name, format="ascii.commented_header")
        
    pred_cat_file_name = "pred_cat_comp_0.txt"
    train_cat_file_name = "train_cat_comp_0.cat"
    test_cat_file_name = "test_cat_comp_0.cat"
    pred_cat_table_stack = Table.read(pred_cat_file_name, format="ascii.commented_header", header_start=10)
    test_cat_table_stack = Table.read(train_cat_file_name, format="ascii.commented_header")
    train_cat_table_stack = Table.read(test_cat_file_name, format="ascii.commented_header")

    test_all_table_stack["prediction"] = pred_all_table_stack["value"]
    test_all_table_stack["prediction_uncert"] = pred_all_table_stack["uncertainty"]
    test_all_table_stack["comp_no"] = 0
    train_all_table_stack["comp_no"] = 0

    test_cat_table_stack["prediction"] = pred_cat_table_stack["value"]
    test_cat_table_stack["prediction_uncert"] = pred_cat_table_stack["uncertainty"]
    test_cat_table_stack["comp_no"] = 0
    train_cat_table_stack["comp_no"] = 0


    for component in np.arange(1,gmm_model_cat.n_components):
        pred_all_file_name = "pred_all_comp_" + str(component) + ".txt"
        test_all_file_name = "test_all_comp_" + str(component) + ".cat"
        train_all_file_name = "train_all_comp_" + str(component) + ".cat"
        pred_all_table = Table.read(pred_all_file_name, format="ascii.commented_header", header_start=10)
        test_all_table = Table.read(test_all_file_name, format="ascii.commented_header")
        train_all_table = Table.read(train_all_file_name, format="ascii.commented_header")

        pred_cat_file_name = "pred_cat_comp_" + str(component) + ".txt"
        test_cat_file_name = "test_cat_comp_" + str(component) + ".cat"
        train_cat_file_name = "train_cat_comp_" + str(component) + ".cat"
        pred_cat_table = Table.read(pred_cat_file_name, format="ascii.commented_header", header_start=10)
        test_cat_table = Table.read(test_cat_file_name, format="ascii.commented_header")
        train_cat_table = Table.read(train_cat_file_name, format="ascii.commented_header")

        test_all_table["prediction"] = pred_all_table["value"]
        test_all_table["prediction_uncert"] = pred_all_table["uncertainty"]
        test_all_table["comp_no"] = component
        train_all_table["comp_no"] = component

        test_cat_table["prediction"] = pred_cat_table["value"]
        test_cat_table["prediction_uncert"] = pred_cat_table["uncertainty"]
        test_cat_table["comp_no"] = component
        train_cat_table["comp_no"] = component

        test_all_table_stack = vstack([test_all_table_stack, test_all_table])
        train_all_table_stack = vstack([train_all_table_stack, train_all_table])

        test_cat_table_stack = vstack([test_cat_table_stack, test_cat_table])
        train_cat_table_stack = vstack([train_cat_table_stack, train_cat_table])


    test_all_table_stack.write("predictions_all_seed_"+str(seed)+"_comp_" + str(gmm_comp) + ".csv", format="csv", overwrite=True)
    train_all_table_stack.write("train_all_seed_"+str(seed)+"_comp_" + str(gmm_comp) + ".csv", format="csv", overwrite=True)

    test_cat_table_stack.write("predictions_cat_seed_"+str(seed)+"_comp_" + str(gmm_comp) + ".csv", format="csv", overwrite=True)
    train_cat_table_stack.write("train_cat_seed_"+str(seed)+"_comp_" + str(gmm_comp) + ".csv", format="csv", overwrite=True)





if __name__ == "__main__":
	main()
