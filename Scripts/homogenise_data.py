#!/usr/bin/env python

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy.polynomial.polynomial import Polynomial
from astropy.table import Table

# User-Serviceable. Where are the files to be converted, and what are the columns required?
plt.rcParams.update({'font.size': 28})


atlas_file = "../Data/ATLAS_AllWISE_SWIRE_OzDES_DES.fits"
stripe_file = "../Data/Stripe_AllWISE_DES_SDSS.fits"
sdss_file = "../Data/RGZ_NVSS_AllWISE_SDSS_NotStripe.fits"

sdss_cols = ["modelMag_g", "modelMag_r", "modelMag_i", "modelMag_z",
    "modelMagErr_g", "modelMagErr_r", "modelMagErr_i", "modelMagErr_z"]
des_cols = ["mag_auto_g", "mag_auto_r", "mag_auto_i", "mag_auto_z",
    "magerr_auto_g", "magerr_auto_r", "magerr_auto_i", "magerr_auto_z"]

# Nothing user-serviceable after this. 

def read_fits(filename, colnames):
    """Read an astronomical table, outputting specific columns. 

    Args:
        filename (String): Name of file to be read
        colnames ([String]): Array of columns to be read. 

    Returns:
        table_data (Astropy Table): Astropy table of data
    """
    # table_data = Table.read(filename)

    hdul = fits.open(filename)
    hdulData = hdul[1].data
    #Create catalogueData array from the redshift column
    table_data = np.reshape(np.array(hdulData.field(colnames[0]), dtype=np.float32), [len(hdulData.field(colnames[0])),1])
    #Add the columns required for the test
    for i in range(1, len(colnames)):
        table_data = np.hstack([table_data,np.reshape(np.array(hdulData.field(colnames[i]), dtype=np.float32), [len(hdulData.field(colnames[i])),1])])
    return table_data


def split_data(x_vals, y_vals, rand_generator, split_rate=0.3):
    """Splits large data set into training and test sets.

    Args:
        x_vals (Numpy Array (2-D)): 2-D Array holding the x-vals
        y_vals (Numpy Array (1-D)): 1-D Array holding the y-vals
        rand_generator (Numpy Random Generator): Random Generator to use to generate Training-Testing Split. 
        split_rate (Decimal; optional): Percentage of data to use as the test set. Default: 0.3 (70% training, 30% test)

    Returns:
        Numpy Arrays: The split Training and Test sets. 
    """
    test_indices = rand_generator.choice(len(x_vals), round(len(x_vals)*split_rate), replace=False)
    train_indices = np.array(list(set(range(len(x_vals))) - set(test_indices)))
    x_vals_train = x_vals[train_indices]
    x_vals_test = x_vals[test_indices]
    y_vals_train = y_vals[train_indices]
    y_vals_test = y_vals[test_indices]

    return x_vals_train, x_vals_test, y_vals_train, y_vals_test


# Read in files
stripe_data_sdss = read_fits(stripe_file, sdss_cols)
stripe_data_des = read_fits(stripe_file, des_cols)

# Split Stripe data into a training and test set, so we can evaluate the difference
stripe_des_train, stripe_des_test, stripe_sdss_train, stripe_sdss_test = \
    split_data(stripe_data_des, stripe_data_sdss, np.random.default_rng(42))

# Calculate the g-z colour to use
des_gz_col = stripe_data_sdss[:,0] - stripe_data_sdss[:,3]

# And the difference between the DES (Auto) and SDSS (Model) magnitudes
g_diff = stripe_data_sdss[:,0] - stripe_data_des[:,0]
r_diff = stripe_data_sdss[:,1] - stripe_data_des[:,1]
i_diff = stripe_data_sdss[:,2] - stripe_data_des[:,2]
z_diff = stripe_data_sdss[:,3] - stripe_data_des[:,3]

print("g diff: ", str(np.mean(g_diff)))
print("r diff: ", str(np.mean(r_diff)))
print("i diff: ", str(np.mean(i_diff)))
print("z diff: ", str(np.mean(z_diff)))

# Fit a 3rd order polynomial to each
g_poly = Polynomial.fit(des_gz_col, g_diff, deg=3)
r_poly = Polynomial.fit(des_gz_col, r_diff, deg=3)
i_poly = Polynomial.fit(des_gz_col, i_diff, deg=3)
z_poly = Polynomial.fit(des_gz_col, z_diff, deg=3)

# Correct the DES to match the SDSS data
des_g_corrected = stripe_data_sdss[:,0] - g_poly(stripe_data_sdss[:,0] - stripe_data_sdss[:,3])
diff = des_g_corrected-stripe_data_des[:,0]
print("Corrected g: ", str(np.mean(diff)))

des_r_corrected = stripe_data_sdss[:,1] - r_poly(stripe_data_sdss[:,0] - stripe_data_sdss[:,3])
diff = des_r_corrected-stripe_data_des[:,1]
print("Corrected r: ", str(np.mean(diff)))

des_i_corrected = stripe_data_sdss[:,2] - i_poly(stripe_data_sdss[:,0] - stripe_data_sdss[:,3])
diff = des_i_corrected-stripe_data_des[:,2]
print("Corrected i: ", str(np.mean(diff)))

des_z_corrected = stripe_data_sdss[:,3] - z_poly(stripe_data_sdss[:,0] - stripe_data_sdss[:,3])
diff = des_z_corrected-stripe_data_des[:,3]
print("Corrected z: ", str(np.mean(diff)))

# Make plot of change:
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,sharex=True, sharey=True, figsize=(15,15))

# g-z using corrected DES values
des_gz_col_corrected = des_g_corrected - des_z_corrected

g_diff = des_g_corrected - stripe_data_des[:,0]
r_diff = des_r_corrected - stripe_data_des[:,1]
i_diff = des_i_corrected - stripe_data_des[:,2]
z_diff = des_z_corrected - stripe_data_des[:,3]

g_z_col_orig = stripe_data_sdss[:,0] - stripe_data_sdss[:,3]

g_diff_orig = stripe_data_sdss[:,0] - stripe_data_des[:,0]
r_diff_orig = stripe_data_sdss[:,1] - stripe_data_des[:,1]
i_diff_orig = stripe_data_sdss[:,2] - stripe_data_des[:,2]
z_diff_orig = stripe_data_sdss[:,3] - stripe_data_des[:,3]

ax1_poly_corrected = Polynomial.fit(des_gz_col_corrected, g_diff, deg=3)
ax2_poly_corrected = Polynomial.fit(des_gz_col_corrected, r_diff, deg=3)
ax3_poly_corrected = Polynomial.fit(des_gz_col_corrected, i_diff, deg=3)
ax4_poly_corrected = Polynomial.fit(des_gz_col_corrected, z_diff, deg=3)

ax1_poly = Polynomial.fit(g_z_col_orig, g_diff_orig, deg=3)
ax2_poly = Polynomial.fit(g_z_col_orig, r_diff_orig, deg=3)
ax3_poly = Polynomial.fit(g_z_col_orig, i_diff_orig, deg=3)
ax4_poly = Polynomial.fit(g_z_col_orig, z_diff_orig, deg=3)






des_gz_err = stripe_data_des[:,4] - stripe_data_des[:,7]

# And the difference between the DES (Auto) and SDSS (Model) magnitudes
g_diff_err = stripe_data_sdss[:,4] - stripe_data_des[:,4]
r_diff_err = stripe_data_sdss[:,5] - stripe_data_des[:,5]
i_diff_err = stripe_data_sdss[:,6] - stripe_data_des[:,6]
z_diff_err = stripe_data_sdss[:,7] - stripe_data_des[:,7]

g_poly_error = Polynomial.fit(des_gz_err, g_diff_err, deg=3)
r_poly_error = Polynomial.fit(des_gz_err, r_diff_err, deg=3)
i_poly_error = Polynomial.fit(des_gz_err, i_diff_err, deg=3)
z_poly_error = Polynomial.fit(des_gz_err, z_diff_err, deg=3)


ax1.set_title(r"$g$")
ax1.scatter(des_gz_col_corrected, g_diff, s=1, alpha=0.4, color="blue", label="Corrected")
ax1.scatter(g_z_col_orig, g_diff_orig, s=1, alpha=0.4, color="darkorange", label="Original")
ax1.axhline(np.mean(g_diff), color="black")
ax1.axhline(np.mean(g_diff) - np.std(g_diff), color="black", ls=":")
ax1.axhline(np.mean(g_diff) + np.std(g_diff), color="black", ls=":")
ax1.grid()
ax1.set_ylabel(r"$g_{SDSS}-g_{DES}$")
ax1.plot(ax1_poly.linspace()[0], ax1_poly.linspace()[1], color="darkorange", lw=3)
ax1.plot(ax1_poly_corrected.linspace()[0], ax1_poly_corrected.linspace()[1], color="blue", lw=3)
ax1.set_xlabel(r"$g_{des}-z_{des}$")
# ax1.legend()

ax2.set_title(r"$r$")
ax2.scatter(des_gz_col_corrected, r_diff, s=1, alpha=0.4, color="blue", label="Corrected")
ax2.scatter(g_z_col_orig, r_diff_orig, s=1, alpha=0.4, color="darkorange", label="Original")
ax2.axhline(np.mean(r_diff), color="black")
ax2.axhline(np.mean(r_diff) - np.std(r_diff), color="black", ls=":")
ax2.axhline(np.mean(r_diff) + np.std(r_diff), color="black", ls=":")
ax2.grid()
ax2.set_ylabel(r"$r_{SDSS}-r_{DES}$")
ax2.plot(ax2_poly.linspace()[0], ax2_poly.linspace()[1], color="darkorange", lw=3)
ax2.plot(ax2_poly_corrected.linspace()[0], ax2_poly_corrected.linspace()[1], color="blue", lw=3)
ax2.set_xlabel(r"$g_{des}-z_{des}$")
# ax2.legend()


ax3.set_title(r"$i$")
ax3.scatter(des_gz_col_corrected, i_diff, s=1, alpha=0.4, color="blue", label="Corrected")
ax3.scatter(g_z_col_orig, i_diff_orig, s=1, alpha=0.4, color="darkorange", label="Original")
ax3.axhline(np.mean(i_diff), color="black")
ax3.axhline(np.mean(i_diff) - np.std(i_diff), color="black", ls=":")
ax3.axhline(np.mean(i_diff) + np.std(i_diff), color="black", ls=":")
ax3.grid()
ax3.set_ylabel(r"$i_{SDSS}-i_{DES}$")
ax3.plot(ax3_poly.linspace()[0], ax3_poly.linspace()[1], color="darkorange", lw=3)
ax3.plot(ax3_poly_corrected.linspace()[0], ax3_poly_corrected.linspace()[1], color="blue", lw=3)
ax3.set_xlabel(r"$g_{des}-z_{des}$")
# ax3.legend()


ax4.set_title(r"$z$")
ax4.scatter(des_gz_col_corrected, z_diff, s=1, alpha=0.4, color="blue", label="Corrected")
ax4.scatter(g_z_col_orig, z_diff_orig, s=1, alpha=0.4, color="darkorange", label="Original")
ax4.axhline(np.mean(z_diff), color="black")
ax4.axhline(np.mean(z_diff) - np.std(z_diff), color="black", ls=":")
ax4.axhline(np.mean(z_diff) + np.std(z_diff), color="black", ls=":")
ax4.grid()
ax4.set_ylabel(r"$z_{SDSS}-z_{DES}$")
ax4.plot(ax4_poly.linspace()[0], ax4_poly.linspace()[1], color="darkorange", lw=3)
ax4.plot(ax4_poly_corrected.linspace()[0], ax4_poly_corrected.linspace()[1], color="blue", lw=3)
ax4.set_xlabel(r"$g_{des}-z_{des}$")

legend_elements = [Line2D([0],[0], marker="o", markersize=5, label="Original", color="darkorange"),
Line2D([0],[0], marker="o", markersize=5, label="Corrected", color="blue")]

ax1.set_ylim(3,-3)


fig.tight_layout()
ax2.legend(handles=legend_elements, loc="upper right")
# plt.show()
plt.savefig("sdss_des_homogenise.pdf")



# Correction is possible, and is done. Now to correct the data

rgz = Table.read(sdss_file)

rgz["g_corrected"] = rgz["modelMag_g"] - g_poly(rgz["modelMag_g"] - rgz["modelMag_z"])
rgz["r_corrected"] = rgz["modelMag_r"] - r_poly(rgz["modelMag_g"] - rgz["modelMag_z"])
rgz["i_corrected"] = rgz["modelMag_i"] - i_poly(rgz["modelMag_g"] - rgz["modelMag_z"])
rgz["z_corrected"] = rgz["modelMag_z"] - z_poly(rgz["modelMag_g"] - rgz["modelMag_z"])


# Calculate errors
des_gz_err_min = (stripe_data_sdss[:,0] - stripe_data_sdss[:,4]) - \
        (stripe_data_sdss[:,3] - stripe_data_sdss[:,7])
des_gz_err_max = (stripe_data_sdss[:,0] + stripe_data_sdss[:,4]) - \
        (stripe_data_sdss[:,3] + stripe_data_sdss[:,7])

# And the difference between the DES (Auto) and SDSS (Model) magnitudes
g_diff_err_min = (stripe_data_sdss[:,0] - stripe_data_sdss[:,4]) - \
    (stripe_data_des[:,0] - stripe_data_des[:,4])
r_diff_err_min = (stripe_data_sdss[:,1] - stripe_data_sdss[:,5]) - \
    (stripe_data_des[:,1] - stripe_data_des[:,5])
i_diff_err_min = (stripe_data_sdss[:,2] - stripe_data_sdss[:,6]) - \
    (stripe_data_des[:,2] - stripe_data_des[:,6])
z_diff_err_min = (stripe_data_sdss[:,3] - stripe_data_sdss[:,7]) - \
    (stripe_data_des[:,3] - stripe_data_des[:,7])

g_poly_error_min = Polynomial.fit(des_gz_err_min, g_diff_err_min, deg=3)
r_poly_error_min = Polynomial.fit(des_gz_err_min, r_diff_err_min, deg=3)
i_poly_error_min = Polynomial.fit(des_gz_err_min, i_diff_err_min, deg=3)
z_poly_error_min = Polynomial.fit(des_gz_err_min, z_diff_err_min, deg=3)

g_diff_err_max = (stripe_data_sdss[:,0] + stripe_data_sdss[:,4]) - \
    (stripe_data_des[:,0] + stripe_data_des[:,4])
r_diff_err_max = (stripe_data_sdss[:,1] + stripe_data_sdss[:,5]) - \
    (stripe_data_des[:,1] + stripe_data_des[:,5])
i_diff_err_max = (stripe_data_sdss[:,2] + stripe_data_sdss[:,6]) - \
    (stripe_data_des[:,2] + stripe_data_des[:,6])
z_diff_err_max = (stripe_data_sdss[:,3] + stripe_data_sdss[:,7]) - \
    (stripe_data_des[:,3] + stripe_data_des[:,7])

g_poly_error_max = Polynomial.fit(des_gz_err_max, g_diff_err_max, deg=3)
r_poly_error_max = Polynomial.fit(des_gz_err_max, r_diff_err_max, deg=3)
i_poly_error_max = Polynomial.fit(des_gz_err_max, i_diff_err_max, deg=3)
z_poly_error_max = Polynomial.fit(des_gz_err_max, z_diff_err_max, deg=3)

atlas_gerr_min = (rgz["modelMag_g"] - rgz["modelMagErr_g"]) - g_poly_error_min( \
    (rgz["modelMag_g"] - rgz["modelMagErr_g"]) - \
    (rgz["modelMag_z"] - rgz["modelMagErr_z"])) - rgz["g_corrected"]
atlas_gerr_max = (rgz["modelMag_g"] + rgz["modelMagErr_g"]) - g_poly_error_max( \
    (rgz["modelMag_g"] + rgz["modelMagErr_g"]) - \
    (rgz["modelMag_z"] + rgz["modelMagErr_z"])) - rgz["g_corrected"]
atlas_gerr = np.max([np.abs(atlas_gerr_min), np.abs(atlas_gerr_max)], axis=0)

atlas_rerr_min = (rgz["modelMag_r"] - rgz["modelMagErr_r"]) - r_poly_error_min( \
    (rgz["modelMag_g"] - rgz["modelMagErr_g"]) - \
    (rgz["modelMag_z"] - rgz["modelMagErr_z"])) - rgz["r_corrected"]
atlas_rerr_max = (rgz["modelMag_r"] + rgz["modelMagErr_r"]) - r_poly_error_max( \
    (rgz["modelMag_g"] + rgz["modelMagErr_g"]) - \
    (rgz["modelMag_z"] + rgz["modelMagErr_z"])) - rgz["r_corrected"]
atlas_rerr = np.max([np.abs(atlas_rerr_min), np.abs(atlas_rerr_max)], axis=0)

atlas_ierr_min = (rgz["modelMag_i"] - rgz["modelMagErr_i"]) - i_poly_error_min( \
    (rgz["modelMag_g"] - rgz["modelMagErr_g"]) - \
    (rgz["modelMag_z"] - rgz["modelMagErr_z"])) - rgz["i_corrected"]
atlas_ierr_max = (rgz["modelMag_i"] + rgz["modelMagErr_i"]) - i_poly_error_max( \
    (rgz["modelMag_g"] + rgz["modelMagErr_g"]) - \
    (rgz["modelMag_z"] + rgz["modelMagErr_z"])) - rgz["i_corrected"]
atlas_ierr = np.max([np.abs(atlas_ierr_min), np.abs(atlas_ierr_max)], axis=0)

atlas_zerr_min = (rgz["modelMag_z"] - rgz["modelMagErr_z"]) - z_poly_error_min( \
    (rgz["modelMag_g"] - rgz["modelMagErr_g"]) - \
    (rgz["modelMag_z"] - rgz["modelMagErr_z"])) - rgz["z_corrected"]
atlas_zerr_max = (rgz["modelMag_z"] + rgz["modelMagErr_z"]) - z_poly_error_max( \
    (rgz["modelMag_g"] + rgz["modelMagErr_g"]) - \
    (rgz["modelMag_z"] + rgz["modelMagErr_z"])) - rgz["z_corrected"]
atlas_zerr = np.max([np.abs(atlas_zerr_min), np.abs(atlas_zerr_max)], axis=0)




# Load full files, so additional columns can be added with the corrected photometry


rgz["g_corrected"] = rgz["modelMag_g"] - g_poly(rgz["modelMag_g"] - rgz["modelMag_z"])
rgz["r_corrected"] = rgz["modelMag_r"] - r_poly(rgz["modelMag_g"] - rgz["modelMag_z"])
rgz["i_corrected"] = rgz["modelMag_i"] - i_poly(rgz["modelMag_g"] - rgz["modelMag_z"])
rgz["z_corrected"] = rgz["modelMag_z"] - z_poly(rgz["modelMag_g"] - rgz["modelMag_z"])




rgz["g_corrected_err"] = atlas_gerr
rgz["r_corrected_err"] = atlas_rerr
rgz["i_corrected_err"] = atlas_ierr
rgz["z_corrected_err"] = atlas_zerr
rgz.write("../Data/RGZ_Corrected.fits", overwrite=True)

# stripe_data = Table.read(stripe_file)
# stripe_data["g_corrected"] = stripe_data["mag_auto_g"] - g_poly(stripe_data["mag_auto_g"] - stripe_data["mag_auto_z"])
# stripe_data["r_corrected"] = stripe_data["mag_auto_r"] - r_poly(stripe_data["mag_auto_g"] - stripe_data["mag_auto_z"])
# stripe_data["i_corrected"] = stripe_data["mag_auto_i"] - i_poly(stripe_data["mag_auto_g"] - stripe_data["mag_auto_z"])
# stripe_data["z_corrected"] = stripe_data["mag_auto_z"] - z_poly(stripe_data["mag_auto_g"] - stripe_data["mag_auto_z"])



# stripe_gerr_min = (stripe_data["mag_auto_g"] - stripe_data["magerr_auto_g"]) - g_poly_error_min( \
#     (stripe_data["mag_auto_g"] - stripe_data["magerr_auto_g"]) - \
#     (stripe_data["mag_auto_z"] - stripe_data["magerr_auto_z"])) - stripe_data["g_corrected"]
# stripe_gerr_max = (stripe_data["mag_auto_g"] + stripe_data["magerr_auto_g"]) - g_poly_error_max( \
#     (stripe_data["mag_auto_g"] + stripe_data["magerr_auto_g"]) - \
#     (stripe_data["mag_auto_z"] + stripe_data["magerr_auto_z"])) - stripe_data["g_corrected"]
# stripe_gerr = np.max([np.abs(stripe_gerr_min), np.abs(stripe_gerr_max)], axis=0)

# stripe_rerr_min = (stripe_data["mag_auto_r"] - stripe_data["magerr_auto_r"]) - r_poly_error_min( \
#     (stripe_data["mag_auto_g"] - stripe_data["magerr_auto_g"]) - \
#     (stripe_data["mag_auto_z"] - stripe_data["magerr_auto_z"])) - stripe_data["r_corrected"]
# stripe_rerr_max = (stripe_data["mag_auto_r"] + stripe_data["magerr_auto_r"]) - r_poly_error_max( \
#     (stripe_data["mag_auto_g"] + stripe_data["magerr_auto_g"]) - \
#     (stripe_data["mag_auto_z"] + stripe_data["magerr_auto_z"])) - stripe_data["r_corrected"]
# stripe_rerr = np.max([np.abs(stripe_rerr_min), np.abs(stripe_rerr_max)], axis=0)

# stripe_ierr_min = (stripe_data["mag_auto_i"] - stripe_data["magerr_auto_i"]) - i_poly_error_min( \
#     (stripe_data["mag_auto_g"] - stripe_data["magerr_auto_g"]) - \
#     (stripe_data["mag_auto_z"] - stripe_data["magerr_auto_z"])) - stripe_data["i_corrected"]
# stripe_ierr_max = (stripe_data["mag_auto_i"] + stripe_data["magerr_auto_i"]) - i_poly_error_max( \
#     (stripe_data["mag_auto_g"] + stripe_data["magerr_auto_g"]) - \
#     (stripe_data["mag_auto_z"] + stripe_data["magerr_auto_z"])) - stripe_data["i_corrected"]
# stripe_ierr = np.max([np.abs(stripe_ierr_min), np.abs(stripe_ierr_max)], axis=0)

# stripe_zerr_min = (stripe_data["mag_auto_z"] - stripe_data["magerr_auto_z"]) - z_poly_error_min( \
#     (stripe_data["mag_auto_g"] - stripe_data["magerr_auto_g"]) - \
#     (stripe_data["mag_auto_z"] - stripe_data["magerr_auto_z"])) - stripe_data["z_corrected"]
# stripe_zerr_max = (stripe_data["mag_auto_z"] + stripe_data["magerr_auto_z"]) - z_poly_error_max( \
#     (stripe_data["mag_auto_g"] + stripe_data["magerr_auto_g"]) - \
#     (stripe_data["mag_auto_z"] + stripe_data["magerr_auto_z"])) - stripe_data["z_corrected"]
# stripe_zerr = np.max([np.abs(stripe_zerr_min), np.abs(stripe_zerr_max)], axis=0)

# stripe_data["g_corrected_err"] = stripe_gerr
# stripe_data["r_corrected_err"] = stripe_rerr
# stripe_data["i_corrected_err"] = stripe_ierr
# stripe_data["z_corrected_err"] = stripe_zerr


# stripe_data.write("Stripe_Corrected.fits", overwrite=True)

# print("G-Diff: ", np.mean(stripe_data["g_corrected"] - stripe_data["modelMag_g"]))
# print("R-Diff: ", np.mean(stripe_data["r_corrected"] - stripe_data["modelMag_r"]))
# print("I-Diff: ", np.mean(stripe_data["i_corrected"] - stripe_data["modelMag_i"]))
# print("Z-Diff: ", np.mean(stripe_data["z_corrected"] - stripe_data["modelMag_z"]))

# print("G-Diff_err: ", np.mean(stripe_data["g_corrected_err"] - stripe_data["modelMagErr_g"]))
# print("R-Diff_err: ", np.mean(stripe_data["r_corrected_err"] - stripe_data["modelMagErr_r"]))
# print("I-Diff_err: ", np.mean(stripe_data["i_corrected_err"] - stripe_data["modelMagErr_i"]))
# print("Z-Diff_err: ", np.mean(stripe_data["z_corrected_err"] - stripe_data["modelMagErr_z"]))
