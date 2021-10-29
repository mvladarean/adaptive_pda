import numpy as np
import itertools
import matplotlib.pyplot as plt
import math
import scipy.special
from sklearn.preprocessing import normalize
from sklearn import datasets
import time
import numpy as np
import torch
import PIL
from PIL import Image
import torchvision.transforms.functional as tf
from matplotlib import rc
import matplotlib
from skimage.metrics import structural_similarity as ssim

def get_best_run_index(loss_list):
    minval = math.inf
    best_index = -1
    for i in range(len(loss_list)):
        # get the last value
        tmp = loss_list[i][-1]
        if tmp <= minval:
            best_index = i
            minval = tmp

    return best_index



##################### Data reading functions #####################

def read_libsvm_data(filename):
    K, b = datasets.load_svmlight_file(filename)
    K = normalize(K) # normalize each sample by the l2 norm
    # K = K.toarray()
    m, dim = K.shape
    b = np.reshape(b, (m, 1))
    b[b == 0] = -1  # just in case it  has zeros
    b[b == 2] = -1  # just in case it  has twos
    assert (np.unique(b) == np.asarray([-1, 1])).all()
    assert np.count_nonzero(b) == len(b)
    print("The dataset {}. The dimensions: m={}, dim={}".format(filename[5:], m, dim))

    return m, dim, K, b

def load_image(path, size=(256, 256)):
    """
    load_image loads the image at the given path and resizes it to the given size
    with bicubic interpolation before normalizing it to 1.
    """
    I = Image.open(path)
    I = tf.resize(I, size, interpolation=PIL.Image.BICUBIC)
    I = tf.to_grayscale(I)
    I = tf.to_tensor(I).numpy()[0, :, :]
    return I


##################### Plotting functions #####################

def plot_stepsize_comparison(path, cv_tau, cv_sigma, apda_tau, apda_sigma, extra_title_info):
    plt.rcParams.update({
        'font.size': 33,
        "text.usetex": True,
        "font.family": "Times",
        "font.sans-serif": ["Helvetica"]})

    legend_line_width = 3
    plot_line_width_apda = 2
    plot_line_width_cv = 4
    alpha = 1


    ###################### Primal stepsize plot ######################
    plt.figure(figsize=(8, 6))
    plt.yscale('log')
    plt.plot(apda_tau, label="APDA", linewidth = plot_line_width_apda, color="blue", zorder=1, alpha=alpha)
    plt.hlines(cv_tau, 0, len(apda_tau)-1, color="red", label="CVA", linewidth=plot_line_width_cv, zorder=2)
    plt.ylabel(r'$\tau$ (primal)')
    plt.xlabel('k (iterations)')

    min_apda = np.floor(np.log10(np.min([np.min(apda_tau), cv_tau])))
    max_apda = np.ceil(np.log10(np.max([np.max(apda_tau), cv_tau])))
    plt.ylim((10**min_apda, 10**max_apda)) # to display y axis nicely with log markers
    ax = plt.gca()
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    leg_lines = plt.legend().get_lines()
    plt.setp(leg_lines[0], linewidth=legend_line_width)
    plt.setp(leg_lines[1], linewidth=legend_line_width)

    plt.tight_layout()
    plt.grid()
    plt.savefig(path + "tau_stepsize_" + extra_title_info + ".pdf", bbox_inches='tight')
    plt.show()

    ###################### Dual stepsize plot ######################
    plt.figure(figsize=(8, 6))
    plt.yscale('log')
    plt.plot(apda_sigma, label="APDA", linewidth = plot_line_width_apda, color="blue", zorder=1, alpha=alpha)
    plt.hlines(cv_sigma, 0, len(apda_sigma)-1, color="red", label="CVA", linewidth=plot_line_width_cv, zorder=2)
    plt.ylabel(r'$\sigma$ (dual)')
    plt.xlabel('k (iterations)')

    min_apda = np.floor(np.log10(np.min([np.min(apda_sigma), cv_sigma])))
    max_apda = np.ceil(np.log10(np.max([np.max(apda_sigma), cv_sigma])))
    plt.ylim((10 ** min_apda, 10 ** max_apda))# to display y axis nicely with log markers
    ax=plt.gca()
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    leg_lines = plt.legend().get_lines()
    plt.setp(leg_lines[0], linewidth=legend_line_width)
    plt.setp(leg_lines[1], linewidth=legend_line_width)

    #plt.tight_layout()
    plt.grid()
    plt.savefig(path + "sigma_stepsize_" + extra_title_info + ".pdf", bbox_inches='tight')
    plt.show()

def plot_run_comparison(path, loss_list, label_list, f_opt=None, extra_title_info=None):
    plt.rcParams.update({
        'font.size': 33,
        "text.usetex": True,
        "font.family": "Times",
        "font.sans-serif": ["Helvetica"]})

    marker = itertools.cycle((".","v","o",",","^","<",">","1","2","3","4","8","s",
                              "p","P","*","h","H","x","X","D","d","|","_"))

    colors = ["blue", "red", "#FAA402", "#118141"]
    legend_line_width = 3
    plot_line_width = 3
    marker_size = 15

    if f_opt is None:
        f_opt = np.min([np.min(loss) for loss in loss_list])
        f_opt -= 0.001

    plt.figure(figsize=(8, 6))
    plt.yscale('log')
    #plt.ylim((10 ** -1, 10 ** 10))
    for i in range(len(label_list)):
        marker_freq = max(1, len(loss_list[i]) // 20)

        if i <= 3:
            plt.plot(loss_list[i] - f_opt, linewidth = plot_line_width, label=label_list[i], color = colors[i],
                     marker=next(marker), markevery=marker_freq, markersize=marker_size)
        else:
            plt.plot(loss_list[i] - f_opt, linewidth=plot_line_width, label=label_list[i],
                     marker=next(marker), markevery=marker_freq, markersize=marker_size)

    plt.ylabel(r'$F(X) - F(X^*)$')
    plt.xlabel('k (iterations)')

    leg_lines = plt.legend().get_lines()
    for i in range(len(leg_lines)):
        plt.setp(leg_lines[i], linewidth=legend_line_width)

    #plt.tight_layout()
    plt.grid()
    plt.savefig(path + "results_" + extra_title_info+ ".pdf", bbox_inches='tight')
    plt.show()

def plot_image_comparison(original_image, intermediary_img, title_intermed_img, reconstruction, title_reconstruction):
    # Plot the reconstructed image alongside the original image and PSNR
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))  # modified size to solve saving issues
    ax[0].imshow(original_image, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(intermediary_img, cmap='gray')
    ax[1].set_title(title_intermed_img)
    ax[2].imshow(reconstruction, cmap="gray")
    ax[2].set_title(title_reconstruction)
    [axi.set_axis_off() for axi in ax.flatten()]
    plt.tight_layout()
    plt.show()

##################### Misc functions #####################

def save_array_as_grayscale_img(output_path, img_array):
    plt.figure(figsize=(8, 6))
    plt.imshow(img_array, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')

def get_prediction_accuracy(test_mat, test_labels, predictor):
    predictions = scipy.special.expit(test_mat @ predictor)
    predictions[predictions <= 0.5] = -1
    predictions[predictions > 0.5] = 1

    match = np.sum(predictions == test_labels)
    return match/len(test_labels)

def safe_division(x, y):
    #Computes safe division x/y for close to 0 values
    if x == 0. and y == 0.:
        #raise ValueError('Indeterminacy 0/0.')
        return 1e-16
    if x == 0. and y != 0.:
        return 0
    return np.exp(np.log(x) - np.log(y)) if y != 0. else 1e16

def TV_norm(X, is_iso=False):
    #X needs to me m x n

    m, n = X.shape
    # I think these should be reversed, but it's only a sign difference
    P1 = X[0:m - 1, :] - X[1:m, :]
    P2 = X[:, 0:n - 1] - X[:, 1:n]

    if is_iso:
        D = np.zeros_like(X)
        D[0:m - 1, :] = P1 ** 2
        D[:, 0:n - 1] = D[:, 0:n - 1] + P2 ** 2
        tv_out = np.sum(np.sqrt(D))
    else:
        raise NotImplementedError
        #tv_out = np.sum(np.abs(P1)) + np.sum(np.abs(P2))

    return tv_out

def psnr(ref, recon):
    mse = np.sqrt(((ref - recon) ** 2).sum() / np.prod(ref.shape))

    return 20 * np.log10((ref.max() - ref.min()) / mse)

def get_psnr_ssim(ref, recon):
    return psnr(ref, recon), ssim(ref, recon)