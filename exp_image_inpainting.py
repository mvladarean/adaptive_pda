import math

import numpy as np
import scipy.sparse
import torch
from copy import copy, deepcopy
from regularizers import *
from objectives import LeastSquares

from optimizers import *
from helpers import plot_run_comparison, plot_stepsize_comparison, plot_image_comparison
import sys
import pickle
sys.path.append('../')

RND_SEED = 10
IMAGES_AND_OPTVALS = {
# legend is : key -> in_path, out_folder, undersampling_rate, best_beta, best_tau, best_sigma, f_opt
    "cameraman_0.4": ("data/cameraman.tiff", "cameraman/", 0.4, 0.01291549665014884, 0.8722236724965257, 0.01831255307197841, 20.8676686205678),
    "cameraman_0.2": ("data/cameraman.tiff", "cameraman/", 0.2, 0.01291549665014884, 0.8722236724965257, 0.01831255307197841, 18.69872842646567),
    "barbara_0.4": ("data/barbara.png", "barbara/", 0.4, 0.01291549665014884, 0.8722236724965257, 0.01831255307197841, 28.761346620031567),
}

def apply_random_mask(image, undersampling_rate=0.25):
   ### apply_random_mask takes an image and applies a random undersampling mask at the given rate.
    if isinstance(image, np.ndarray):
        mask = np.random.random(image.shape)
    else:
        mask = torch.rand(image.shape)
    mask[mask > undersampling_rate] = 0
    mask[mask > 0.] = 1
    return image * mask, mask

def remove_pixels(undersampling_rate, orig_image, out_path):
    np.random.seed(RND_SEED)
    print("Undersampling rate = " + str(undersampling_rate))
    ruined_image, mask = apply_random_mask(orig_image, undersampling_rate)  # 256 x 256
    save_array_as_grayscale_img(out_path + "ruined_image-" + str(undersampling_rate) + ".pdf", ruined_image)

    return ruined_image, mask

def create_least_squares_data(mask, N, orig_image):
    np.random.seed(RND_SEED)
    indices = np.nonzero(mask.flatten())[0]  # the nonzero indices of the mask flattened
    perm_mat = scipy.sparse.identity(N, format="csr")
    perm_mat = perm_mat[indices, :]
    tmp = orig_image.reshape((N, 1))
    b = perm_mat @ tmp

    return perm_mat, b

def tune_apda(adaptive_pda, out_path, f_opt, num_iters, undersampling_rate, plot_from_file=False):
    np.random.seed(RND_SEED)
    loss_list = []
    label_list = []

    if not plot_from_file:
        beta_sweep = np.logspace(-5, 2, 10)
        adaptive_pda.num_iters = num_iters
        for beta in beta_sweep:
            adaptive_pda.beta = beta
            loss, _  = adaptive_pda.run()

            loss_list.append(loss)
            label_list.append("APDAbeta = " + "{:.2e}".format(beta))
            print(loss[-1])

        with open(out_path + "beta_tuning.pkl", 'wb') as f:
            pickle.dump((loss_list, label_list, beta_sweep), f)
    else:
        with open(out_path + "beta_tuning.pkl", "rb") as f:
            (loss_list, label_list, beta_sweep) = pickle.load(f)

    plot_run_comparison(out_path, loss_list, label_list, f_opt=f_opt,
                        extra_title_info="numit=" + "{:.2e}".format(num_iters)
                                         + "_samplrate=" + "{:.2e}".format(undersampling_rate))

    best_index = get_best_run_index(loss_list)
    print("minval= " + str(loss_list[best_index][-1]))
    print("best_beta = " + str(beta_sweep[best_index]))

    return beta_sweep[best_index]

def tune_condatvu(condat_vu, out_path, f_opt, num_iters, undersampling_rate, plot_from_file=False):
    np.random.seed(RND_SEED)
    loss_list = []
    label_list = []
    tau_list= []
    sigma_list = []

    if not plot_from_file:
        normA = scipy.sparse.linalg.svds(condat_vu.A, k=1, return_singular_vectors=False)[0] if scipy.sparse.issparse(
            condat_vu.A) \
            else np.linalg.norm(condat_vu.A, 2)
        validity_fc = lambda tau, sigma: (1 / tau - condat_vu.objective.lips()) / sigma >= normA ** 2

        condat_vu.num_iters = num_iters
        p_space = np.logspace(-5, 3, 15)
        for p in p_space:
            condat_vu.tau = 1. / (normA / p + condat_vu.objective.lips())
            condat_vu.sigma = 1. / (p * normA)
            loss, _  = condat_vu.run()
            tau_list.append(condat_vu.tau)
            sigma_list.append(condat_vu.sigma)

            loss_list.append(loss)
            label_list.append("ct={:.2e}".format(p))

        xi_space = np.logspace(-5, 1, 7)
        tau_space = np.logspace(-5, 2, 10)
        for tau in  tau_space:
            for xi in xi_space:
                condat_vu.tau = tau
                condat_vu.sigma = tau * xi
                if not validity_fc(condat_vu.tau, condat_vu.sigma):
                    continue

                loss, _ = condat_vu.run()
                tau_list.append(condat_vu.tau)
                sigma_list.append(condat_vu.sigma)
                loss_list.append(loss)
                label_list.append("ct={:.2e},tau={:.2e}".format(xi, tau))

        with open(out_path + "cva_tuning.pkl", 'wb') as f:
            pickle.dump((loss_list, label_list, tau_list, sigma_list), f)
    else:
        with open(out_path + "cva_tuning.pkl", "rb") as f:
            (loss_list, label_list, tau_list, sigma_list) = pickle.load(f)

    plot_run_comparison(out_path, loss_list, label_list, f_opt=f_opt,
                        extra_title_info="numit=" + "{:.2e}".format(num_iters)
                                         + "_samplrate=" + "{:.2e}".format(undersampling_rate))

    best_index = get_best_run_index(loss_list)
    print("minval = " + str(loss_list[best_index][-1]))
    print("best_tau = " + str(tau_list[best_index]))
    print("best_sigma = " + str(sigma_list[best_index]))

    return tau_list[best_index], sigma_list[best_index]

def run_and_plot_algs(optimizers, image_size, orig_image, ruined_image, undersampling_rate, out_path, f_opt, plot_last_saved = False):
    np.random.seed(RND_SEED)
    loss_list = []
    label_list = []
    loaded_opts = {}

    ################## Set misc helpers for plotting ##################
    get_plot_title_reconst = lambda alg_name, sampl_rate, psnr_tv, ssim_tv: \
        '{}, samp_rate={:.2f}, TV - PSNR = {:.4f},\n SSIM={:.4f}' \
            .format(alg_name, sampl_rate, psnr_tv, ssim_tv)
    get_figname_to_save = lambda out_path, alg_label, undersampling_rate, psnr_tv, ssim_tv: \
        "{}{}_rate={:.2e}_psnr={:.4e}_sim={:.4e}.pdf" \
            .format(out_path, alg_label, undersampling_rate, psnr_tv, ssim_tv)


    for opt in optimizers:
        if not plot_last_saved:
            np.random.seed(RND_SEED)
            t_start = time.time()
            loss, reconstruction = opt.run()
            t_tv = time.time() - t_start
            print("Optimization time " + opt.label + ": " + str(t_tv) + " seconds.")

            reconstruction = reconstruction.reshape((image_size, image_size))
            with open(out_path + "last_run_" + opt.label + ".pkl", 'wb') as f:
                pickle.dump((loss, opt, reconstruction), f)

        else:
            with open(out_path + "last_run_" + opt.label + ".pkl", "rb") as f:
                (loss, optt, reconstruction) = pickle.load(f)
                loaded_opts[opt.label] = optt

        loss_list.append(loss)
        label_list.append(opt.label)

        psnr_tv, ssim_tv = get_psnr_ssim(orig_image, reconstruction)
        title_reconst = get_plot_title_reconst(opt.label, undersampling_rate, psnr_tv, ssim_tv)
        plot_image_comparison(orig_image,
                              ruined_image, "Original with missing pixels",
                              reconstruction, title_reconst)
        fig_name = get_figname_to_save(out_path, opt.label, undersampling_rate, psnr_tv, ssim_tv)
        save_array_as_grayscale_img(fig_name, reconstruction)

    plot_run_comparison(out_path, loss_list, label_list, f_opt=f_opt,
                        extra_title_info="numit=" + "{:.2e}".format(optimizers[0].num_iters)
                                         + "_samplrate=" + "{:.2e}".format(undersampling_rate))


    return loss_list, label_list, loaded_opts

def main():
    ################## Problem setup ##################
    out_path = "figures/inpainting/"
    scenario = "cameraman_0.4"
    # scenario = "barbara_0.4"
    # scenario = "cameraman_0.2"
    orig_img_path, out_folder, undersampling_rate, best_beta, best_tau, best_sigma, f_opt = IMAGES_AND_OPTVALS[scenario]
    out_path += out_folder

    shape = (256, 256)
    m = shape[0]
    N = shape[0] * shape[1]
    orig_image = load_image(orig_img_path, shape)
    save_array_as_grayscale_img(out_path + "original_image.pdf", orig_image)
    ruined_image, mask = remove_pixels(undersampling_rate, orig_image, out_path)

    ################## Create objective and regularizer ##################
    perm_mat, b = create_least_squares_data(mask, N, orig_image)
    obj_least_sq = LeastSquares(perm_mat, b)

    is_isotropic = True
    lmbd = 1e-2
    reg_tv_norm = TVNormForPD(lmbd, is_iso=is_isotropic)

    ################## Initialize iterates ##################
    np.random.seed(RND_SEED)
    x0 = np.random.normal(0, 1, (shape[0] * shape[1], 1))
    np.random.seed(RND_SEED)
    y0 = np.random.normal(0, 1, (2 * x0.shape[0], 1))

    ################## Set up algorithms ##################
    A = reg_tv_norm.build_finite_diff_operator(x0.shape[0])
    num_iters = 1000
    condat_vu = CondatVu(x0, y0, best_tau, best_sigma, obj_least_sq, reg_tv_norm, num_iters, A)
    adaptive_pda = APDA(x0, y0, obj_least_sq, reg_tv_norm, num_iters, A, beta=best_beta)

    ################ Run algorithms ##################
    read_from_file = False
    opt_list = [adaptive_pda, condat_vu]
    loss_list, label_list, loaded_opts = run_and_plot_algs(opt_list, m, orig_image, ruined_image, undersampling_rate, out_path, f_opt, read_from_file)

    if not read_from_file:
        plot_stepsize_comparison(out_path, condat_vu.tau, condat_vu.sigma, adaptive_pda.stepsizes[:, 0],
                                 adaptive_pda.stepsizes[:, 1],
                                 "numit = " + "{:.2e}".format(num_iters))

        condat_vu.tau, condat_vu.sigma = adaptive_pda.stepsizes[-1, 0], adaptive_pda.stepsizes[-1, 1]
        loss, reconstruction = condat_vu.run()
        loss_list.append(loss)
        label_list.append('CVA w/ APDA stepsizes')

        with open(out_path + "last_run_cva_apda_steps.pkl", 'wb') as f:
            pickle.dump((loss, condat_vu.tau, condat_vu.sigma), f)
    else:
        plot_stepsize_comparison(out_path, loaded_opts[condat_vu.label].tau, loaded_opts[condat_vu.label].sigma,
                                 loaded_opts[adaptive_pda.label].stepsizes[:, 0],
                                 loaded_opts[adaptive_pda.label].stepsizes[:, 1],
                                 "numit = " + "{:.2e}".format(num_iters))

        with open(out_path + "last_run_cva_apda_steps.pkl", 'rb') as f:
            (loss, condat_vu.tau, condat_vu.sigma) = pickle.load(f)
        loss_list.append(loss)
        label_list.append('CVA w/ APDA stepsizes')


    plot_run_comparison(out_path, loss_list, label_list, f_opt=f_opt,
                        extra_title_info="numit=" + "{:.2e}".format(num_iters)
                                         + "_samplrate=" + "{:.2e}".format(undersampling_rate))

    # ################# Tune algorithms #################

    # best_beta = tune_apda(adaptive_pda, out_path, f_opt, num_iters = 300, undersampling_rate=undersampling_rate)
    # best_tau, best_sigma = tune_condatvu(condat_vu, out_path, f_opt, num_iters = 200, undersampling_rate=undersampling_rate)


if __name__ == "__main__":
    main()