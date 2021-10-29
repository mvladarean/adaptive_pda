import numpy as np
from regularizers import *
from objectives import PhaseRetrieval
from optimizers import *
import pickle
from  copy  import copy

RND_SEED = 10
IMAGES_AND_OPTVALS = {
    # legend:  orig_img_path, out_folder, corruption_rate, beta_opt, p_opt, tau_opt, f_opt
    "cameraman_0.1": ("data/cameraman.tiff", "cameraman/", 0.1, 2.78e02, 1.02e0, 1e-4, 46768.05953462379),
    # "cameraman_0.2": ("data/cameraman.tiff", "cameraman/", 0.2, 58681.16993179049),
    # "barbara_0.1": ("data/barbara.png", "barbara/", 0.1, 49996.16038218798),
}
get_plot_title_reconst = lambda alg_name, corruption, psnr_tv, ssim_tv: \
        '{}_corr={:.1e}, TV - PSNR = {:.4f},\n SSIM={:.4f}' \
            .format(alg_name, corruption, psnr_tv, ssim_tv)
get_figname_to_save = lambda out_path, alg_label, corruption, psnr_tv, ssim_tv: \
    "{}{}_corr={:.1e}_psnr={:.4e}_sim={:.4e}.pdf" \
        .format(out_path, alg_label, corruption, psnr_tv, ssim_tv)

def get_right_reconstruction_sign(reconstruction, orig_image):
    sgn = np.sign(reconstruction.dot(orig_image))
    return sgn * reconstruction

def create_least_squares_data(m, N, orig_image, corruption_rate):
    np.random.seed(RND_SEED)
    rvs = scipy.stats.norm(loc=0, scale=1).rvs
    K = scipy.sparse.random(m, N, density=0.3, format='csr', data_rvs=rvs)
    num_corrupt = int(np.floor(m * corruption_rate))
    random_indices = np.random.randint(0, m, (1, num_corrupt))
    b = np.square(K @ orig_image)
    b[random_indices] = 0

    return K, b

# num_iters = 300
def tune_apda(adaptive_pda, out_path, num_iters, orig_image, shape, corruption_rate):
    loss_list = []
    label_list = []
    orig_image_vec = copy(orig_image)
    orig_image = orig_image.reshape(shape)

    lambda_sweep = np.logspace(-4, 4, 5)
    beta_sweep = np.logspace(-3, 4, 10)
    adaptive_pda.num_iters = num_iters
    for lmbd in lambda_sweep:
        for beta in beta_sweep:
            # reg_TV = TVNormForPD(lmbd, is_iso=is_isotropic)
            adaptive_pda.regularizer.lmbd = lmbd
            adaptive_pda.beta = beta
            loss, reconstruction  = adaptive_pda.run()
            sgn = np.sign(reconstruction.T.dot(orig_image_vec))
            reconstruction = reconstruction.reshape(shape)
            reconstruction = sgn * reconstruction

            loss_list.append(loss)
            label_list.append("bet={:.2e}, lmbd={:.2e}".format(beta, lmbd))

            psnr_tv, ssim_tv = get_psnr_ssim(orig_image, reconstruction)
            title_reconst = get_plot_title_reconst(adaptive_pda.label + "bet={:.2e}, lmbd={:.2e}".format(beta, lmbd), corruption_rate, psnr_tv, ssim_tv)

            plot_image_comparison(orig_image,
                                  (-1) * reconstruction, "negative reconstruction.",
                                  reconstruction, title_reconst)
            fig_name = get_figname_to_save(out_path, adaptive_pda.label, corruption_rate, psnr_tv, ssim_tv)
            save_array_as_grayscale_img(fig_name, reconstruction)

    plot_run_comparison(out_path, loss_list, label_list, f_opt=None,
                        extra_title_info="numit=" + "{:.2e}".format(num_iters))

# numiters = 300
def tune_condatvu(condat_vu, out_path, num_iters, orig_image, shape, corruption_rate):
    loss_list = []
    label_list = []
    orig_image_vec = copy(orig_image)
    orig_image = orig_image.reshape(shape)

    tau_space = np.logspace(-4, 4, 8)
    sigma_space = np.logspace(-2, 2, 5)
    normA = scipy.sparse.linalg.svds(condat_vu.A, k=1, return_singular_vectors=False)[0] if scipy.sparse.issparse(
        condat_vu.A) \
        else np.linalg.norm(condat_vu.A, 2)

    condat_vu.num_iters = num_iters
    for tau_i in tau_space:
        for p in sigma_space:
            condat_vu.tau = tau_i
            condat_vu.sigma = 1. / (p * tau_i * normA)
            loss, reconstruction  = condat_vu.run()
            if loss.size == 0:
                print("CVA diverged, ignoring stepsizes.")
                continue

            sgn = np.sign(reconstruction.T.dot(orig_image_vec))
            reconstruction = reconstruction.reshape(shape)
            reconstruction = sgn * reconstruction

            loss_list.append(loss)
            label_list.append("tau={:.2e}, p={:.2e}".format(tau_i, p))
            orig_image = orig_image.reshape(shape)
            psnr_tv, ssim_tv = get_psnr_ssim(orig_image, reconstruction)
            title_reconst = get_plot_title_reconst(condat_vu.label + "tau={:.2e}, p={:.2e}".format(tau_i, p), corruption_rate, psnr_tv, ssim_tv)

            plot_image_comparison(orig_image,
                                  (-1) * reconstruction, "negative reconstruction.",
                                  reconstruction, title_reconst)
            fig_name = get_figname_to_save(out_path, condat_vu.label, corruption_rate, psnr_tv, ssim_tv)
            save_array_as_grayscale_img(fig_name, reconstruction)

    print(loss_list)
    plot_run_comparison(out_path, loss_list, label_list, f_opt=None,
                            extra_title_info="numit=" + "{:.2e}".format(num_iters))



def run_and_plot_algs(optimizers, image_shape, orig_image, corruption_rate, out_path, f_opt, plot_last_saved = False):
    loss_list = []
    label_list = []

    ################## Plotting helpers ##################
    get_plot_title_reconst = lambda alg_name, corruption, psnr_tv, ssim_tv: \
        '{}_corr={:.1e}, TV - PSNR = {:.4f},\n SSIM={:.4f}' \
            .format(alg_name, corruption, psnr_tv, ssim_tv)
    get_figname_to_save = lambda out_path, alg_label, corruption, psnr_tv, ssim_tv: \
        "{}{}_corr={:.1e}_psnr={:.4e}_sim={:.4e}.pdf" \
            .format(out_path, alg_label, corruption, psnr_tv, ssim_tv)
    orig_image = orig_image.reshape(image_shape)

    ################## Run algs ##################
    for opt in optimizers:
        if not plot_last_saved:
            t_start = time.time()
            loss, reconstruction = opt.run()
            t_tv = time.time() - t_start
            print("Optimization time " + opt.label + ": " + str(t_tv) + " seconds.")
            reconstruction = get_right_reconstruction_sign(reconstruction.reshape(image_shape), orig_image)

            with open(out_path + "last_run" + opt.label + ".pkl", 'wb') as f:
                pickle.dump((loss, opt, reconstruction), f)

        else:
            with open(out_path + "last_run" + opt.label + ".pkl", 'rb') as f:
                (loss, opt, reconstruction) = pickle.load(f)

        loss_list.append(loss / np.linalg.norm(optimizers[0].objective.b, 2))
        label_list.append(opt.label)

        psnr_tv, ssim_tv = get_psnr_ssim(orig_image, reconstruction)
        title_reconst = get_plot_title_reconst(opt.label, corruption_rate, psnr_tv, ssim_tv)
        plot_image_comparison(orig_image,
                              (-1) * reconstruction, "negative reconstruction.",
                              reconstruction, title_reconst)
        fig_name = get_figname_to_save(out_path, opt.label, corruption_rate, psnr_tv, ssim_tv)
        save_array_as_grayscale_img(fig_name, reconstruction)

    #### normalize losses with norm b
    plot_run_comparison(out_path, loss_list, label_list, f_opt=f_opt / np.linalg.norm(optimizers[0].objective.b, 2),
                        extra_title_info="numit=" + "{:.2e}".format(optimizers[0].num_iters))

    return loss_list, label_list

def main():
    out_path = "figures/phaseret/"

    ################## Problem setup ##################
    scenario = "cameraman_0.1"
    # scenario = "cameraman_0.2"
    # scenario = "barbara_0.1"
    orig_img_path, out_folder, corruption_rate, beta_opt, p_opt, tau_opt, f_opt = IMAGES_AND_OPTVALS[scenario]
    print("Signal corruption rate = " + str(corruption_rate))
    out_path += out_folder

    shape = (84, 84)
    N = shape[0] * shape[1]
    m = int(np.ceil(N * np.log(N)))
    orig_image = load_image(orig_img_path, shape)
    save_array_as_grayscale_img(out_path + "original_image.pdf", orig_image)
    orig_image = orig_image.reshape((N, 1))

    K, b = create_least_squares_data(m, N, orig_image, corruption_rate)

    ################## Create objective and regularizer ##################
    obj_phase_ret = PhaseRetrieval(K, b)
    is_isotropic = True
    lmbd = 1e2
    reg_TV = TVNormForPD(lmbd, is_iso=is_isotropic)

    ################## Initialize iterates ##################
    np.random.seed(RND_SEED)
    y0 = np.random.normal(0, 1, (2 * orig_image.shape[0], 1))
    x0 = np.random.normal(0, 1, (orig_image.shape[0], 1))

    ################## Set up algorithms ##################
    A = reg_TV.build_finite_diff_operator(N)
    normA = np.sqrt(8)
    num_iters = 1000
    condat_vu = CondatVu(x0, y0, tau_opt, 1. / (p_opt * tau_opt * normA), obj_phase_ret, reg_TV, num_iters, A)
    adaptive_pda = APDA(x0, y0, obj_phase_ret, reg_TV, num_iters, A, beta=beta_opt, normA=normA)

    ################# Set misc details for plotting ##################
    orig_image = orig_image.reshape(shape)
    loss_list, label_list = run_and_plot_algs([adaptive_pda, condat_vu], shape, orig_image, corruption_rate, out_path,
                                              f_opt, plot_last_saved=False)

    plot_stepsize_comparison(out_path, condat_vu.tau, condat_vu.sigma, adaptive_pda.stepsizes[:, 0],
                             adaptive_pda.stepsizes[:, 1],
                             "numit = " + "{:.2e}".format(num_iters))

    ################ CVA diverges when run with APDA stepsizes #####################
    # print(type(loss_list))
    # condat_vu.tau, condat_vu.sigma = adaptive_pda.stepsizes[-1, 0], adaptive_pda.stepsizes[-1, 1]
    # loss, _ = condat_vu.run()
    # loss_list.append(loss / np.linalg.norm(b, 2))
    # label_list.append('CVA w/ APDA stepsizes')
    #
    # plot_run_comparison(out_path, loss_list, label_list, f_opt=f_opt / np.linalg.norm(b, 2),
    #                     extra_title_info="numit=" + "{:.2e}".format(num_iters))

    ################ Tune algorithms #####################
    tune_iters = 300
    #tune_apda(adaptive_pda, out_path, tune_iters, orig_image, shape, corruption_rate)
    #tune_condatvu(condat_vu, out_path, tune_iters, orig_image, shape, corruption_rate)

if __name__ == "__main__":
    main()

