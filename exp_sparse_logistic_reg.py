import numpy as np
import scipy.linalg as LA
import scipy.sparse
import scipy.sparse.linalg as spr_LA
from regularizers import *
from objectives import *
from optimizers import *
import pickle
RND_SEED = 10
DATASETS_AND_OPTVALS = {
    # legend is : key -> in_path, out_folder, best_beta, best_tau, best_sigma, f_opt
    "ijcnn": ("data/ijcnn1.bz2", "ijcnn/", 2.68e3, 0.0009869919142811382, 11.253355826007647, 9034.985436395973),
    "a9a": ("data/a9a", "a9a/", 5.18e4, 0.0002655986002082452, 78.96522868499726, 12123.188294441956),
    "mushrooms": ("data/mushrooms", "mushrooms/", 3.16e1, 0.0009936668983870924, 5.878016072274912, 675.9896825919595),
    "covtype": ("data/covtype.libsvm.binary.bz2", "covtype/", 3.73e-1, 7.728825775486957e-06, 1e-06, 392908.30868796416),
}

def tune_apda(adaptive_pda, out_path, f_opt, num_iters):
    betas = np.logspace(-3, 6, 15)

    loss_list = []
    label_list = []
    adaptive_pda.num_iters= num_iters
    for beta in betas:
        adaptive_pda.beta = beta
        loss, _ = adaptive_pda.run()
        loss_list.append(loss)
        label_list.append("APDAbeta=" + "{:.2e}".format(beta))

    plot_run_comparison(out_path, loss_list, label_list, f_opt=f_opt,
                        extra_title_info="gridsearch_beta_numit=" + "{:.2e}".format(num_iters))

    best_index = get_best_run_index(loss_list)
    print("best_beta = " + str(betas[best_index]))

    return betas[best_index]

def tune_condatvu(condat_vu, out_path, dataset, f_opt, num_iters):
    loss_list = []
    label_list = []
    tau_list = []
    sigma_list = []

    condat_vu.num_iters = num_iters
    normA = scipy.sparse.linalg.svds(condat_vu.A, k=1, return_singular_vectors=False)[0] if scipy.sparse.issparse(condat_vu.A) \
                else np.linalg.norm(condat_vu.A, 2)
    validity_fc = lambda tau, sigma: (1 / tau - condat_vu.objective.lips()) / sigma >= normA ** 2

    ################## First tune with p ##################
    ps = np.logspace(-5, 6, 40)
    for p in ps:
        condat_vu.tau = 1 / (normA / p + condat_vu.objective.lips())
        tau_list.append(condat_vu.tau)

        condat_vu.sigma = 1 / (p * normA)
        sigma_list.append(condat_vu.sigma)

        loss, _ = condat_vu.run()
        loss_list.append(loss)
        label_list.append("CondatVup=" + "{:.2e}".format(p))

    ################## Second tune free ##################
    xi_space = np.logspace(-5, 2, 7)
    tau_space = np.logspace(-10, 2, 10)
    for tau in tau_space:
        for xi in xi_space:
            condat_vu.tau = tau
            condat_vu.sigma = tau * xi
            if not validity_fc(condat_vu.tau, condat_vu.sigma):
                continue
            tau_list.append(condat_vu.tau)
            sigma_list.append(condat_vu.sigma)

            loss, _ = condat_vu.run()
            loss_list.append(loss)
            label_list.append("CondatVu;" + "tau={:.2e}; sigma={:.2e}".format(condat_vu.tau, condat_vu.sigma))

    plot_run_comparison(out_path, loss_list, label_list, f_opt=f_opt,
                        extra_title_info="gridsearch_p_numit=" + "{:.2e}".format(num_iters))

    with open(out_path + "cva-tune-" + dataset + ".pkl", 'wb') as f:
        pickle.dump((loss_list, label_list, tau_list, sigma_list), f)

    best_index = get_best_run_index(loss_list)
    best_tau = tau_list[best_index]
    best_sigma = sigma_list[best_index]
    print(best_index)
    print("best_tau = " + str(best_tau))
    print("best_sigma = " + str(best_sigma))

    return  best_tau, best_sigma

def get_strong_convexity_status(K):
    tmp = K.T @ K
    tmp = (tmp + tmp.T) / 2
    np.random.seed(RND_SEED)
    try:
        print(tmp.toarray().dtype)
        eigs = np.linalg.svd(tmp.toarray(), compute_uv=False, hermitian=True)
        smallest_eigval = np.min(eigs)
        largest_eigvale = np.max(eigs)
        print("Smallest eigval = " + str(smallest_eigval))
        print("Condition number = " + str(largest_eigvale / smallest_eigval))
    except Exception as e:
        print(e)
        print("No eigenvalue to sufficient accuracy. Default to neutral setting.")
        smallest_eigval = -1

    is_strcnvx = False
    if smallest_eigval > 1e-13: # anything below this value is essentially 0 for us
        is_strcnvx = True
    print("Is strongly convex = " + str(is_strcnvx))

    return is_strcnvx

def run_algs_and_plot_convergence(optimizers, f_opt, out_path, dataset):
    ############## Run algorithms ##################
    loss_list = []
    label_list = []

    for opt in optimizers:
        loss, _ = opt.run()
        loss_list.append(loss)
        label_list.append(opt.label)

    norm_loss_list = [l / f_opt for l in loss_list]
    plot_run_comparison(out_path, norm_loss_list, label_list, f_opt=f_opt / f_opt,
                        extra_title_info=dataset + "_numit=" + "{:e}".format(optimizers[0].num_iters))

def main(dataset):
    ################## Problem setup ##################
    out_path = "figures/logistic/"
    file_path, out_folder, beta_opt, tau_opt, sigma_opt, f_opt = DATASETS_AND_OPTVALS[dataset]

    m, dim, K, b = read_libsvm_data(file_path)
    is_strcnvx = get_strong_convexity_status(K)

    ################## Create objective and regularizer ##################
    f_logistic_loss = LogisticRegression(K, b)

    lmbd = 0.005 * LA.norm(K.T.dot(b), np.inf)
    reg_l1 = L1Norm(lmbd)

    ################## Initialize iterates ##################
    np.random.seed(RND_SEED)
    x0 = np.random.normal(0, 0.00001, (dim, 1))
    y0 = np.random.normal(0, 0.00001, (dim, 1))
    num_iters = 2000

    ################## Set up algorithms ##################
    A = scipy.sparse.identity(dim, format="csr")
    normA = 1
    is_full_row_rank_A = True

    adaptive_pda = APDA(x0, y0, objective=f_logistic_loss, regularizer=reg_l1, num_iters=num_iters, A=A, beta=beta_opt,
                        normA=normA)
    if (is_strcnvx and is_full_row_rank_A):
        adaptive_pda.is_strcnvx = True
        adaptive_pda.label += "-strcnv"

    condat_vu = CondatVu(x0, y0, tau=tau_opt, sigma=sigma_opt, objective=f_logistic_loss, regularizer=reg_l1,
                         num_iters=num_iters, A=A)
    fista = FISTA(x0, 1. / (f_logistic_loss.lips()), objective=f_logistic_loss, regularizer=reg_l1, num_iters=num_iters)

    ################## Run algorithms & plot stuff ##################
    run_algs_and_plot_convergence([adaptive_pda, condat_vu, fista], f_opt, out_path, dataset)

    plot_stepsize_comparison(out_path, condat_vu.tau, condat_vu.sigma, adaptive_pda.stepsizes[0:1000, 0],
                             adaptive_pda.stepsizes[0:1000, 1],
                             "numit=" + "{:e}".format(num_iters) + ",dataset=" + dataset)

    ################## Tune algorithms ##################
    # tune_iters = 500
    # tune_apda(adaptive_pda, out_path, f_opt, tune_iters)
    # tune_condatvu(condat_vu, out_path, dataset, f_opt, tune_iters)



if __name__ == "__main__":
    datasets = ["ijcnn", "a9a", "mushrooms", "covtype"]

    for dataset in datasets:
        main(dataset)
