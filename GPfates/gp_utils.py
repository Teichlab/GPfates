import numpy as np
from scipy import optimize
import pandas as pd
from tqdm import tqdm
from GPy.util.linalg import pdinv, dpotrs
from GPclust import OMGP


def breakpoint_linear(x, ts, k1, k2, c1):
    '''Function representing a step-wise linear curve with one
    breakpoint located at ts.
    '''
    return np.piecewise(x, [x < ts], [lambda x: k1 * x + c1,
                                      lambda x: k2 * x + (k1 - k2) * ts + c1])


def omgp_model_bound(omgp):
    ''' Calculate the part of the omgp bound which does not depend
    on the response variable.
    '''
    GP_bound = 0.0

    LBs = []
    # Precalculate the bound minus data fit,
    # and LB matrices used for data fit term.
    for i, kern in enumerate(omgp.kern):
        K = kern.K(omgp.X)
        B_inv = np.diag(1. / ((omgp.phi[:, i] + 1e-6) / omgp.variance))
        Bi, LB, LBi, Blogdet = pdinv(K + B_inv)
        LBs.append(LB)

        # Penalty
        GP_bound -= 0.5 * Blogdet

        # Constant
        GP_bound -= 0.5 * omgp.D * np.einsum('j,j->', omgp.phi[:, i], np.log(2 * np.pi * omgp.variance))

    model_bound = GP_bound + omgp.mixing_prop_bound() + omgp.H

    return model_bound, LBs


def identify_bifurcation_point(omgp, n_splits=30):
    ''' Linear breakpoint model to infer drastic likelihood decrease
    '''
    mix_m = OMGP(omgp.X, omgp.Y, K=omgp.K, kernels=omgp.kern)
    mix_m.variance = omgp.variance

    phi = omgp.phi

    log_liks = []

    t_splits = np.linspace(mix_m.X.min(), mix_m.X.max(), n_splits)
    for t_s in tqdm(t_splits):
        mask = mix_m.X > t_s
        phi_mix = np.ones_like(phi) * 0.5
        phi_mix[mask[:, 0]] = phi[mask[:, 0]]

        mix_m.phi = phi_mix
        log_liks.append(mix_m.log_likelihood())

    x = t_splits
    y = np.array(log_liks)
    p, e = optimize.curve_fit(breakpoint_linear, x, y)

    return p[0]


def bifurcation_statistics(omgp_gene, expression_matrix):
    ''' Given an OMGP model and an expression matrix, evaluate how well
    every gene fits the model.
    '''
    bif_stats = pd.DataFrame(index=expression_matrix.index)
    bif_stats['bif_ll'] = np.nan
    bif_stats['amb_ll'] = np.nan
    bif_stats['shuff_bif_ll'] = np.nan
    bif_stats['shuff_amb_ll'] = np.nan

    # Make a "copy" of provided OMGP but assign ambiguous mixture parameters
    omgp_gene_a = OMGP(omgp_gene.X, omgp_gene.Y,
                       K=omgp_gene.K,
                       kernels=[k.copy() for k in omgp_gene.kern],
                       prior_Z=omgp_gene.prior_Z,
                       variance=float(omgp_gene.variance))

    omgp_gene_a.phi = np.ones_like(omgp_gene.phi) * 1. / omgp_gene.K

    # To control FDR, perform the same likelihood calculation, but with permuted X values

    shuff_X = np.array(omgp_gene.X).copy()
    np.random.shuffle(shuff_X)

    omgp_gene_shuff = OMGP(shuff_X, omgp_gene.Y,
                           K=omgp_gene.K,
                           kernels=[k.copy() for k in omgp_gene.kern],
                           prior_Z=omgp_gene.prior_Z,
                           variance=float(omgp_gene.variance))

    omgp_gene_shuff.phi = omgp_gene.phi

    omgp_gene_shuff_a = OMGP(shuff_X, omgp_gene.Y,
                             K=omgp_gene.K,
                             kernels=[k.copy() for k in omgp_gene.kern],
                             prior_Z=omgp_gene.prior_Z,
                             variance=float(omgp_gene.variance))

    omgp_gene_shuff_a.phi = np.ones_like(omgp_gene.phi) * 1. / omgp_gene.K

    # Precalculate response-variable independent parts
    omgps = [omgp_gene, omgp_gene_a, omgp_gene_shuff, omgp_gene_shuff_a]
    column_list = ['bif_ll', 'amb_ll', 'shuff_bif_ll', 'shuff_amb_ll']
    precalcs = [omgp_model_bound(omgp) for omgp in omgps]

    # Calculate the likelihoods of the models for every gene
    for gene in tqdm(expression_matrix.index):
        Y = expression_matrix.loc[gene]
        YYT = np.outer(Y, Y)

        for precalc, column in zip(precalcs, column_list):
            model_bound, LBs = precalc
            GP_data_fit = 0.
            for LB in LBs:
                GP_data_fit -= .5 * dpotrs(LB, YYT)[0].trace()

            bif_stats.loc[gene, column] = model_bound + GP_data_fit

    bif_stats['phi0_corr'] = expression_matrix.corrwith(pd.Series(omgp_gene.phi[:, 0], index=expression_matrix.columns), 1)
    bif_stats['D'] = bif_stats['bif_ll'] - bif_stats['amb_ll']
    bif_stats['shuff_D'] = bif_stats['shuff_bif_ll'] - bif_stats['shuff_amb_ll']

    return bif_stats
