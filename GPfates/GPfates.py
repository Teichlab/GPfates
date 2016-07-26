import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import GPy
from GPclust import OMGP

class GPfates(object):
    """An object for GPfates analysis"""
    def __init__(self, sample_info=None, expression_matrix=None, pseudotime_column=None):
        super(GPfates, self).__init__()

        self.s = sample_info
        self.e = expression_matrix[sample_info.index]

    def _gene_filter(gene_filter=None):
        if not gene_filter:
            Y = self.e
        else:
            Y = self.e.loc[gene_filter]

        return Y

    def infer_pseudotime(self, priors=None, gene_filter=None, s_columns=None):
        ''' Infer pseudotiem using a 1-dimensional Bayesian GPLVM
        '''
        Y = self._gene_filter(gene_filter)

        self.time_model = GPy.models.BayesianGPLVM(1, Y)

        if priors:
            for i, p in enumerate(priors):
                prior = GPy.priors.Gaussian(tcells.day_int[i], 1.)
                self.time_model.X.mean[i, [0]].set_prior(prior, warning=False)

        self.time_model.optimize()

        self.s['pseudotime'] = self.time_model.X.mean[:, [0]]

    def plot_psuedotime_uncertainty(self, **kwargs):
        yerr = 2 * np.sqrt(self.time_model.X.variance)
        plt.errorbar(self.s['pseudotime'], self.s['pseudotime'], yerr=yerr, fmt='none')
        plt.scatter(self.s['pseudotime'], self.s['pseudotime'], **kwargs)

    def dimensionality_reduction(self, gene_filter=None, name='bgplvm'):
        ''' Use a Bayesian GPLVM to infer a low-dimensional representation
        '''
        Y = self._gene_filter(gene_filter)

        gplvm = GPy.models.BayesianGPLVM(5, Y)
        self.dr_models[name] = gplvm

        gplvm.optimize()

    def model_fates(self, t='pseudotime', X=['bgplvm_1', 'bgplvm_2'], step_length=0.01):
        self.fate_model = OMGP(self.s[[t]], self.s[X], K=2, prior='DP')
        self.fate_model.optimize(step_length=step_length)
