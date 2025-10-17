import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import GPy
from GPclust import OMGP

from .gp_utils import bifurcation_statistics
from .gp_utils import identify_bifurcation_point

class GPfates:
    """An object for GPfates analysis."""
    def __init__(self,
                 sample_info: pd.DataFrame,
                 expression_matrix: pd.DataFrame):
        """
        Create a GPfates analysis.
        Parameters
        ----------
        sample_info : pd.DataFrame
            Dataframe containing sample metadata (e.g. pseudotime). The DataFrame should be cells x metadata.
        expression_matrix : pd.DataFrame
            Dataframe containing expression data of shape genes x cells.
        """
        super().__init__()
        self.s = sample_info
        self.e = expression_matrix[sample_info.index]
        self.dr_models = {}

    def _gene_filter(self, gene_filter=None):
        """Helper for the common operation of optionally filtering genes."""
        if not gene_filter:
            Y = self.e
        else:
            Y = self.e.loc[gene_filter]
        return Y

    def infer_pseudotime(self, priors=None, gene_filter=None, s_columns=None):
        """Infer pseudotime using a 1-dimensional Bayesian GPLVM."""
        if s_columns:
            Y = self.s[s_columns].values
        else:
            Y = self._gene_filter(gene_filter).values.T

        self.time_model = GPy.models.BayesianGPLVM(Y, 1, init='random')

        self.time_model.rbf.lengthscale.constrain_fixed(2., warning=False)
        self.time_model.rbf.variance.constrain_fixed(200., warning=False)

        if priors is not None:
            for i, p in enumerate(priors):
                prior = GPy.priors.Gaussian(p, 2.)
                self.time_model.X.mean[i, [0]].set_prior(prior, warning=False)

        self.time_model.optimize(max_iters=2000, messages=True)

        self.s['pseudotime'] = self.time_model.X.mean[:, [0]]

    def plot_psuedotime_uncertainty(self, **kwargs):
        yerr = 2 * np.sqrt(self.time_model.X.variance)
        plt.errorbar(self.s['pseudotime'], self.s['pseudotime'], yerr=yerr, fmt='none')
        plt.scatter(self.s['pseudotime'], self.s['pseudotime'], zorder=2, **kwargs)

    def dimensionality_reduction(self, gene_filter=None, name='bgplvm'):
        """Use a Bayesian GPLVM to infer a low-dimensional representation."""
        Y = self._gene_filter(gene_filter).values.T

        gplvm = GPy.models.BayesianGPLVM(Y, 5)
        self.dr_models[name] = gplvm

        gplvm.optimize(max_iters=2000, messages=True)

    def store_dr(self, name='bgplvm', dims=[0, 1]):
        """Put a low-dimensional representation in the sample table."""
        gplvm = self.dr_models[name]
        for d in dims:
            self.s['{}_{}'.format(name, d)] = gplvm.X.mean[:, [d]]

    def model_fates(self,
                    t='pseudotime',
                    X=['bgplvm_0', 'bgplvm_1'],
                    C=2,
                    step_length=0.01,
                    maxiter: int = 1000):
        """Model multiple cell fates using OMGP."""
        self.fate_model = OMGP(self.s[[t]].values, self.s[X].values, K=C, prior_Z='DP')
        self.fate_model.variance.constrain_fixed(0.05)
        self.fate_model['(.*)lengthscale'].constrain_fixed(1.)
        self.fate_model.hyperparam_interval = 1e3
        self.fate_model.optimize(maxiter=maxiter, step_length=step_length)

    def make_fates_viz(self, s_columns=['bgplvm_0', 'bgplvm_1']):
        """Make an OMGP model based on the fate model which visualizes the
        trends in a representative space."""
        XY = self.s[s_columns].values
        self.fates_viz = OMGP(self.fate_model.X, XY, prior_Z='DP')
        self.fates_viz[:] = self.fate_model[:]
        self.fates_viz.phi = self.fate_model.phi

    def identify_bifurcation_point(self, n_splits=30):
        """Linear breakpoint model to infer drastic likelihood decrease."""
        omgp = self.fate_model
        return identify_bifurcation_point(omgp, n_splits=n_splits)

    def calculate_bifurcation_statistics(self, gene_filter=None):
        """Calculate the bifurcation statistics for all or a subset of genes."""
        return bifurcation_statistics(self.fate_model, self.e)
