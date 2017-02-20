import pandas as pd
import numpy as np

from GPfates import GPfates

etpm = pd.read_table('tapio_tcell_tpm.txt', index_col=0)
etpm = etpm[(etpm > 2).sum(1) > 2]
logexp = np.log10(etpm + 1)

tcells = pd.read_csv('tcells_rebuttal.csv', index_col=0)

m = GPfates.GPfates(tcells, logexp)

# m.dimensionality_reduction()
#
# m.store_dr()
#
# m.infer_pseudotime(priors=m.s.day_int, s_columns=['bgplvm_0', 'bgplvm_1'])

# m.infer_pseudotime(priors=m.s.day_int, s_columns=['bgplvm_2d_0', 'bgplvm_2d_1'])

# GPfates.plt.scatter(m.s.scaled_pseudotime, m.s.pseudotime); GPfates.plt.show()

# m.model_fates(X=['bgplvm_1'])

m.model_fates(X=['bgplvm_2d_1'])

# p = m.identify_bifurcation_point()
# print(p)

# m.calculate_bifurcation_statistics()


# m.fate_model.plot(); GPfates.plt.show()

m.make_fates_viz(['bgplvm_2d_0', 'bgplvm_2d_1'])

m.fates_viz.plot(); GPfates.plt.show()
