"""Decode the example participants and compare with the labels of the final system."""
import glob
import os

import numpy as np
from scipy.sparse import csr_matrix

from decode import decode

here = os.path.dirname(os.path.abspath(__file__))
for f in sorted(glob.glob(os.path.join(here, "examples", "*.npz"))):
    d = np.load(f)
    A = csr_matrix((d["A_data"], d["A_indices"], d["A_indptr"]), shape=tuple(d["A_shape"]))
    pred = decode(d["logp"], A, beta=float(d["beta"]), lo=float(d["lo"]), hi=float(d["hi"]))
    print("%s: %d windows, %d differ from the final system"
          % (os.path.basename(f)[:-4], len(pred), int((pred != d["expected"]).sum())))
