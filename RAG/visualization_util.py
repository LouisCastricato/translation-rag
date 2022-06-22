# this file uses the modalcollapse library as a harness for RAG models to measure collapse during training

from modalcollapse.utils import *
from modalcollapse.indexing.faiss_utils import singular_value_plot_faiss, batch
from tqdm import tqdm
from functools import partial 
import matplotlib.pyplot as plt
# this function takes a list of datasets and produces local SVD plots for each
def get_intrinsic_dimension_plot(datasets):
    batched = batch(datasets)

    def cosine_filter_condition(pt1, pt2):
        return (np.dot(pt1, pt2) / (np.linalg.norm(pt1) * np.linalg.norm(pt2)) > 0.0)

    # build indices
    print("building indices.")
    indexes = [batched(t) for t in tqdm(range(len(datasets)))]
    # get singular values
    print("computing singular values.")
    #filtered_singular_value_plot_faiss = partial(singular_value_plot_faiss, filter_condition=cosine_filter_condition)
    return list(map(singular_value_plot_faiss, tqdm(indexes)))

# returns the AUC of min(get_intrinsic_dimension_plot(datasets))
def get_intrinsic_dimension_auc(datasets):
    # get intrinsic dimension plot
    svl = np.asarray(get_intrinsic_dimension_plot(datasets), dtype=np.float64)

    # semilogy and abs
    svl = np.log10(svl)

    # replace all -infs and NaNs with 0
    svl[np.isinf(svl)] = 0
    svl[np.isnan(svl)] = 0
    svl[svl < 0] = 0

    # for each set of svl curves find the minimum
    def get_min_svl(x):
        return np.min(x, axis=0)

    # for each set of svl curves find the maximum
    def get_max_svl(x):
        return np.max(x, axis=0)

    # get min_svl for each dataset
    min_svl = np.array(list(map(get_min_svl, svl)))
    max_svl = np.array(list(map(get_max_svl, svl)))

    # for each graph in min_svl, integrate
    return list(map(np.trapz, min_svl)), list(map(np.trapz, max_svl))

# test functionality
if __name__ == "__main__":
    mean = [0.] * 512
    cov = np.eye(512)

    # generate a list of 2 randomly initialized numpy arrays
    datasets = [get_splooch_points(set_size=50000, dim=512, splooches=100) for _ in range(2)]
    datasets2 = [np.random.multivariate_normal(mean, cov, size=50000)]

    datasets += datasets2

    # convert datasets to float32
    datasets = [np.array(d, dtype=np.float32) for d in datasets]

    # get the AUC of the intrinsic dimension plot
    auc = get_intrinsic_dimension_auc(datasets)
    print(auc)
