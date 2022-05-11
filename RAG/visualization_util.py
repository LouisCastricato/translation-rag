# this file uses the modalcollapse library as a harness for RAG models to measure collapse during training

from modalcollapse.utils import *
from modalcollapse.indexing.faiss_utils import singular_value_plot_faiss, batch
from tqdm import tqdm

# this function takes a list of datasets and produces local SVD plots for each
def get_intrinsic_dimension_plot(datasets):
    batched = batch(datasets)

    def cosine_filter_condition(pt1, pt2):
        return (np.dot(pt1, pt2) / (np.linalg.norm(pt1) * np.linalg.norm(pt2)) > 0.2)

    # build indices
    print("building indices.")
    indexes = [batched(t) for t in tqdm(range(len(datasets)))]
    # get singular values
    print("computing singular values.")
    return list(map(singular_value_plot_faiss, tqdm(indexes)))

# returns the AUC of min(get_intrinsic_dimension_plot(datasets))
def get_intrinsic_dimension_auc(datasets):
    svl = np.array(get_intrinsic_dimension_plot(datasets))

    def get_min_svl(dataset):
        min_svl = [min(s[:, t]) for s in svl]
        return min_svl
    # get min_svl for each dataset
    min_svl = np.array(list(map(get_min_svl, datasets)))
    # for each graph in min_svl, integrate
    return list(map(np.trapz, min_svl))

# test functionality
if __name__ == "__main__":
    # generate a list of 2 randomly initialized numpy arrays
    datasets = [np.random.rand(10000, 100) for _ in range(2)]
    # get the AUC of the intrinsic dimension plot
    auc = get_intrinsic_dimension_auc(datasets)
    print(auc)