from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import torch
import eMem_NDT
import egcn_o_orignal
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, ward
import numpy  as np
file_path=""
device=''
net = egcn_o_orignal.my_model(feat_gcn=[6, 128, 256], skip=False, device=device, ad_learning=True,
                      activation=torch.nn.LeakyReLU())


def hierarchical_clustering_from_last_layer(net, file_path):
    # Load model parameters

    prameter_dict = torch.load(file_path, map_location='cuda:0')
    net.load_state_dict(prameter_dict[])
    for k, grc in enumerate(net.egcn.GRCU_layers):
        grc.evolve_weights.load_state_dict(prameter_dict[])


    # Extract weights from the last layer
    weights = net.predctor.weight.detach().cpu().numpy()

    # Perform hierarchical clustering
    model = AgglomerativeClustering(linkage="ward", distance_threshold=0, n_clusters=None)
    model = model.fit(weights)

    # Step 2: Convert to linkage matrix format
    linkage_matrix = ward(weights)

    # Step 3: Plot the dendrogram
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(linkage_matrix, ax=ax)
    plt.show()
    linkage_matrixs = linkage_matrix[:, :2]


    np.savetxt('clusters_loki.txt', linkage_matrixs, fmt =' %d')

    return linkage_matrix
# def hierarchical_clustering_from_last_layer(net, file_path):
#     # Load model parameters
#     prameter_dict = torch.load(file_path, map_location='cuda:0')
#     net.load_state_dict(prameter_dict['main'])
#     for k, grc in enumerate(net.egcn.GRCU_layers):
#         grc.evolve_weights.load_state_dict(prameter_dict[f'gcn_evolve{k}'])
#
#     # Extract weights from the last layer of 'main'
#     weights = net.predctor.weight.detach().cpu().numpy()
#
#     # Perform hierarchical clustering using 'average' method
#     linkage_matrix = linkage(weights, method='average')
#
#     # Plot the dendrogram
#     fig, ax = plt.subplots(figsize=(10, 6))
#     dendrogram(linkage_matrix, ax=ax)
#     plt.show()
#     linkage_matrixs = linkage_matrix[:, :2]
#
#     np.savetxt('clusters_loki.txt', linkage_matrixs, fmt =' %d')
#
#     return linkage_matrix

hierarchical_clustering_from_last_layer(net=net,file_path=file_path)


