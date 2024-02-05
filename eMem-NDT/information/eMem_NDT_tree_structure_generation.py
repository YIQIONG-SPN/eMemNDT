import openai
openai.api_key="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"
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
def tree_structure_generate(information:list,model="text-embedding-ada-002"):

    s=[openai.Embedding.create(model=model,input=label)['data'][0]['embedding'] for label in information ]
    ###获取当前information的数组以后进行层次聚类
    Z = linkage(np.array(s), method='average', metric='euclidean')

    # 绘制聚类树
    plt.figure(figsize=(10, 6))
    dendrogram(Z)
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.show()
    return
leaf_node_information=['overtaking_from_left','overtaking_from_right','decelerate_straight','driving_away_to_left',
      'driving away to right','driving in from right','driving in from left','accelerate straight',
      'uniformly straight drriving','parallel driving in left_lane of ego vehcile','parallel driving in right_lane of ego vehicle','stopping','others driving_manner('
'there is an extra event type of participants(specified as “others”) for denoting the ambiguousinteractive event.)'
]
leaf_node_information2=['stopped','parked','Turn left','Turn right','Driving forward','Cut in to the left',
                        'Cut in to the right','Lane change to the left','Lane change to the right','other moving']
tree_structure_generate(information=leaf_node_information2)

