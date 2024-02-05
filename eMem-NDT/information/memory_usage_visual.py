import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import seaborn as sns

class MemoryUsagePlotter:
    def __init__(self, memory_capacity, memory_usage):
        """
        Initialize the MemoryUsagePlotter with memory capacity and usage data.

        Parameters:
        - memory_capacity: A dictionary where the key is the leaf node ID and the value is the memory capacity for that node.
        - memory_usage: A list of Counter objects representing the memory usage for each leaf node.
        """
        self.memory_capacity = {int(k): v for k, v in memory_capacity.items()}
        self.memory_usage = memory_usage

    def plot_memory_usage_rate(self):
        """
        Plot the memory usage rate for each leaf node.

        Returns:
        - A bar plot showing the memory usage rate for each leaf node.
        """
        # Get node IDs for x-axis
        node_ids = list(self.memory_capacity.keys())

        # Calculate memory usage rate for each leaf node
        memory_usage_rate = [len(usage) / self.memory_capacity[node] for node, usage in zip(node_ids, self.memory_usage)]

        # Plot
        plt.figure(figsize=(12, 7))
        plt.bar(node_ids, memory_usage_rate, color='skyblue')
        plt.xlabel("Leaf Node ID")
        plt.ylabel("Memory Usage Rate")
        plt.tight_layout()
        plt.show()

    def plot_heatmap(self):
        """
        Plot a heatmap of memory usage for each leaf node.

        Returns:
        - A heatmap showing the memory usage for each leaf node.
        """
        node_ids = list(self.memory_capacity.keys())
        memory_matrix_adjusted = np.full((max(self.memory_capacity.values()), len(node_ids)), np.nan)

        for j, (node, capacity) in enumerate(self.memory_capacity.items()):
            for i in range(capacity):
                memory_matrix_adjusted[i, j] = self.memory_usage[j].get(i, 0)

        memory_matrix_adjusted_flipped = np.flipud(memory_matrix_adjusted)

        # Plot
        plt.figure(figsize=(15, 10))
        sns.heatmap(memory_matrix_adjusted_flipped, cmap="YlGnBu", annot=True, fmt=".0f", cbar_kws={'label': 'Usage Count'}, xticklabels=node_ids, yticklabels=list(reversed(range(memory_matrix_adjusted_flipped.shape[0]))))
        plt.xlabel("ID of Leaf Node")
        plt.ylabel("ID of Memory")
        plt.tight_layout()
        plt.show()

    def plot_individual_memory_usage(self, selected_nodes):
        """
        Plot the memory usage for each memory ID of the selected leaf nodes.

        Parameters:
        - selected_nodes: A list of leaf node IDs for which the memory usage should be plotted.

        Returns:
        - Bar plots showing the memory usage for each memory ID of the selected leaf nodes.
        """

        fig, axs = plt.subplots(len(selected_nodes), 1, figsize=(12, 5 * len(selected_nodes)))

        if len(selected_nodes) == 1:
            axs = [axs]

        for ax, node in zip(axs, selected_nodes):
            memory_ids = list(self.memory_usage[node].keys())
            usage_counts = list(self.memory_usage[node].values())
            ax.bar(memory_ids, usage_counts, color='lightgreen')
            ax.set_title(f"Memory Usage for Leaf Node {node}")
            ax.set_xlabel("ID of Memory")
            ax.set_ylabel("Usage Count")

        plt.tight_layout()
        plt.show()
# Sample usage:
memory_capacity_data = {'0': 9, '1': 6, '2': 10, '3': 10, '4': 11, '5': 11, '6': 9, '7': 12, '8': 20, '9': 21, '10': 24, '11': 29, '12': 20}
memory_usage_data = [Counter({4: 6678, 1: 1059, 3: 7}),
Counter({2: 5629, 0: 1938, 4: 169, 5: 5, 1: 3}),
Counter({9: 6272, 4: 1230, 7: 192, 8: 25, 1: 24, 0: 1}),
Counter({7: 3939, 9: 3044, 5: 625, 3: 50, 0: 31, 1: 26, 2: 25, 4: 3, 6: 1}),
Counter({4: 2833, 0: 1612, 8: 1480, 1: 929, 10: 254, 5: 215, 6: 212, 7: 164, 3: 17, 2: 16, 9: 12}),
Counter({5: 3387, 6: 1716, 4: 1252, 8: 430, 1: 351, 9: 286, 2: 102, 0: 84, 10: 81, 7: 35, 3: 20}),
Counter({8: 4336, 0: 1441, 7: 1319, 5: 371, 1: 271, 6: 6}),
Counter({3: 7055, 5: 302, 0: 219, 6: 80, 2: 52, 8: 12, 9: 9, 7: 7, 4: 4, 11: 4}),
Counter({5: 5847, 1: 1373, 15: 379, 16: 73, 6: 26, 19: 12, 2: 10, 18: 7, 11: 6, 3: 3, 13: 2,
7: 2, 8: 1, 0: 1, 14: 1, 17: 1}),
Counter({17: 1911, 7: 1537, 20: 902, 15: 804, 5: 612, 9: 566, 19: 430, 8: 348, 14: 291, 12: 132, 18: 117, 13: 44,
10: 24, 0: 6, 16: 4, 11: 3, 2: 3, 1: 3, 4: 3, 6: 2, 3: 2}),
Counter({13: 4988, 23: 1408, 21: 425, 16: 375, 5: 306, 22: 144, 6: 67, 20: 31}),
Counter({17: 7681, 15: 54, 20: 6, 25: 1, 22: 1, 28: 1}),

 Counter({10: 5986, 18: 790, 16: 554, 1: 380, 8: 22, 15: 7, 4: 3, 11: 2})]

plotter = MemoryUsagePlotter(memory_capacity_data, memory_usage_data)
plotter.plot_memory_usage_rate()
plotter.plot_heatmap()
plotter.plot_individual_memory_usage([4,5])