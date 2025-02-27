import networkx as nx
import random
# import matplotlib.pyplot as plt
import pickle
import csv
import os
from tqdm import tqdm

class GraphGenerator:
    def __init__(self, N=5, T=10, p=0.5, gtype="ER", num_graphs=5000):
        self.N = N
        self.T = T
        self.p = p
        self.gtype = gtype
        self.num_graphs = num_graphs
        self.graphs = {gtype: []}
        self.base_dir = f"../graph_data"
        self.name = f"n{N}_t{T}_p{p}"

        # Create directories for saving graphs and images
        self.graph_dir = os.path.join(self.base_dir, f"graphs_{self.name}/{self.gtype}")
        self.image_dir = os.path.join(self.base_dir, f"images_{self.name}/{self.gtype}")
        os.makedirs(self.graph_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    def generate_dy_er(self):
        G = nx.erdos_renyi_graph(self.N, self.p)
        edges_wt = [(u, v, random.randint(0, self.T - 1)) for u, v in G.edges()]
        return G, edges_wt

    def visualize_g(self, G, edges_wt, filename):
        return
        # pos = nx.spring_layout(G)
        # edge_labels = {(u, v): t for u, v, t in edges_wt}
        # plt.figure(figsize=(10, 8))
        # nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10, font_weight="bold")
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='green')
        # plt.title("Dynamic Graph with Timestamps")
        # plt.savefig(filename)
        # plt.close()

    def save_graph_to_file(self, filename):
        """Save all graphs to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.graphs, f)

    def save_edges_to_csv(self, edges, filename):
        """Save edges with timestamps to a CSV file."""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Source', 'Target', 'Weight or Time'])
            for edge in edges:
                writer.writerow(edge)

    def generate_and_save_all_graphs(self):
        """Generate, visualize, and save multiple graphs."""
        for i in tqdm(range(self.num_graphs),desc="generate graphs"):
            G, edges_with_wt = self.generate_dy_er()
            self.graphs[self.gtype].append((G, edges_with_wt))

            # Save graph visualization
            image_filename = os.path.join(self.image_dir, f"{self.gtype}_graph_{i}.png")
            # self.visualize_g(G, edges_with_wt, image_filename)

            # Save edges to CSV
            csv_filename = os.path.join(self.graph_dir, f"{self.gtype}_graph_{i}.csv")
            self.save_edges_to_csv(edges_with_wt, csv_filename)


if __name__ == "__main__":
    generator = GraphGenerator(N=5, T=10, p=1, gtype="ER", num_graphs=5000)
    generator.generate_and_save_all_graphs()
