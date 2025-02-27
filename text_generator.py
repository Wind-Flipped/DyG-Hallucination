import random
import os
import networkx as nx
import pandas as pd
import json
from tqdm import tqdm


class TaskGenerator:
    def __init__(self, N=5, T=10, p=0.5, edges=None, task_num=50):
        if edges is None:
            edges = []
        self.N = N
        # T = W t =w
        self.T = T
        self.p = p
        self.edges = edges
        self.name = f"n{self.N}_t{self.T}_p{self.p}"
        self.task_num = task_num
        self.input_folder_path = f'../graph_data/graphs_{self.name}/ER'
        self.output_folder_path = f'../new_text_data/graphs_{self.name}/ER'
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)
        self.tasks = [
            "is link",
            "when link",
            "is connect",
            "when connect",
            "degree count",
            "when degree",
            "is tri_closure",
            "when tri_closure",
            "shortest path",
            "cycle"

        ]

        self.graph_form_instruction = {
            "": "In an undirected dynamic graph, (u, v, t) means that node u and node v are linked with an edge at time t, and the edge will persist at all timestamps greater than or equal to t.",
            "[]": "In an undirected dynamic graph, u[t]=v means that node u and node v are linked with an edge at time t, and the edge will persist at all timestamps greater than or equal to t.",
            ":": "In an undirected dynamic graph, t:(u,v) means that node u and node v are linked with an edge at time t, and the edge will persist at all timestamps greater than or equal to t."

        }
        self.task_description = {
            "is link": "Your task is to answer whether there exists an edge between the given two nodes at the specified timestamp.",
            "when link": "Your task is to answer the earliest timestamp when two nodes become linked in the dynamic graph. Two nodes are considered linked if there exists a temporal edge between them.",
            "is connect": "Your task is to answer whether there exists a connection between the given two nodes at the specified timestamp in the graph. A connection means the nodes are reachable through one or more edges.",
            "when connect": "Your task is to answer the earliest timestamp when two nodes become connected in the dynamic graph. Two nodes are considered connected if there exists a path between them at the specified time.",
            "degree count": "Your task is to answer the degree of a given node at a specified timestamp in the dynamic graph. The degree is defined as the number of edges connected to the node at that time.",
            "when degree": "Your task is to find the earliest timestamp when the degree of a given node exceeds a specified threshold in the dynamic graph.",
            "is tri_closure": "Your task is to answer whether a triadic closure exists for the given three nodes at a specified timestamp in the dynamic graph. A triadic closure exists if all three nodes are directly linked to each other by edges at that time.",
            "when tri_closure": "Your task is to answer the earliest timestamp when a triadic closure forms for the given three nodes in the dynamic graph. A triadic closure forms when all three nodes become directly linked to each other by edges.",
            "shortest path": "Your task is to answer the shortest path distance between the given two nodes at a specified timestamp in the dynamic graph. ",
            "cycle": "Your task is to find the earliest timestamp when a cycle forms in the dynamic graph. A cycle exists if there is a closed loop of nodes connected by edges at that time."
        }

    def set_edges(self, edges):
        self.edges = edges

    def graph_description(self, natural):
        nodes = ", ".join([f"{i}" for i in range(self.N)])
        if natural == "natural":
            edges_str = ", ".join(
                [f"node {src} and node {tgt} are linked at time {wt}" for src, tgt, wt in self.edges])
        elif natural == "":
            edges_str = ", ".join([f"({src}, {tgt}, {wt})" for src, tgt, wt in self.edges])
        elif natural == "[]":
            edges_str = ", ".join([f"{src}[{wt}]={tgt}" for src, tgt, wt in self.edges])
        elif natural == ":":
            edges_str = ", ".join([f"{wt}:({src},{tgt})" for src, tgt, wt in self.edges])

        if natural == "natural":
            return f"Given a dynamic graph with nodes [{nodes}]. In this graph, {edges_str}. "
        else:
            return f"Given a dynamic graph with nodes [{nodes}] and edges [{edges_str}]. "

    def isLink(self, param):
        node_range = list(range(self.N))
        if param == "No":
            node1, node2 = random.sample(node_range, 2)
            t = random.randint(0, self.T)
        elif param == "Node":
            node1 = random.randint(0, self.N)
            node2 = self.N
            t = random.randint(0, self.T)
        else:
            print("wrong param in isLink")
            return
        question = f"Is there an edge between the node {node1} and the node {node2} at time {t}? Respond with 'Yes' or 'No'."
        answer = 'No'
        for src, tgt, time in self.edges:
            if (src == node1 and tgt == node2) or (src == node2 and tgt == node1):
                if time <= t:
                    answer = 'Yes'
                    break
        return question, answer

    def whenLink(self, param):
        node_range = list(range(self.N))
        if param == "No":
            node1, node2 = random.sample(node_range, 2)
        elif param == "Node":
            node1 = random.randint(0, self.N)
            node2 = self.N
        else:
            print("wrong param in whenLink")
            return
        question = f"When is the earliest time that an edge exists between the node {node1} and the node {node2}? If the answer does not exist, please respond with 'Answer: -1'."
        answer = -1
        for src, tgt, time in self.edges:
            if (src == node1 and tgt == node2) or (src == node2 and tgt == node1):
                answer = time
                break
        return question, answer

    def isConnect(self, param):
        node_range = list(range(self.N))
        if param == "No":
            node1, node2 = random.sample(node_range, 2)
            t = random.randint(0, self.T)
        elif param == "Node":
            node1 = random.randint(0, self.N)
            node2 = self.N
            t = random.randint(0, self.T)
        else:
            print("wrong param in isConnect")
            return

        question = f"Are nodes {node1} and {node2} connected at time {t}? Respond with 'Yes' or 'No'."
        answer = 'No'

        G = nx.Graph()
        G.add_nodes_from(list(range(self.N)))
        for src, tgt, time in self.edges:
            if time <= t:
                G.add_edge(src, tgt)

        if nx.has_path(G, node1, node2):
            answer = 'Yes'

        return question, answer

    def whenConnect(self, param):
        node_range = list(range(self.N))
        if param == "No":
            node1, node2 = random.sample(node_range, 2)
        elif param == "Node":
            node1 = random.randint(0, self.N)
            node2 = self.N
        else:
            print("wrong param in whenConnect")
            return

        question = f"When is the earliest time that nodes {node1} and {node2} become connected? If the answer does not exist, please respond with 'Answer: -1'."
        answer = -1

        self.edges.sort(key=lambda x: x[2])

        G = nx.Graph()
        G.add_nodes_from(list(range(self.N)))
        for src, tgt, time in self.edges:
            G.add_edge(src, tgt)
            if nx.has_path(G, node1, node2):
                answer = time
                break

        return question, answer

    def isTriClosure(self, param):
        node_range = list(range(self.N))
        if param == "No":
            nodes = random.sample(node_range, 3)
            t = random.randint(0, self.T)
        else:
            print("wrong param in isTriClosure")
            return

        question = f"Do the nodes {nodes[0]}, {nodes[1]}, and {nodes[2]} form a triadic closure at time {t}? Respond with 'Yes' or 'No'."
        answer = 'No'

        G = nx.Graph()
        G.add_nodes_from(list(range(self.N)))
        for src, tgt, time in self.edges:
            if time <= t:
                G.add_edge(src, tgt)

        subgraph = G.subgraph(nodes)
        if nx.is_connected(subgraph) and subgraph.number_of_edges() == 3:
            answer = 'Yes'

        return question, answer

    def whenTriClosure(self, param):
        node_range = list(range(self.N))
        if param == "No":
            nodes = random.sample(node_range, 3)
        else:
            print("wrong param in whenTriClosure")
            return

        question = f"When is the earliest time that nodes {nodes[0]}, {nodes[1]}, and {nodes[2]} form a triadic closure? If the answer does not exist, please respond with 'Answer: -1'."
        answer = -1

        # Sort edges by time
        self.edges.sort(key=lambda x: x[2])

        # Use a dynamic graph to check triadic closure at each time step
        G = nx.Graph()
        G.add_nodes_from(list(range(self.N)))
        for src, tgt, time in self.edges:
            G.add_edge(src, tgt)
            subgraph = G.subgraph(nodes)
            if nx.is_connected(subgraph) and subgraph.number_of_edges() == 3:
                answer = time
                break

        return question, answer

    def degreeCount(self, param):
        node_range = list(range(self.N))
        if param == "No":
            node = random.choice(node_range)
            t = random.randint(0, self.T)
        else:
            print("wrong param in degreeCount")
            return

        question = f"What is the degree of node {node} at time {t}?"
        # Create a graph with edges up to time t
        G = nx.Graph()
        G.add_nodes_from(list(range(self.N)))
        for src, tgt, time in self.edges:
            if time <= t:
                G.add_edge(src, tgt)

        answer = G.degree[node]

        return question, answer

    def whenDegree(self, param):
        node_range = list(range(self.N))
        if param == "No":
            node = random.choice(node_range)
            threshold = random.randint(1, self.N)
        else:
            print("wrong param in whenDegree")
            return

        question = f"When is the earliest time that the degree of node {node} exceeds {threshold}? If the answer does not exist, please respond with 'Answer: -1'."
        answer = -1

        self.edges.sort(key=lambda x: x[2])

        G = nx.Graph()
        G.add_nodes_from(list(range(self.N)))
        for src, tgt, time in self.edges:
            G.add_edge(src, tgt)
            if node in G and G.degree[node] > threshold:
                answer = time
                break

        return question, answer

    def shortestPathAtTime(self, param):
        node_range = list(range(self.N))
        if param == "No":
            src = random.choice(node_range)
            tgt = random.choice(node_range)
            while tgt == src:  # 确保源节点和目标节点不同
                tgt = random.choice(node_range)
        else:
            print("wrong param in shortestPathAtTime")
            return


        t = random.randint(0, self.T)
        question = f"What is the shortest path distance between node {src} and node {tgt} at time {t}? If there is no path, respond with 'Answer: -1'."
        answer = -1

        self.edges.sort(key=lambda x: x[2])
        G = nx.Graph()
        G.add_nodes_from(node_range)

        for u, v, time in self.edges:
            if time > t:
                break
            G.add_edge(u, v)

        if nx.has_path(G, src, tgt):
            answer = nx.shortest_path_length(G, src, tgt)

        return question, answer

    def hasCycleAtTime(self, param):
        if param != "No":
            print("wrong param in hasCycleAtTime")
            return

        t = random.randint(0, self.T)
        question = f"Does the graph form a cycle considering at time {t}? Respond with 'Yes' or 'No'."
        answer = "No"

        self.edges.sort(key=lambda x: x[2])
        G = nx.Graph()
        G.add_nodes_from(list(range(self.N)))

        for src, tgt, time in self.edges:
            if time > t:
                break
            G.add_edge(src, tgt)
            if len(list(nx.simple_cycles(G.to_directed()))) > 0:  # 检测是否有环
                answer = "Yes"
                break

        return question, answer

    def execute_task(self, task_type, param):
        if task_type == "is link":
            result = self.isLink(param)
        elif task_type == "when link":
            result = self.whenLink(param)
        elif task_type == "is connect":
            result = self.isConnect(param)
        elif task_type == "when connect":
            result = self.whenConnect(param)
        elif task_type == "degree count":
            result = self.degreeCount(param)
        elif task_type == "when degree":
            result = self.whenDegree(param)
        elif task_type == "is tri_closure":
            result = self.isTriClosure(param)
        elif task_type == "when tri_closure":
            result = self.whenTriClosure(param)
        elif task_type == "shortest path":
            result = self.shortestPathAtTime(param)
        elif task_type == "cycle":
            result = self.hasCycleAtTime(param)
        else:
            print("wrong type:" + task_type)
            return
        return result

    def generate_tasks(self, param="No", file_name=""):

        tasks_json = {
            "is link": [],
            "when link": [],
            "is connect": [],
            "when connect": [],
            "degree count": [],
            "when degree": [],
            "is tri_closure": [],
            "when tri_closure": []
        }
        for task in self.tasks:
            if task not in tasks_json:
                continue
            for _ in range(self.task_num):
                question, answer = self.execute_task(task, param)
                if question is None or answer is None:
                    continue

                tasks_json[task].append(
                    {
                        "graph_instruction": self.graph_form_instruction[""],
                        "graph_instruction_utv": self.graph_form_instruction["[]"],
                        "graph_instruction_tuv": self.graph_form_instruction[":"],
                        "task_description": self.task_description[task],
                        "graph_description": self.graph_description(""),
                        "graph_description_natural": self.graph_description("natural"),
                        "graph_utv": self.graph_description("[]"),
                        "graph_tuv": self.graph_description(":"),
                        "question": question,  # 获取 question
                        "answer": answer
                    })

        for key, tasks in tasks_json.items():
            if tasks is None or tasks == []:
                continue
            output_file_path = os.path.join(self.output_folder_path, key, file_name.replace(".csv", ".json"))
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, 'w') as f:
                json.dump(tasks, f, indent=4)

    def process_graphs(self, param="No"):
        self.output_folder_path = f'../new_text_data/edge_description/graphs_{self.name}/ER' + '/' + param
        for filename in tqdm(os.listdir(self.input_folder_path), desc="Process graphs"):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.input_folder_path, filename)
                df = pd.read_csv(file_path)
                edges = list(df.itertuples(index=False, name=None))
                self.set_edges(edges)
                self.generate_tasks(param=param, file_name=filename)


if __name__ == "__main__":
    task_generator = TaskGenerator(N=5, T=10, p=0.5)
    task_generator.process_graphs("No")
