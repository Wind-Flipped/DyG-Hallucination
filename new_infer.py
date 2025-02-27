import os
import json
from tqdm import tqdm
from LLM import LLM
from template import PromptTemplate
import time
from Zhipu import GLM
from openai import OpenAI
from Deepseek import DeepSeek
from Qwen import Qwen
from local_Llama import LLaMA3_1_LLM


class InferLLM:
    example = {
        "dyg":
            {
                "is link": "Here is an example. Given a dynamic static graph with nodes[0, 1, 2] and edges[(1, 2, 1), (1, 0, 2)]. Is there an edge between the node 1 and the node 0 at time 2? The answer is 'Yes'.",
                "whenis link": "Here is an example. Given a dynamic static graph with nodes[0, 1, 2] and edges[(1, 2, 1), (1, 0, 2)]. When is the earliest time that an edge exists between the node 0 and the node 1? The answer is '2'.",
                "is connect": "Here is an example. Given a dynamic static graph with nodes[0, 1, 2] and edges[(1, 2, 1), (1, 0, 2)]. Are nodes 0 and 2 connected at time 3? The answer is 'Yes'.",
                "when connect": "Here is an example. Given a dynamic static graph with nodes[0, 1, 2] and edges[(1, 2, 1), (1, 0, 2)]. When is the earliest time that nodes 0 and 2 become connected? The answer is '2'.",
                "is tri_closure": "Here is an example. Given a dynamic static graph with nodes[0, 1, 2] and edges[(1, 2, 1), (1, 0, 2), (2 ,0 ,3)]. Do the nodes 1, 0, and 2 form a triadic closure at time 3? The answer is 'Yes'.",
                "when tri_closure": "Here is an example. Given a dynamic static graph with nodes[0, 1, 2] and edges[(1, 2, 1), (1, 0, 2), (2 ,0 ,3)]. When is the earliest time that nodes 2, 0, and 1 form a triadic closure? The answer is '3'.",
                "is link": "Here is an example. Given a dynamic static graph with nodes[0, 1, 2] and edges[(1, 2, 1), (1, 0, 2)]. Is there an edge between the node 1 and the node 0 at time 2? The answer is 'Yes'.",
                "degree count": "Here is an example. Given a dynamic static graph with nodes[0, 1, 2] and edges[(1, 2, 1), (1, 0, 2)]. What is the degree of node 2 at time 2? The answer is '1'.",
                "when degree": "Here is an example. Given a dynamic static graph with nodes[0, 1, 2] and edges[(1, 2, 1), (1, 0, 2)]. When is the earliest time that the degree of node 1 exceeds 1? The answer is '2'.",
            }
    }

    def __init__(self, model_name="meta-llama/llama-3.1-8b-instruct",
                 api_key=""):
        self.model_name = model_name
        self.api_key = api_key
        if model_name == "llama3.1":
            self.llm = LLaMA3_1_LLM(
                mode_name_or_path="")
        self.template = PromptTemplate()

    def process_json_files_in_folder(self, input_folder_path, cot, explain, graph_old, nature, need_json):
        results = []

        with open(input_folder_path, 'r', encoding='utf=8') as infile:
            data = json.load(infile)
        if not data:
            return
        if cot == "zero":
            prompts = []
        elif cot == "think_step":
            prompts = ["Think step by step."]
        elif cot == "construct":
            prompts = ["Let’s construct a graph with the nodes and edges first."]
        elif cot == "confidence":
            prompts = ["Give your confidence (0% to 100%) after your explanation."]
        elif cot == "node2time":
            prompts = [
                "Think node then time."
            ]
        elif cot == "time2node":
            prompts = [
                "Think time than node."
            ]
        elif cot == "one_edge":
            prompts = [
                '''The "Explanation" field must be output in the following format. Please think step by step, with each step starting with 'STEP_index:'. For each step, only consider adding one new edge based on the previous step, and declare which edge you are considering at the beginning of that step. In the "Explanation" field, output your thought process following the example.
Example:
Suppose you are solving a problem where you need to determine the earliest timestamp when node A and B become connected in a dynamic graph.
##STEP_1: Consider the edge between node A and node C at time t_1.
At time t_1, an edge links node A to node C. This means nodes A and C are directly linked. However, since the target nodes A and B are not yet connected, we will proceed to consider additional edges.
##STEP_2: Consider the edge between node B and node C at time t_2.
At time t_2, an edge links node B to node C. This creates an indirect connection between node A and node B through node C. Now, nodes A and B are connected for the first time at timestamp t_2.
##Conclusion:
By evaluating edges in the dynamic graph sequentially, we determine that the earliest timestamp when nodes A and B become connected is t_2.'''
            ]
        elif cot == "one_node":
            prompts = [
                '''The "Explanation" field must be output in the following format. Please think step by step, with each step starting with 'STEP_index:'. For each step, only consider adding one new node based on the previous step, and declare which edge you are considering at the beginning of that step. In the "Explanation" field, output your thought process following the example.
Example:
Suppose you are solving a problem where you need to determine the earliest timestamp when node A and node B become connected in a dynamic graph.
##STEP_1: Consider node A.
At the first step, we examine the edges linked to node A. Node A is connected to node C,D at time t_1. This means node A has a direct connection to node C,D at t_1, but it is not yet connected to node B. The next step will involve examining the edges linked to node B to determine if a connection to node A exists.
##STEP_2: Consider node B.
At the next step, we examine the edges linked to node B. Node B is connected to node C,E at time t_2. Since node C is already connected to node A from step 1, this creates an indirect connection between node A and node B through node C. Thus, nodes A and B become connected for the first time at t_2.
##Conclusion:
By examining the edges linked to nodes A and B, we determine that the earliest timestamp when nodes A and B become connected is t_2.''']
        elif cot == "one_shot":
            if "is link" in input_folder_path:
                prompts = [self.example["dyg"]["is link"]]
            elif "when link" in input_folder_path:
                prompts = [self.example["dyg"]["when link"]]
            elif "is tri_closure" in input_folder_path:
                prompts = [self.example["dyg"]["is tri_closure"]]
            elif "when tri_closure" in input_folder_path:
                prompts = [self.example["dyg"]["when tri_closure"]]
            elif "is connect" in input_folder_path:
                prompts = [self.example["dyg"]["is connect"]]
            elif "when connect" in input_folder_path:
                prompts = [self.example["dyg"]["when connect"]]
            elif "when degree" in input_folder_path:
                prompts = [self.example["dyg"]["when degree"]]
            elif "degree count" in input_folder_path:
                prompts = [self.example["dyg"]["degree count"]]
            else:
                print("wrong infile")
        elif cot == "verify":
            prompts = [
                "You must think step by step, with each step starting with '##STEP_index:'. ",
                "You must list all the involved nodes and edges.",
                "You must ensure that the involved edges and nodes exist at the corresponding timestamps.",
                '''Example:
Suppose you are solving a problem where you need to determine the earliest timestamp when node A and node B become connected in a dynamic graph.
##STEP_1: 
At the first step, we examine the edges linked to node A. Node A is connected to node C,D at time t_1. This means node A has a direct connection to node C,D at t_1, but it is not yet connected to node B. The next step will involve examining the edges linked to node B to determine if a connection to node A exists.
##STEP_2: 
At the next step, we examine the edges linked to node B. Node B is connected to node C,E at time t_2. Since node C is already connected to node A from step 1, this creates an indirect connection between node A and node B through node C. Thus, nodes A and B become connected for the first time at t_2.
##Conclusion:
By examining the edges linked to nodes A and B, we determine that the earliest timestamp when nodes A and B become connected is t_2.'''
            ]
        else:
            print("cot is wrong")
            return

        if graph_old:
            graph_form_instruction = "In an undirected dynamic graph, (u, v, t) means that node u and node v are linked with an edge at time t."
        else:
            graph_form_instruction = "In an undirected dynamic graph, (u, v, t) means that node u and node v are linked with an edge at time t, and the edge will persist at all timestamps greater than or equal to t."

        for item in tqdm(data, desc=f"{self.model_name} {cot}: Processing one json"):
            if nature:
                input_text = f'''
-{graph_form_instruction}
-{item["task_description"]}
-{item["graph_description_natural"]}
-Question: {item["question"]} 
            '''
            else:
                input_text = f'''
-{graph_form_instruction}
-{item["task_description"]}
-{item["graph_description"]}
-Question: {item["question"]} 
                '''
            input_text = self.template(request=input_text, prompt=prompts, need_explanation=explain,
                                       need_json=need_json)
            print(input_text)
            response = self.llm(input_text)
            self.llm.clear_history()
            print(response)
            results.append(
                {"graph_description": item["graph_description"],
                 "graph_description_natural": item["graph_description_natural"],
                 "question": item["question"],
                 "input": input_text,
                 "output": response, "truth": item["answer"]})
        return results

    def process_all_folders(self, base_folder_path, base_output_path, cot, explain, graph_form, nature, need_json):
        for json_name in os.listdir(base_folder_path):
            if "json" not in json_name:
                continue
            if "path" in json_name or "cycle" in json_name:
                continue
            json_folder_path = base_folder_path + '/' + json_name
            print(json_folder_path)
            output_file_path = os.path.join(base_output_path, json_name)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            folder_result = self.process_json_files_in_folder(json_folder_path, cot, explain, graph_form,
                                                              nature, need_json)
            if folder_result:
                with open(output_file_path, 'w', encoding='utf-8') as outfile:
                    json.dump(folder_result, outfile, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    seed = 42

    param = "No"
    model = {
        "llama3.1": "llama3.1",
    }
    for model_name in ["llama3.1"]:
        llm = InferLLM(model_name=model[model_name])
        for graph_form in [""]:
            for explain in [""]:
                for natural in [""]:
                    for need_json in [""]:
                        for n in ['5']:
                            for COT in ["zero"]:
                                base_folder_path = f'../new_filter_text/{seed}/graphs_n{n}_t10_p0.8/ER/{param}'
                                base_output_path = f'../new_output/{seed}/{model_name}/graphs_n{n}_t10_p0.8/ER/{param}'
                                print(COT)
                                if COT:
                                    base_output_path = base_output_path + "/" + COT + "_" + explain + "_" + natural + "_" + graph_form + "_" + need_json
                                if not os.path.exists(base_output_path):
                                    os.makedirs(base_output_path)
                                start_time = time.time()
                                llm.process_all_folders(base_folder_path, base_output_path, COT, explain == "explain",
                                                        graph_form == "old",
                                                        natural == "natural", need_json == "json")
                                end_time = time.time()
                                elapsed_time = end_time - start_time
                                print(f"Total processing time: {elapsed_time:.2f} seconds")
