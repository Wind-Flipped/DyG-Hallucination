import os
import pandas as pd
import random
import networkx as nx
import json
import collections

myN = 5
myT = 10
myp = 0.5
param = "No"
seed = 42

random.seed(seed)

gtype = "dyg"
name = f"n{myN}_t{myT}_p{myp}"
my_folder_path = f'../new_text_data/edge_description/graphs_{name}/ER/{param}'
output_path = f'../new_filter_text/edge_description/{seed}/graphs_{name}/ER/{param}'
os.makedirs(output_path, exist_ok=True)


def save_to_json(data, filename):
    try:
        filename = filename + '.json'
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"保存数据到 {filename} 时出错: {e}")


for filename in os.listdir(my_folder_path):
    merged_data = []
    print(filename)
    if "cycle" in filename or "path" in filename:
        continue
    task_path = os.path.join(my_folder_path, filename)
    for graph in os.listdir(task_path):
        graph_path = os.path.join(task_path, graph)
        with open(graph_path, 'r') as file:
            data = json.load(file)
            merged_data.extend(data)
    random.shuffle(merged_data)
    answer_counts = collections.Counter(item['answer'] for item in merged_data if 'answer' in item)
    filtered_data = {answer: [] for answer in answer_counts.keys()}
    if len(answer_counts) > 5:
        max_num = 11
    else:
        max_num = 100 // len(answer_counts)
    for answer, count in answer_counts.items():
        print(f"Answer: {answer}, Count: {count}")
    for item in merged_data:
        add_num = 1 if item['answer'] == 1 and max_num == 11 else 0
        if 'answer' in item and len(filtered_data[item['answer']]) < max_num + add_num:
            filtered_data[item['answer']].append(item)

    #     # Save each filtered set to a separate file
    # for answer, data in filtered_data.items():
    #     if len(answer_counts) > 5 and (answer == -1 or answer == 0):
    #         continue
    #     output_file_path = os.path.join(output_path, filename)
    #     os.makedirs(output_file_path, exist_ok=True)
    #     output_file_path = os.path.join(output_path, filename, f"{filename}_{answer}.json")
    #     with open(output_file_path, 'w') as output_file:
    #         json.dump(data, output_file, indent=4)
    #     print(f"Saved {len(data)} entries for answer '{answer}' to {output_file_path}")
    result = []
    for answer, data in filtered_data.items():
        if len(answer_counts) > 5 and (answer == -1 or answer == 0):
            continue
        result.extend(data)
    output_file_path = os.path.join(output_path, f"{filename}.json")
    with open(output_file_path, 'w') as output_file:
        json.dump(result, output_file, indent=4)
    print(f"Saved {len(result)} entries to {output_file_path}")
