import json
import tqdm
from evaluation.evaluator import InteractiveRetrievalEvaluator

# 读取json
with open('anno/VisDial_v1.0_queries_val.json', 'r', encoding='utf-8') as f:
    visdial_data = json.load(f)
    

img_root_path = "/home/chenyang/datasets/unlabeled2017"
evaluator = InteractiveRetrievalEvaluator(img_root_path)

total_retrieval_ranks = []
for i in tqdm.trange(len(visdial_data)):
    target_img_relative_path = visdial_data[i]['img'].split('/')[-1]  # Extract the image filename from the path
    initial_query = visdial_data[i]['dialog'][0]

    retrieval_ranks = evaluator.evaluate(target_img_relative_path, initial_query)
    
    total_retrieval_ranks.append(retrieval_ranks)
    
# 保存结果
with open('evaluation/results/visdial_retrieval_ranks.json', 'w', encoding='utf-8') as f:
    json.dump(total_retrieval_ranks, f, indent=4, ensure_ascii=False)
    