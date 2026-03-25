from reasoning_eval.eval import eval_reasoning
from consis import eval_consist
from img_quality_eval.eval import eval_image_quality
from phy_rationality_eval.eval import eval_phy_rationality
from cal_score.relax import cal_relax
from cal_score.strict import cal_strict
from data_utils.check_resume import *
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import json
import os 

# data
data_json = "result.json" # path to video result json with frame_path
root_dir = "" ## Intermediate file storage root directory  Generation model + Evaluation model naming
os.makedirs(root_dir,exist_ok=True)
#-----socre files---------
relax_save_root = ""
strcit_save_root = ""
os.makedirs(relax_save_root, exist_ok=True)
os.makedirs(strcit_save_root, exist_ok=True)

#api config
GPT_URL = ""
GPT_KEY = ""

#****** default config *************
GPT_MODEL = "gpt-5"
GPT_MODEL_IMG_QUALITY = "gpt-5-mini"
WOKERS = 20 # Parallel evaluation count
#-----reasoning interval-----------
reasoning_result_json = os.path.join(root_dir, "temp_reasoning_result")  #Only save results of gpt vqa
reasoning_score_json = os.path.join(root_dir, "temp_reasoning_score")
os.makedirs(reasoning_result_json, exist_ok=True)
os.makedirs(reasoning_score_json, exist_ok=True)
#-----consis interval---------
consis_result_json = os.path.join(root_dir, "temp_consis_result")  
os.makedirs(consis_result_json, exist_ok=True)
#-----image_quality interval---------
image_quality_result_json = os.path.join(root_dir, "temp_image_quality_result")
os.makedirs(image_quality_result_json, exist_ok=True)
#-----phy_rationality interval---------
phy_rationality_result_json = os.path.join(root_dir, "temp_phy_rationality_result")
os.makedirs(phy_rationality_result_json, exist_ok=True)


#for data in datas:
def eval_one(data):
    print(f"Evaluating {data['task_id']}... \n")

    reasoning_score = eval_reasoning(data, root_dir, reasoning_result_json, reasoning_score_json, GPT_URL, GPT_KEY, GPT_MODEL)
    print(f"  Reasoning socre: {reasoning_score} \n")

    consis_score = eval_consist(data, consis_result_json, GPT_URL, GPT_KEY, GPT_MODEL)
    print(f"  Consistency score: {consis_score}\n")

    iq_score = eval_image_quality(data, image_quality_result_json, GPT_URL, GPT_KEY, GPT_MODEL_IMG_QUALITY)
    print(f"  Image Quality score: {iq_score}\n")

    if data['task_id'].startswith("log"):
        print("  [Skip] logical task for phy_rationality eval\n")
        return
    phy_rat_score = eval_phy_rationality(data, phy_rationality_result_json, GPT_URL, GPT_KEY, GPT_MODEL)
    print(f"  Phy Rationality score: {phy_rat_score}\n")

if __name__ == "__main__":
    with open(data_json, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    # TODO: Auto resume
    json_paths = [
        reasoning_score_json,
        consis_result_json,
        image_quality_result_json,
        phy_rationality_result_json,   
    ]

    done_list = filter_by_common_task_ids(json_paths, datas)

    sensitive_list = [] 
    
    eval_datas = [] 
    for data in datas:
        if data["task_id"] not in done_list and data["task_id"] not in sensitive_list:
            eval_datas.append(data)

    print(f"[INFO] data need to eval: {len(eval_datas)}")
    with ProcessPoolExecutor(max_workers=WOKERS) as ex:
        process_map(eval_one, eval_datas, max_workers=WOKERS)
    print("inference done")

    for folder in json_paths:
        merge_datas =[]
        merge_count = 0
        for dir_name, _, files in os.walk(folder):
            for file in files:
                if file.split('.')[-1] == "json":
                    with open(os.path.join(dir_name, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    merge_datas.append(data)
                    merge_count += 1
            name = os.path.basename(folder)
            merge_file = os.path.join(root_dir, f"{name}.json")
            with open(merge_file, 'w', encoding='utf-8') as f:
                json.dump(merge_datas, f, indent=4, ensure_ascii = False)
            print(f"Merge {merge_count} results in {merge_file} :)")
    
    #cal strict & relax score
    cal_relax(root_dir, relax_save_root)
    cal_strict(root_dir, strcit_save_root)             
