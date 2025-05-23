import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union
import random

import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

TASKS = [
    "Reasoning",
    "Perception",
]

SUBTASKS = [
    "Monitoring",
    "Autonomous_Driving",
    "OCR with Complex Context",
    "Diagram and Table",
    "Remote Sensing",
]




import base64
import io

from PIL import Image


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def medxpert_doc_to_visual(doc):
    return [Image.open(os.path.join("/nfs/datasets/MedXpertQA/images",img_path)).convert("RGB") for img_path in doc["images"]]

def medxpert_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    return question

def medxpert_doc_to_answer(doc):
    return doc["label"]


def medxpert_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case medxpert score), value: metric value
    """
    pattern = r"\((a|b|c|d|e|f)\)"  # 正则表达式匹配 (A)、(B)、(C)、(D)、(E)

    pred = results[0]
    pred_ans = pred.lower().split(".")[0].replace(" ", "")
    print(pred_ans)
    candidate = ["b", "c", "d", "e"]
    if len(pred_ans)>=3:
        try:
            pred_ans = re.findall(pattern, pred_ans)[0]
        except:
            pred_ans = random.choice(candidate)
        
    gt_ans = doc["label"].lower().split(".")[0].replace(" ", "")
    
    
    if gt_ans == pred_ans:
        score = [1.0]
    else:
        score = [0.0]
    
    data_dict = {"task_category": "medxpert", "pred_answer": pred_ans, "answer": gt_ans, "score": score}
    # print(score)
    return {"medxpert_perception_score": score}
    # return {f"imagenet": data_dict}

def medxpert_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    # print(results)
    scores = []
    for result in results:
        score = result[0]
        scores.append(score)
    avg_score = sum(scores) / len(scores)
    return avg_score
