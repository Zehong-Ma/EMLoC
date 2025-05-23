import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

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


def imagenet100_doc_to_visual(doc):
    
    return [Image.open(doc["image"]).convert("RGB")]

def imagenet100_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    return question

def imagenet100_doc_to_answer(doc):
    if doc["answer"].endswith("."):
        return doc["answer"]
    else:
        return doc["answer"]+"."


def imagenet100_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case imagenet100 score), value: metric value
    """
    pred = results[0]
    pred_ans = pred.lower().replace(".", "").replace(" ", "")
    gt_ans = doc["answer"].lower().replace(".", "").replace(" ", "")

    if gt_ans in pred_ans:
        score = [1.0]
    else:
        score = [0.0]
    
    data_dict = {"task_category": "imagenet100", "pred_answer": pred_ans, "answer": gt_ans, "score": score}
    # print(score)
    return {"imagenet100_perception_score": score}
    # return {f"imagenet": data_dict}

def imagenet100_aggregate_results(results):
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
