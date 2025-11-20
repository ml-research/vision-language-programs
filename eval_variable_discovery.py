from datetime import datetime
from urllib import response
import Levenshtein
from tqdm import tqdm
from rtpt import RTPT
import re
import ast
import os
import difflib
import pandas as pd
import numpy as np


from utils.args import parse_args
from utils.dataset_utils import load_data
from discover_properties import discover_properties, discover_objects, discover_actions

from method.experiment_helper import make_program_checker_with_accuracy
from method.run_experiment import run_algorithm
from utils.prompters import get_prompter
from eval import n_tasks_per_dataset, params_per_dataset


Clevr_objects_from_gt = {
    "Class_0": ["cube", "cylinder"],
    "Class_1": ["cube", "sphere"],
    "Class_2": ["sphere"],
}

Clevr_properties_from_gt = {
    "Class_0": ["large", "gray"],
    "Class_1": ["small", "metal"],
    "Class_2": ["large", "blue", "small", "yellow"],
}

cocologic_objects_from_gt = {
    "Ambiguous Pairs (Pet vs Ride Paradox)": [
        "dog",
        "bicycle",
        "motorcycle",
    ],  # no cats
    "Pair of Pets": ["cat", "dog", "bird"],
    "Rural Animal Scene": ["cow", "horse", "sheep"],
    "Conflicted Companions (Leash vs Licence)": ["dog", "car"],
    "Animal Meets Traffic": [
        "horse",
        "cow",
        "sheep",
        "bus",
        "traffic light",
    ],  # no cars
    "Occupied Interior": ["couch", "chair", "person"],
    "Empty Seat": ["couch", "chair", "person"],
    "Odd Ride Out": ["bicycle", "motorcycle", "car", "bus"],
    "Personal Transport XOR Car": ["person", "motorcycle", "car"],
    "Unlikely Breakfast Guests": ["bowl", "cat", "horse", "cow", "sheep"],  # no dog
}


def get_objects_from_hoi_gt(gt_rule):
    # split gt by "++"
    parts = gt_rule.split("++")
    object_part = parts[1]
    # replace _ with space
    object_part = object_part.replace("_", " ")
    return [object_part]


def get_actions_from_hoi_gt(gt_rule):
    # split gt by "++"
    parts = gt_rule.split("++")
    action_part = parts[0]
    actions = action_part.split("_")
    final_actions = []
    fill_words = [
        "on",
        "at",
        "in",
        "with",
        "to",
        "and",
        "or",
        "multiple",
        "person",
        "like",
        "about",
        "inside",
        "under",
        "camera",
    ]
    for a in actions:
        if a not in fill_words:
            final_actions.append(a)
    return final_actions


def almost_equal(a, b, max_distance=1):
    return Levenshtein.distance(a, b) <= max_distance


def check_in_list(item, lst):
    for elem in lst:
        if almost_equal(item, elem):
            return True
    return False


def parse_list_from_response(response):

    # Step 1: Extract the list directly from the response
    list_match = re.search(r"\s*(\[.*\])", response, re.DOTALL)
    if list_match:
        list_str = list_match.group(1)

        # Step 2: Safely parse the string into a Python list
        parsed_list = ast.literal_eval(list_str)
        print(parsed_list)

        return parsed_list
    else:
        raise ValueError("No list found after in the response.")


def parse_score_from_response(response):
    """Format:##
    Format:
    Required: [list]
    Found: [list]
    Missing: [list]
    Output: [score]
    """
    match = re.search(r"Output:\s*([0-9]*\.?[0-9]+)", response)
    if match:
        score = float(match.group(1))
        return score
    else:
        # Handle parse error
        raise ValueError("Could not parse output score")


def parse_match_from_response(response):
    match = re.search(r"MATCH:\s*(True|False)", response)
    if match:
        is_match = match.group(1) == "True"
        return is_match
    else:
        # Handle parse error
        raise ValueError("Could not parse match result")


def retrieve_objects_from_gt(gt_rule, dataset):

    if dataset == "CLEVR-Hans3-unconfounded":
        objects = Clevr_objects_from_gt[gt_rule]
    elif dataset == "cocologic":
        objects = cocologic_objects_from_gt[gt_rule]
    elif dataset == "bongard-hoi":
        objects = get_objects_from_hoi_gt(gt_rule)
    else:
        prompt_path = "prompts/judge/extract_objects.txt"
        # read prompt
        with open(prompt_path, "r") as f:
            prompt_template = f.read()
        prompt = prompt_template.replace("{gt_rule}", gt_rule)

        # prompter = get_prompter("Qwen2.5-VL-7B-Instruct", "gt_rules", 0)
        prompter = get_prompter("gpt-4o", "gt_rules", 0)
        response = prompter.prompt_with_text(prompt, max_new_tokens=512)
        # prompter.remove_from_gpu()

        # parse Python list from response
        try:
            objects = parse_list_from_response(response)
        except:
            objects = []

    return objects


def retrieve_properties_from_gt(gt_rule, dataset):

    if dataset == "CLEVR-Hans3-unconfounded":
        properties = Clevr_properties_from_gt[gt_rule]
    elif dataset == "cocologic":
        properties = []
    elif dataset == "bongard-hoi":
        properties = []
    else:
        prompt_path = "prompts/judge/extract_properties.txt"
        # read prompt
        with open(prompt_path, "r") as f:
            prompt_template = f.read()
        prompt = prompt_template.replace("{gt_rule}", gt_rule)
        # prompter = get_prompter("Qwen2.5-VL-7B-Instruct", "gt_rules", 0)
        prompter = get_prompter("gpt-4o", "gt_rules", 0)
        response = prompter.prompt_with_text(prompt, max_new_tokens=512)
        # prompter.remove_from_gpu()

        # parse Python list from response
        try:
            properties = parse_list_from_response(response)
        except:
            properties = []

    return properties


def retrieve_actions_from_gt(gt_rule, dataset):

    if dataset == "cocologic":
        actions = []
    elif dataset == "CLEVR-Hans3-unconfounded":
        actions = []
    elif dataset == "bongard-hoi":
        actions = get_actions_from_hoi_gt(gt_rule)
    else:
        prompt_path = "prompts/judge/extract_actions.txt"
        # read prompt
        with open(prompt_path, "r") as f:
            prompt_template = f.read()
        prompt = prompt_template.replace("{gt_rule}", gt_rule)

        # prompter = get_prompter("Qwen2.5-VL-7B-Instruct", "gt_rules", 0)
        prompter = get_prompter("gpt-4o", "gt_rules", 0)
        response = prompter.prompt_with_text(prompt, max_new_tokens=512)
        # prompter.remove_from_gpu()

        # parse Python list from response
        try:
            actions = parse_list_from_response(response)
        except:
            actions = []

    return actions


def judge_object_discovery(gt_objects, objects):

    hits = 0
    n_gt_objects = len(gt_objects)

    # prompter = get_prompter("InternVL3-8B", "gt_rules", 0)
    prompter = get_prompter("gpt-4o", "gt_rules", 0)

    prompt_path = "prompts/judge/judge_object_in_discovered.txt"

    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    for gt_obj in gt_objects:
        hit = 0
        # TODO: check if gt_obj is in objects via prompt
        prompt = prompt_template.replace("{target_object}", gt_obj).replace(
            "{detected_objects}", str(objects)
        )
        response = prompter.prompt_with_text(
            prompt, max_new_tokens=512, do_sample=False, overwrite_memory=False
        )
        print(response)
        try:
            hit = parse_match_from_response(response)
            if hit:
                hits += 1
        except:
            pass

    # prompter.remove_from_gpu()

    ratio = hits / n_gt_objects if n_gt_objects > 0 else 1

    return ratio


def judge_property_discovery(gt_properties, properties):

    hits = 0
    n_gt_properties = len(gt_properties)

    prompter = get_prompter("gpt-4o", "gt_rules", 0)

    prompt_path = "prompts/judge/judge_property_in_discovered.txt"

    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    for gt_prop in gt_properties:
        hit = 0
        prompt = prompt_template.replace("{target_property}", gt_prop).replace(
            "{detected_properties}", str(properties)
        )
        response = prompter.prompt_with_text(
            prompt, max_new_tokens=512, do_sample=False, overwrite_memory=False
        )
        print(response)
        try:
            hit = parse_match_from_response(response)
            if hit:
                hits += 1
        except:
            pass

    # prompter.remove_from_gpu()
    ratio = hits / n_gt_properties if n_gt_properties > 0 else 1

    return ratio


def judge_action_discovery(gt_actions, actions):

    hits = 0
    n_gt_actions = len(gt_actions)

    prompter = get_prompter("gpt-4o", "gt_rules", 0)

    prompt_path = "prompts/judge/judge_action_in_discovered.txt"

    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    for gt_act in gt_actions:
        hit = 0
        prompt = prompt_template.replace("{target_action}", gt_act).replace(
            "{detected_actions}", str(actions)
        )
        response = prompter.prompt_with_text(
            prompt, max_new_tokens=512, do_sample=False, overwrite_memory=False
        )
        print(response)
        try:
            hit = parse_match_from_response(response)
            if hit:
                hits += 1
        except:
            pass

    # prompter.remove_from_gpu()

    ratio = hits / n_gt_actions if n_gt_actions > 0 else 1

    return ratio


def eval_variable_discovery(args):

    log_path = "logs/variable_discovery/"
    os.makedirs(log_path, exist_ok=True)
    prompt_name = args.object_prompt.split("/")[-1].replace(".txt", "")
    log_path = os.path.join(
        log_path, f"eval_{args.dataset}_{args.model}_{prompt_name}.log"
    )

    # if log file exists, open text
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_text = f.read()
        if args.seed == 0:
            log_text = ""
    else:
        log_text = ""
    log_text += f"Dataset: {args.dataset}\n"
    log_text += f"Model: {args.model}\n"
    log_text += f"Seed: {args.seed}\n"

    if args.object_prompt == "combi":
        object_prompt = "prompts/discovery/objects.txt"
        property_prompt = "prompts/discovery/properties.txt"
        action_prompt = "prompts/discovery/actions.txt"
    else:
        object_prompt = args.object_prompt
        property_prompt = args.object_prompt.replace("objects", "properties")
        action_prompt = args.object_prompt.replace("objects", "actions")

    # load data
    data = load_data(args.dataset, max_imgs=args.max_imgs)

    # Create RTPT object
    rtpt = RTPT(
        name_initials="XX",
        experiment_name=f"VLP_{args.dataset}_{args.model}_{args.max_imgs}_{args.variable_distribution}_{args.seed}",
        max_iterations=n_tasks_per_dataset[args.dataset],
    )
    rtpt.start()

    # initialize prompter
    prompter = get_prompter(args.model, args.dataset, args.seed)

    hit_object_ratios = []
    hit_property_ratios = []
    hit_action_ratios = []

    # start loop over data
    for i, sample in tqdm(enumerate(data)):
        # if i >= 10:
        #     break

        print(f"Running sample {i}...")
        log_text += f"Running sample {i}...\n\n"

        hit_object_ratio = 0
        hit_property_ratio = 0
        hit_action_ratio = 0

        pos_imgs_paths, neg_imgs_paths, pos_test_imgs, neg_test_imgs, gt_rule = sample

        print(gt_rule)
        gt_objects = retrieve_objects_from_gt(gt_rule, args.dataset)
        print(f"GT objects: {gt_objects}")
        log_text += f"GT objects: {gt_objects}\n"

        # Start variable discovery
        if args.use_positive_examples_only:
            train_images = pos_imgs_paths
        else:
            train_images = pos_imgs_paths + neg_imgs_paths

        # discover objects
        # print(f"Discovering {args.n_objects} objects...")

        print("Using: ", prompter.model_name)
        objects = discover_objects(
            train_images,
            n_min_properties=args.n_objects,
            prompter=prompter,
            prompt_path=object_prompt,
        )
        # print(f"Discovered objects: {objects}")
        if len(objects) == 0:
            objects = ["empty"]

        log_text += f"Discovered objects: {objects}\n"

        # # judge object discovery
        hit_object_ratio = judge_object_discovery(gt_objects, objects)
        hit_object_ratios.append(hit_object_ratio)
        # hit_object_ratio = 0

        log_text += f"Hit Object Ratio: {hit_object_ratio}\n\n"

        if args.n_properties > 0:
            # discover properties
            print("Discovering properties...")
            properties = discover_properties(
                train_images,
                objects,
                n_min_properties=args.n_properties,
                prompter=prompter,
                prompt_path=property_prompt,
            )
            print(f"Discovered properties: {properties}")
            # properties = []
        else:
            properties = []

        gt_properties = retrieve_properties_from_gt(gt_rule, args.dataset)
        print(f"GT properties: {gt_properties}")
        log_text += f"GT properties: {gt_properties}\n"
        log_text += f"Discovered properties: {properties}\n"

        if len(gt_properties) > 0:
            # judge property discovery
            hit_property_ratio = judge_property_discovery(gt_properties, properties)
            hit_property_ratios.append(hit_property_ratio)
            log_text += f"Hit Property Ratio: {hit_property_ratio}\n\n"

        if args.n_actions > 0:
            # discover actions
            print("Discovering actions...")
            actions = discover_actions(
                train_images,
                objects,
                n_min_actions=args.n_actions,
                prompter=prompter,
                prompt_path=action_prompt,
            )
            print(f"Discovered actions: {actions}")
            # actions = []
        else:
            actions = []

        gt_actions = retrieve_actions_from_gt(gt_rule, args.dataset)
        print(f"GT actions: {gt_actions}")
        log_text += f"GT actions: {gt_actions}\n"

        log_text += f"Discovered actions: {actions}\n"

        if len(gt_actions) > 0:
            # judge action discovery
            hit_action_ratio = judge_action_discovery(gt_actions, actions)
            hit_action_ratios.append(hit_action_ratio)
            log_text += f"Hit Action Ratio: {hit_action_ratio}\n\n"

        objects = [obj for obj in objects if type(obj) == str]
        properties = [prop for prop in properties if type(prop) == str]
        actions = [act for act in actions if type(act) == str]

        print("\n")

        rtpt.step()
        # save log
        with open(log_path, "w") as f:
            f.write(log_text)

    # get mean ratios
    mean_hit_object_ratio = np.mean(hit_object_ratios)
    mean_hit_property_ratio = np.mean(hit_property_ratios)
    mean_hit_action_ratio = np.mean(hit_action_ratios)

    # add to log
    log_text += (
        f"----------------\nOBJECT SCORE: {mean_hit_object_ratio}\n-----------------\n"
    )
    log_text += f"----------------\nPROPERTY SCORE: {mean_hit_property_ratio}\n-----------------\n"
    log_text += (
        f"----------------\nACTION SCORE: {mean_hit_action_ratio}\n-----------------\n"
    )

    # save log
    with open(log_path, "w") as f:
        f.write(log_text)

    # remove prompter from gpu
    prompter.remove_from_gpu()

    return mean_hit_object_ratio, mean_hit_property_ratio, mean_hit_action_ratio


def variable_discovery(args):
    log_path = "logs/variable_discovery/"
    os.makedirs(log_path, exist_ok=True)
    prompt_name = args.object_prompt.split("/")[-1].replace(".txt", "")
    log_path = os.path.join(
        log_path, f"eval_{args.dataset}_{args.model}_{prompt_name}.log"
    )

    # if log file exists, open text
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_text = f.read()
        if args.seed == 0:
            log_text = ""
    else:
        log_text = ""
    log_text += f"Dataset: {args.dataset}\n"
    log_text += f"Model: {args.model}\n"
    log_text += f"Seed: {args.seed}\n"

    property_prompt = args.object_prompt.replace("objects", "properties")
    action_prompt = args.object_prompt.replace("objects", "actions")

    # load data
    data = load_data(args.dataset, max_imgs=args.max_imgs)

    # Create RTPT object
    rtpt = RTPT(
        name_initials="XX",
        experiment_name=f"VLP_{args.dataset}_{args.model}_{args.max_imgs}_{args.variable_distribution}_{args.seed}",
        max_iterations=n_tasks_per_dataset[args.dataset],
    )
    rtpt.start()

    # initialize prompter
    prompter = get_prompter(args.model, args.dataset, args.seed)

    # start loop over data
    for i, sample in tqdm(enumerate(data)):
        if i >= 10:
            break

        print(f"Running sample {i}...")
        log_text += f"Running sample {i}...\n\n"

        pos_imgs_paths, neg_imgs_paths, pos_test_imgs, neg_test_imgs, gt_rule = sample

        # Start variable discovery
        if args.use_positive_examples_only:
            train_images = pos_imgs_paths
        else:
            train_images = pos_imgs_paths + neg_imgs_paths

        # discover objects
        # print(f"Discovering {args.n_objects} objects...")

        print("Using: ", prompter.model_name)
        objects = discover_objects(
            train_images,
            n_min_properties=args.n_objects,
            prompter=prompter,
            prompt_path=args.object_prompt,
        )
        # print(f"Discovered objects: {objects}")
        if len(objects) == 0:
            objects = ["empty"]

        log_text += f"Discovered objects: {objects}\n"

        # discover properties
        print("Discovering properties...")
        properties = discover_properties(
            train_images,
            objects,
            n_min_properties=args.n_properties,
            prompter=prompter,
            prompt_path=property_prompt,
        )
        print(f"Discovered properties: {properties}")
        # properties = []

        log_text += f"Discovered properties: {properties}\n"

        if args.n_actions > 0:
            # discover actions
            print("Discovering actions...")
            actions = discover_actions(
                train_images,
                objects,
                n_min_actions=args.n_actions,
                prompter=prompter,
                prompt_path=action_prompt,
            )
            print(f"Discovered actions: {actions}")
            # actions = []

            log_text += f"Discovered actions: {actions}\n"
        else:
            actions = []

        objects = [obj for obj in objects if type(obj) == str]
        properties = [prop for prop in properties if type(prop) == str]
        actions = [act for act in actions if type(act) == str]

        print("\n")

        rtpt.step()
        # save log
        with open(log_path, "w") as f:
            f.write(log_text)

    # save log
    with open(log_path, "w") as f:
        f.write(log_text)

    # remove prompter from gpu
    prompter.remove_from_gpu()

    return 0, 0, 0


if __name__ == "__main__":
    args = parse_args()

    args.object_prompt = "combi"

    models = [
        "InternVL3-8B",
        "InternVL3-14B",
        "Qwen2.5-VL-7B-Instruct",
        "Kimi-VL-A3B-Instruct",
    ]
    # models = ["InternVL3-8B", "InternVL3-14B"]
    datasets = [
        "bongard-op",
        "bongard-hoi",
        "bongard-rwr",
        "cocologic",
        "CLEVR-Hans3-unconfounded",
    ]
    # datasets = ["CLEVR-Hans3-unconfounded"]
    # datasets = ["cocologic", "CLEVR-Hans3-unconfounded", "bongard-hoi", "bongard-op"]
    # datasets = ["cocologic"]

    df = pd.DataFrame()

    for dataset in datasets:

        for use_positive_examples_only in [False]:

            for model in models:

                for seed in [0, 1, 2]:
                    print(f"Evaluating dataset: {dataset}")

                    args.dataset = dataset
                    args.model = model
                    args.seed = seed

                    params_for_dataset = params_per_dataset[dataset]
                    args.n_objects = params_for_dataset["n_objects"]
                    args.n_properties = params_for_dataset["n_properties"]
                    # args.n_properties = 0
                    args.n_actions = params_for_dataset["n_actions"]
                    # args.n_actions = 0
                    args.max_program_depth = params_for_dataset["max_program_depth"]
                    args.max_imgs = params_for_dataset["max_imgs"]
                    args.use_positive_examples_only = use_positive_examples_only

                    (
                        mean_hit_object_ratio,
                        mean_hit_property_ratio,
                        mean_hit_action_ratio,
                    ) = eval_variable_discovery(args)

                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    "Dataset": [dataset],
                                    "Use Positive Examples Only": [
                                        use_positive_examples_only
                                    ],
                                    "Model": [model],
                                    "Seed": [seed],
                                    "n_objects": [args.n_objects],
                                    "n_properties": [args.n_properties],
                                    "n_actions": [args.n_actions],
                                    "Hit Object Ratio": [mean_hit_object_ratio],
                                    "Hit Property Ratio": [mean_hit_property_ratio],
                                    "Hit Action Ratio": [mean_hit_action_ratio],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )

    # average over seeds
    df = (
        df.groupby(
            [
                "Dataset",
                "Use Positive Examples Only",
                "Model",
                "n_objects",
                "n_properties",
                "n_actions",
            ]
        )
        .mean()
        .reset_index()
    )

    # currently just pos imgs
    print(df)

    # get name of object prompt
    object_prompt = args.object_prompt.split("/")[-1].replace(".txt", "")

    # save df to csv
    df.to_csv(
        f"results/variable_discovery/variable_discovery_results_{object_prompt}.csv",
        index=False,
    )
