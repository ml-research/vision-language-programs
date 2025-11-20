import argparse
import os
import json

from main import add_variables_to_dsl
from method.DSL.dsl_with_img_repr import get_dsl
from method.dsl import DSL
from method.experiment_helper import __get_type_request
from method.type_system import *

from precollect_vlm_answers import (
    pre_collect_img_repr_final,
    pre_collect_vlm_answers,
    pre_collect_img_representations,
)
import re
import ast
import torch
from tqdm import tqdm
from rtpt import RTPT
from discover_properties import (
    discover_properties,
    discover_objects,
    discover_actions,
    discover_image_scenery,
    discover_object_conditions,
    variable_discovery,
)


from method.experiment_helper import (
    make_program_checker,
    make_program_checker_with_accuracy,
)
from method.run_experiment import run_algorithm

from eval import n_tasks_per_dataset
from utils.args import parse_args
from utils.dataset_utils import load_data
from utils.prompters import get_prompter
from utils.util import reserve_gpus


def parse_response(response, dict_key="rule"):

    code_block = re.search(r"```python(.*?)```", response, re.DOTALL)
    if code_block:
        code = code_block.group(1).strip()

        # Step 2: Extract the dict part (after '=')
        dict_match = re.search(r"=\s*(\{.*\})", code, re.DOTALL)
        if dict_match:
            dict_str = dict_match.group(1)

            # Step 3: Safely parse the string into a Python dict
            parsed_dict = ast.literal_eval(dict_str)
            print(parsed_dict)

            try:
                rule = parsed_dict[dict_key]
            except:
                rule = parsed_dict

            return rule
        else:
            raise ValueError("No dictionary found after '=' in the code block.")
    else:
        raise ValueError("No Python code block found in the response.")


def evaluate_program(program, pos_img, neg_img, dsl):
    """Evaluate the program on a single positive and negative image."""

    pos_result = program.eval_naive(dsl, [pos_img])
    neg_result = program.eval_naive(dsl, [neg_img])

    pos_acc = 1 if pos_result else 0
    neg_acc = 0 if neg_result else 1
    accuracy = (pos_acc + neg_acc) / 2

    # print(f"Positive image result: {pos_result}, accuracy: {pos_acc}")
    return accuracy, [pos_acc, neg_acc]


def evaluate_programs_on_images(program, pos_imgs, neg_imgs, dsl):
    """Evaluate a list of programs on a list of positive and negative images."""
    accuracies = []
    all_single_accs = []
    for pos_img, neg_img in zip(pos_imgs, neg_imgs):
        accuracy, single_accs = evaluate_program(program, pos_img, neg_img, dsl)
        accuracies.append(accuracy)
        all_single_accs += single_accs
    avg_accuracy = sum(accuracies) / len(accuracies)

    return avg_accuracy, all_single_accs


def evaluate_programs(programs, pos_img, neg_img, dsl):
    """Evaluate a list of programs on positive and negative images."""
    results = []
    if not isinstance(pos_img, list):
        pos_img = [pos_img]
    if not isinstance(neg_img, list):
        neg_img = [neg_img]
    for program in programs:
        accuracy, single_accs = evaluate_programs_on_images(
            program, pos_img, neg_img, dsl
        )
        results.append((accuracy, program, single_accs))
    return results


def __get_type(el, fallback=None):
    if isinstance(el, str):
        if ".png" in el or ".jpg" in el or ".jpeg" in el:
            return IMG
        else:
            return PROPERTY  # TODO: okay like this?
    if isinstance(el, bool):
        return BOOL
    elif isinstance(el, int):
        return INT
    elif isinstance(el, list):
        if len(el) > 0:
            return List(__get_type(el[0]))
        else:
            return __get_type(fallback[0], fallback[1:])
    elif isinstance(el, tuple):
        assert el[-1] == None
        return __get_type(el[0], el[1:-1])
    assert False, f"Unknown type for:{el}"


def __get_type_request(examples):
    input, output = examples[0]
    return Arrow(
        __get_type(input[0], [i[0] for i, o in examples[1:]]),
        __get_type(output, [o for i, o in examples[1:]]),
    )


def main(args):

    # load data
    data = load_data(args.dataset, max_imgs=args.max_imgs)

    # Create RTPT object
    rtpt = RTPT(
        name_initials="XX",
        experiment_name=f"B_with_structure_{args.dataset}_{args.model}_{args.max_imgs}_{args.seed}",
        max_iterations=n_tasks_per_dataset[args.dataset],
    )
    rtpt.start()
    reserve_gpus()

    print("No sampling: ", args.no_sampling)

    # initialize prompter
    prompter = get_prompter(
        args.model,
        args.dataset,
        args.seed,
        reasoning=False,
        sampling=not args.no_sampling,
    )

    img_obj_representations = {}
    discovered_programs = {}

    all_results = []

    # start loop over data
    for i, sample in tqdm(enumerate(data)):

        result = {}

        print(f"Running sample {i}...")

        pos_imgs_paths, neg_imgs_paths, pos_test_imgs, neg_test_imgs, gt = sample

        print(f"Ground truth: {gt}")

        # Set up tasks for program synthesis
        examples = []
        for img_path in pos_imgs_paths:
            examples.append(([img_path], True))
        for img_path in neg_imgs_paths:
            examples.append(([img_path], False))

        # Limit number of images for discovery to avoid cuda memory issues
        if len(pos_imgs_paths) > 10:
            train_images = pos_imgs_paths[:10] + neg_imgs_paths[:10]
        else:
            train_images = pos_imgs_paths + neg_imgs_paths

        variables = variable_discovery(prompter, train_images, args)

        examples = []
        for img_path in pos_imgs_paths:
            examples.append(([img_path], True))
        for img_path in neg_imgs_paths:
            examples.append(([img_path], False))

        if len(examples) != 12:
            print(
                f"Warning: Expected 12 examples, but got {len(examples)}. This might lead to unexpected results."
            )

        semantics, primitive_types = get_dsl(
            args.model, args.dataset, prompter, variables
        )

        problem_semantics = semantics.copy()
        problem_primitive_types = primitive_types.copy()

        add_variables_to_dsl(problem_semantics, problem_primitive_types, variables)

        # create dsl
        dsl = DSL(problem_semantics, problem_primitive_types)

        (
            img_object_representations,
            img_action_representations,
            _,
            variables,
        ) = pre_collect_img_representations(
            examples,
            variables,
            problem_semantics,
            problem_primitive_types,
        )

        print("Pre-collecting VLM answers done.")

        img_obj_representations[i] = img_object_representations

        ### START VLM PROMPTING ###

        prompt_path = "prompts/structure_baseline.txt"
        with open(prompt_path, "r") as f:
            prompt_template = f.read()

        structured_image_representations = ""

        for idx, example in enumerate(examples):
            img_path, label = example
            img_path = img_path[0]
            objects_from_img = semantics["get_objects"]
            actions_from_img = semantics.get("get_actions", lambda x: [[]])
            obj_repr = objects_from_img(img_path)
            action_repr = actions_from_img(img_path)
            structured_image_representations += f"Image: {idx+1}\n"
            structured_image_representations += f"Objects: {obj_repr}\n"
            structured_image_representations += f"Actions: {action_repr}\n"
            structured_image_representations += (
                f"Label: {'Positive' if label else 'Negative'}\n\n"
            )

        prompt = prompt_template.replace(
            "{representations}", structured_image_representations
        )

        print(prompt)

        # TODO: prompt the model
        print("Prompting the model with structured representations...")
        response = prompter.prompt_with_text(
            prompt_text=prompt,
            max_new_tokens=16384,
        )
        print("Response from the model:")
        print(response)

        # parse the python rule from the response
        # Here we assume the response is a valid Python expression
        try:
            parsed_response = parse_response(response)
        except Exception as e:
            print(f"Error parsing response: {e}")
            parsed_response = "Error parsing response"

        test_prompt_path = "prompts/structure_baseline_test.txt"
        with open(test_prompt_path, "r") as f:
            test_prompt_template = f.read()

        # test_prompt = f"Given the rule '{parsed_response}', determine if the image follows the rule or not. Answer with 'Yes' or 'No', nothing else."

        get_objects = semantics["get_objects"]
        get_actions = semantics.get("get_actions", lambda x: [[]])

        pos_test_responses = []
        for pos_test_img in pos_test_imgs:
            object_repr = get_objects(pos_test_img)
            action_repr = get_actions(pos_test_img)

            structured_image_repr = ""
            structured_image_repr += f"Objects: {object_repr}\n"
            structured_image_repr += f"Actions: {action_repr}\n"

            cur_test_prompt = test_prompt_template.replace(
                "{rule}", str(parsed_response)
            ).replace("{representation}", structured_image_repr)

            print(f"Test prompt: {cur_test_prompt}")
            test_response_1 = prompter.prompt_with_text(
                prompt_text=cur_test_prompt,
            )
            pos_test_responses.append(test_response_1)

        neg_test_responses = []
        for neg_test_img in neg_test_imgs:

            object_repr = get_objects(neg_test_img)
            action_repr = get_actions(neg_test_img)
            structured_image_repr = ""
            structured_image_repr += f"Objects: {object_repr}\n"
            structured_image_repr += f"Actions: {action_repr}\n"
            cur_test_prompt = test_prompt_template.replace(
                "{rule}", str(parsed_response)
            ).replace("{representation}", structured_image_repr)
            print(f"Test prompt: {cur_test_prompt}")
            test_response_2 = prompter.prompt_with_text(
                prompt_text=cur_test_prompt,
            )
            neg_test_responses.append(test_response_2)

        print(f"Response for test images: ")
        print(f"Positive: {pos_test_responses}")
        print(f"Negative: {neg_test_responses}")

        pos_correct = 0
        neg_correct = 0

        for pos_test_response in pos_test_responses:
            if (
                "yes" in pos_test_response.lower()
                and "no" not in pos_test_response.lower()
            ):
                pos_correct += 1

        for neg_test_response in neg_test_responses:
            if (
                "no" in neg_test_response.lower()
                and "yes" not in neg_test_response.lower()
            ):
                neg_correct += 1

        print(
            f"Correct answers: {pos_correct + neg_correct} out of {len(pos_test_imgs + neg_test_imgs)}"
        )

        # balanced acc
        balanced_acc = (
            pos_correct / len(pos_test_imgs) + neg_correct / len(neg_test_imgs)
        ) / 2

        print(f"Accuracy: {balanced_acc}")

        result["rule"] = parsed_response
        result["pos_test_responses"] = pos_test_responses
        result["neg_test_responses"] = neg_test_responses
        result["accuracy"] = balanced_acc

        all_results.append(result)

        # save results in txt file
        results_file = f"results/baseline_with_structure/{args.dataset}_{args.model}_{args.max_imgs}_{args.seed}.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=4)

        rtpt.step(subtitle=f"it={i}")

    # save results in txt file
    results_file = f"results/baseline_with_structure/{args.dataset}_{args.model}_{args.max_imgs}_{args.seed}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    args = parse_args()

    main(args)
