import argparse
import os
import json
from utils.args import parse_args
from utils.dataset_utils import load_data
from method.DSL.dsl_with_img_repr import get_dsl
from method.dsl import DSL
from method.experiment_helper import __get_type_request
from method.type_system import *

from precollect_vlm_answers import (
    pre_collect_img_repr_final,
    pre_collect_vlm_answers,
    pre_collect_img_representations,
)
import torch
from tqdm import tqdm
from rtpt import RTPT
from discover_properties import (
    discover_properties,
    discover_objects,
    discover_actions,
    variable_discovery,
)

from method.experiment_helper import make_program_checker_with_accuracy
from method.run_experiment import run_algorithm
from utils.prompters import get_prompter
from utils.util import reserve_gpus
from eval import n_tasks_per_dataset


def evaluate_program(program, pos_img, neg_img, dsl):
    """Evaluate the program on a single positive and negative image."""

    pos_result = program.eval_naive(dsl, [pos_img])
    neg_result = program.eval_naive(dsl, [neg_img])

    pos_acc = 1 if pos_result else 0
    neg_acc = 0 if neg_result else 1
    accuracy = (pos_acc + neg_acc) / 2

    return accuracy, [pos_acc, neg_acc]


def evaluate_program_on_image(program, img, expected_output, dsl):
    """Evaluate a program on a single image and expected output."""
    result = program.eval_naive(dsl, [img])
    acc = 1 if result == expected_output else 0
    return acc


def evaluate_programs_on_images(program, pos_imgs, neg_imgs, dsl):
    """Evaluate a list of programs on a list of positive and negative images."""
    pos_accuracies = []
    neg_accuracies = []
    all_single_accs = []

    for pos_img in pos_imgs:
        accuracy = evaluate_program_on_image(program, pos_img, True, dsl)
        pos_accuracies.append(accuracy)
        all_single_accs.append(accuracy)

    for neg_img in neg_imgs:
        accuracy = evaluate_program_on_image(program, neg_img, False, dsl)
        neg_accuracies.append(accuracy)
        all_single_accs.append(accuracy)

    avg_pos_accuracy = (
        sum(pos_accuracies) / len(pos_accuracies) if pos_accuracies else 0
    )
    avg_neg_accuracy = (
        sum(neg_accuracies) / len(neg_accuracies) if neg_accuracies else 0
    )

    balanced_accuracy = (avg_pos_accuracy + avg_neg_accuracy) / 2

    return balanced_accuracy, all_single_accs


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
            return PROPERTY
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


def add_variables_to_dsl(problem_semantics, problem_primitive_types, variables):

    objects = variables.get("objects", [])
    properties = variables.get("properties", [])
    actions = variables.get("actions", [])

    # add objects to semantics and primitive_types
    for o in objects:
        if o in problem_semantics:
            continue
        # add to semantics
        problem_semantics[o] = o
        # add to primitive_types
        problem_primitive_types[o] = OBJECT

    # add properties to semantics and primitive_types
    for p in properties:
        # make sure property is not already in semantics
        if p in problem_semantics:
            continue
        problem_semantics[p] = p
        problem_primitive_types[p] = PROPERTY

    # add actions to semantics and primitive_types
    for a in actions:
        if a in problem_semantics:
            continue
        problem_semantics[a] = a
        problem_primitive_types[a] = ACTION

    return problem_semantics, problem_primitive_types


def main(args):

    # load data
    data = load_data(args.dataset, max_imgs=args.max_imgs)

    # Create RTPT object
    rtpt = RTPT(
        name_initials="XX",
        experiment_name=f"VLP_{args.dataset}_{args.model}_{args.max_imgs}_{args.variable_distribution}_{args.seed}",
        max_iterations=n_tasks_per_dataset[args.dataset],
    )
    rtpt.start()
    reserve_gpus()

    used_tokens = 0

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

    # start loop over data
    for i, sample in tqdm(enumerate(data)):

        print(f"Running sample {i}...")

        pos_imgs_paths, neg_imgs_paths, pos_test_imgs, neg_test_imgs, gt = sample

        print(f"Ground truth: {gt}")

        # Set up tasks for program synthesis
        examples = []
        for img_path in pos_imgs_paths:
            examples.append(([img_path], True))
        for img_path in neg_imgs_paths:
            examples.append(([img_path], False))

        # Start variable discovery
        # Limit number of images for discovery to avoid cuda memory issues
        if len(pos_imgs_paths) > 10:
            train_images = pos_imgs_paths[:10] + neg_imgs_paths[:10]
        else:
            train_images = pos_imgs_paths + neg_imgs_paths

        # report number of pos and neg images used for discovery
        print(
            f"Using {len(pos_imgs_paths)} positive and {len(neg_imgs_paths)} negative images for variable discovery."
        )

        variables = variable_discovery(prompter, train_images, args)

        # get dsl
        semantics, primitive_types = get_dsl(
            args.model, args.dataset, prompter, variables
        )

        # INTERACTION functions
        if args.xil_add_functions:
            _semantics, _primitive_types = get_dsl(
                args.model, "xil-add-functions", prompter, variables
            )
            semantics.update(_semantics)
            primitive_types.update(_primitive_types)

        problem_semantics = semantics.copy()
        problem_primitive_types = primitive_types.copy()

        add_variables_to_dsl(problem_semantics, problem_primitive_types, variables)

        # create dsl
        dsl = DSL(problem_semantics, problem_primitive_types)

        print("Start pre-collecting VLM answers...")

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

        # set up fresh dsl
        problem_semantics = semantics.copy()
        problem_primitive_types = primitive_types.copy()

        add_variables_to_dsl(problem_semantics, problem_primitive_types, variables)

        # update dsl with new semantics and primitive_types
        dsl = DSL(problem_semantics, problem_primitive_types)

        # final collection of image representations
        pre_collect_img_repr_final(examples, problem_semantics)

        # collect tokens
        used_tokens += prompter.get_produced_tokens()
        prompter.reset_produced_tokens()

        type_request = __get_type_request(examples)
        print(type_request)

        # create grammar from dsl (with program type, depth etc.)
        cfg = dsl.DSL_to_CFG(
            type_request,
            max_program_depth=args.max_program_depth,  # 4 # 6
            min_variable_depth=1,  # 1
            upper_bound_type_size=10,
            n_gram=2,
        )

        img_representations = {
            "objects": img_object_representations,
            "properties": img_object_representations,
            "actions": img_action_representations,
        }

        if args.variable_distribution == "uniform":
            print("Using uniform variable distribution")
            pcfg = cfg.CFG_to_Uniform_PCFG()

        elif args.variable_distribution == "naive_frequency":
            print("Using naive frequency for variable distribution")
            pcfg = cfg.CFG_to_PCFG_with_naive_frequency_ratio(
                img_representations, variables
            )
        elif args.variable_distribution == "naive_weighted":
            print("Using naive weighted for variable distribution")
            pcfg = cfg.CFG_to_PCFG_with_naive_weighted(img_representations, variables)
        elif args.variable_distribution == "positive_ratio":
            print("Using positive ratio for variable distribution")
            pcfg = cfg.CFG_to_PCFG_with_positives_only(img_representations, variables)
        else:
            raise ValueError(
                f"Unknown variable distribution: {args.variable_distribution}"
            )

        is_correct_program = make_program_checker_with_accuracy(dsl, examples)
        algo_index = 0

        # Print DSL info
        # print("DSL Semantics:")
        # for k, v in dsl.semantics.items():
        #     print(f"{k}: {v}")

        # Start search
        results = run_algorithm(
            is_correct_program,
            pcfg,
            algo_index,
            timeout=args.search_timeout,
            n_candidates=10,
        )

        programs = results[0]
        # sort programs by accuracy and probability
        programs = sorted(programs, key=lambda x: (x[0], x[2]), reverse=True)

        for acc, program, prob in programs:
            print(
                f"Accuracy: {acc:.2f} \t Program: {program} \t Probability: {prob:.2f}"
            )

        # for counting tokens, only consider first program
        program_eval_results_first = evaluate_programs(
            [[p[1] for p in programs][0]],
            pos_imgs_paths,
            neg_imgs_paths,
            dsl,
        )
        # add tokens from evaluation
        used_tokens += prompter.get_produced_tokens()
        prompter.reset_produced_tokens()

        program_eval_results = evaluate_programs(
            [p[1] for p in programs],
            pos_test_imgs,
            neg_test_imgs,
            dsl,
        )

        # get best program
        best_program = programs[0]

        final_results = [
            (
                program,
                round(prob, 8),
                round(train_acc, 2),
                round(test_acc, 2),
                single_test_accs,
            )
            for (train_acc, program, prob), (test_acc, _, single_test_accs) in zip(
                programs, program_eval_results
            )
        ]

        # discovered_programs[i] = [str(program) for program in programs]
        discovered_programs[i] = [str(final_result) for final_result in final_results]

        # create directory if it does not exist
        top_folder = f"results/{args.dataset}"

        if args.xil_remove_confounders and args.xil_add_functions:
            top_folder += "/remove_confounders_and_add_functions"

        elif args.xil_remove_confounders:
            top_folder += "/remove_confounders"

        elif args.xil_add_functions:
            top_folder += "/add_functions"

        if not os.path.exists(top_folder):
            os.makedirs(top_folder)

        if args.no_sampling:
            top_folder += "/no_sampling"
            if not os.path.exists(top_folder):
                os.makedirs(top_folder)

        # save program to file
        with open(
            f"{top_folder}/discovered_programs_{args.model}_{args.search_timeout}_{args.max_program_depth}_{args.n_objects}_{args.n_properties}_{args.n_actions}_{args.max_imgs}_{args.variable_distribution}_{args.seed}.json",
            "w",
        ) as f:
            json.dump(discovered_programs, f, indent=4)

        print("SAVE TO: ", f.name)

        # save img_obj_representations to file
        with open(
            f"{top_folder}/img_obj_representations_{args.model}_{args.n_objects}_{args.n_properties}_{args.n_actions}_{args.max_imgs}_{args.seed}.json",
            "w",
        ) as f:
            try:
                json.dump(img_obj_representations, f, indent=4)
            except Exception as e:
                print(img_obj_representations)
                print(f"Failed to save img_obj_representations: {e}")

        rtpt.step(subtitle=f"it={i}")

    print(f"Total used tokens: {used_tokens}")
    # add tokens to result file
    with open(
        f"{top_folder}/discovered_programs_{args.model}_{args.search_timeout}_{args.max_program_depth}_{args.n_objects}_{args.n_properties}_{args.n_actions}_{args.max_imgs}_{args.variable_distribution}_{args.seed}_used_tokens.txt",
        "w",
    ) as f:
        f.write(f"Total used tokens: {used_tokens}\n")


if __name__ == "__main__":
    args = parse_args()

    main(args)
