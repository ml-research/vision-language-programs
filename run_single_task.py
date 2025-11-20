import argparse
import os
import json
from main import add_variables_to_dsl
from utils.args import parse_args
from utils.dataset_utils import load_data
from method.DSL.dsl_with_img_repr import get_dsl
from method.dsl import DSL
from main import __get_type_request
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


def run_single_task(sample, args):

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
    semantics, primitive_types = get_dsl(args.model, args.dataset, prompter, variables)

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
        raise ValueError(f"Unknown variable distribution: {args.variable_distribution}")

    is_correct_program = make_program_checker_with_accuracy(dsl, examples)
    algo_index = 0

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

    final_program = programs[0]

    return final_program, dsl
