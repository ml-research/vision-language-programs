import argparse


def parse_args(jupyter=False):
    parser = argparse.ArgumentParser(description="Run the program.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="bongard-op",
        choices=[
            "bongard-op",
            "bongard-hoi",
            "bongard-hoi-max-img",
            "bongard-rwr",
            "cocologic",
            "cocologic-max-img",
            "CLEVR-Hans3-unconfounded",
        ],
        help="Dataset to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="InternVL3-8B",
        help="Model to use (InternVL3, Qwen, etc.)",
    )

    parser.add_argument(
        "--search_timeout",
        type=int,
        default=10,
        help="Timeout for the search algorithm in seconds",
    )

    parser.add_argument(
        "--n_objects",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--n_properties",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--n_actions",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max_program_depth",
        type=int,
        default=4,
        help="Maximum depth of the program",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for initialization",
    )

    parser.add_argument(
        "--max_imgs",
        type=int,
        default=12,
        help="Maximum number of images to use from the dataset (default: 12)",
    )

    parser.add_argument(
        "--variable_distribution",
        type=str,
        default="naive_weighted",
        choices=["uniform", "naive_weighted"],
        help="Distribution to use for variable sampling",
    )

    parser.add_argument(
        "--use_positive_examples_only",
        type=bool,
        default=False,
        help="Whether to use positive examples only",
    )

    parser.add_argument(
        "--object_prompt",
        type=str,
        default="prompts/discovery/objects_questions.txt",
        help="Prompt to use for object discovery",
    )

    # flag without argument
    parser.add_argument(
        "--xil_remove_confounders",
        action="store_true",
    )

    parser.add_argument(
        "--xil_add_functions",
        action="store_true",
    )

    parser.add_argument(
        "--xil_add_properties",
        action="store_true",
    )

    parser.add_argument(
        "--no_sampling",
        action="store_true",
        help="Whether to use sampling for VLMs",
    )

    if jupyter:
        return parser.parse_args([])
    else:
        return parser.parse_args()
