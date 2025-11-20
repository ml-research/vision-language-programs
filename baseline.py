import argparse
import os
import json
import sys
import re
import ast

import torch
from tqdm import tqdm
from rtpt import RTPT
from utils.dataset_utils import load_data, load_dataset

from models.internvl.main import InternVLPrompter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
from PIL import Image
import requests
from io import BytesIO
from utils.prompters import get_prompter


def plot_train_imgs(data_sample, id):
    pos_imgs_paths, neg_imgs_paths, test_imgs = data_sample

    image_paths = pos_imgs_paths + [test_imgs[0]] + neg_imgs_paths + [test_imgs[1]]

    plot_images(image_paths, id)


def plot_images(image_paths, id):
    fig, axs = plt.subplots(4, 4, figsize=(10, 15))
    # axs = axs.flatten()

    for i, path in enumerate(image_paths):
        if i <= 6:
            x = i % 2
            y = i // 2
        else:
            x = i % 2 + 2
            y = (i - 7) // 2

        # Open the image
        img = Image.open(path)
        axs[y, x].imshow(img)
        axs[y, x].set_title(f"Image {i + 1}")

        # Open the image
        img = Image.open(path)
        axs[y, x].imshow(img)
        axs[y, x].set_title(f"Image {i + 1}")

    # remove axes
    for ax in axs.flat:
        ax.axis("off")

    plt.tight_layout()
    # save the figure
    plt.savefig(f"results/qualitative/bongard_op_train_imgs_bp_{id}.png")
    plt.show()


def parse_response(response, dict_key="rule"):

    # Step 1: Try to extract Python code block (optional)
    code_block = re.search(r"```python(.*?)```", response, re.DOTALL)
    code = code_block.group(1).strip() if code_block else response.strip()

    # Step 2: Find a dictionary assignment pattern like "x = {...}"
    dict_match = re.search(r"=\s*(\{.*\})", code, re.DOTALL)
    if not dict_match:
        raise ValueError("No dictionary found after '=' in the response.")

    dict_str = dict_match.group(1)

    # Step 3: Safely parse the string into a Python dict
    try:
        parsed_dict = ast.literal_eval(dict_str)
    except Exception as e:
        raise ValueError(f"Failed to parse dictionary: {e}")

    # Step 4: Return the desired key or whole dict
    if dict_key in parsed_dict:
        return parsed_dict[dict_key]
    return parsed_dict


def eval(
    data_sample,
    bp_id,
    prompter,
    xil=False,
    max_imgs=None,
    think=False,
):

    result = {}

    pos_imgs = data_sample[0]
    neg_imgs = data_sample[1]

    train_imgs = pos_imgs + neg_imgs
    print("NUMBER OF TRAINING IMAGES:", len(train_imgs))

    prompt_path = "prompts/baseline_prompt.txt"

    prompt = open(prompt_path, "r").read()

    n_imgs = len(train_imgs)

    prompt = prompt.replace("{n}", str(n_imgs))
    prompt = prompt.replace("{m}", str(len(pos_imgs)))
    prompt = prompt.replace("{o}", str(len(neg_imgs)))

    pos_test_imgs = data_sample[2]
    neg_test_imgs = data_sample[3]
    test_imgs = pos_test_imgs + neg_test_imgs

    # assert len(neg_test_imgs) == 10 or len(neg_test_imgs) == 6

    all_output_tokens = 0

    print("Think: ", think)

    # prompt the model with the training images
    if think:
        print("Using thinking steps...")
        raw_response = prompter.prompt_with_images(
            prompt_text=prompt,
            paths=train_imgs,
            url=False,
            max_new_tokens=32768,
            use_memory=True,
            thinking=think,
            overwrite_memory=False,  # TODO
        )
    else:
        raw_response = prompter.prompt_with_images(
            prompt_text=prompt,
            paths=train_imgs,
            url=False,
            max_new_tokens=5000,
            use_memory=True,
            overwrite_memory=False,
        )

    print(f"Response for training images: {raw_response}")
    print(f"Output TOKEN: {all_output_tokens}")

    # parse the python rule from the response
    try:
        response = parse_response(raw_response, dict_key="rule")
        print(f"Parsed rule: {response}")
    except Exception as e:
        print(f"Error parsing response: {e}")
        response = "Error parsing response"

    # use rule to predict the test images
    test_prompt = f"Given the rule '{response}', determine if the image follows the rule or not. Answer with 'Yes' or 'No', nothing else."
    # test_prompt = f"Given the rule \"{response}\", determine if the following image follows the rule or not. Please answer with 'Yes' or 'No'."
    print(f"Test prompt: {test_prompt}")
    pos_test_responses = []
    for pos_test_img in pos_test_imgs:
        if think:
            test_response_1 = prompter.prompt_with_images(
                prompt_text=test_prompt,
                paths=[pos_test_img],
                url=False,
                max_new_tokens=32768,
                use_memory=True,
                thinking=think,
            )
        else:
            test_response_1 = prompter.prompt_with_images(
                prompt_text=test_prompt, paths=[pos_test_img], url=False
            )
        pos_test_responses.append(test_response_1)

    neg_test_responses = []
    for neg_test_img in neg_test_imgs:
        if think:
            test_response_2 = prompter.prompt_with_images(
                prompt_text=test_prompt,
                paths=[neg_test_img],
                url=False,
                max_new_tokens=32768,
                use_memory=True,
                thinking=think,
            )
        else:
            test_response_2 = prompter.prompt_with_images(
                prompt_text=test_prompt, paths=[neg_test_img], url=False
            )

        neg_test_responses.append(test_response_2)

    print(f"Response for test images: ")
    print(f"Positive: {pos_test_responses}")
    print(f"Negative: {neg_test_responses}")

    pos_correct_answers = 0
    neg_correct_answers = 0

    for pos_test_response in pos_test_responses:
        if "yes" in pos_test_response.lower() and "no" not in pos_test_response.lower():
            pos_correct_answers += 1

    for neg_test_response in neg_test_responses:
        if "no" in neg_test_response.lower() and "yes" not in neg_test_response.lower():
            neg_correct_answers += 1

    print(
        f"Correct answers: {pos_correct_answers+neg_correct_answers} out of {len(test_imgs)}"
    )

    balanced_acc = (
        pos_correct_answers / len(pos_test_imgs)
        + neg_correct_answers / len(neg_test_imgs)
    ) / 2

    result["full_response"] = raw_response
    result["rule"] = response
    result["pos_test_responses"] = pos_test_responses
    result["neg_test_responses"] = neg_test_responses
    result["accuracy"] = balanced_acc
    result["output_tokens"] = all_output_tokens

    return balanced_acc, result


def main(args):

    data = load_data(args.dataset, max_imgs=args.max_imgs)

    prompter = get_prompter(
        args.model,
        args.dataset,
        args.seed,
        sampling=not args.no_sampling,
        reasoning=args.think,
    )

    # Create RTPT object
    rtpt = RTPT(
        name_initials="XX",
        experiment_name=f"Baseline_{args.dataset}_{args.model}_{args.seed}_{args.max_imgs}",
        max_iterations=len(data),
    )
    rtpt.start()

    all_accs = []
    results = []

    for bp_id, data_sample in enumerate(tqdm(data, desc="Evaluating Problems")):

        acc, result = eval(
            data[bp_id],
            bp_id,
            prompter,
            xil=args.xil,
            max_imgs=args.max_imgs,
            think=args.think,
        )

        all_accs.append(acc)
        results.append(result)

        # save results to a json file
        if args.xil:
            results_file = f"results/qualitative/{args.dataset}/xil/direct_results_{args.model}_{args.seed}_{args.max_imgs}.json"
        elif args.think:
            results_file = f"results/qualitative/{args.dataset}/direct_results_{args.model}_{args.seed}_{args.max_imgs}_think.json"
            if args.no_sampling:
                results_file = f"results/qualitative/{args.dataset}/no_sampling/direct_results_{args.model}_{args.seed}_{args.max_imgs}_think.json"
        elif args.no_sampling:
            results_file = f"results/qualitative/{args.dataset}/no_sampling/direct_results_{args.model}_{args.seed}_{args.max_imgs}.json"
        else:
            results_file = f"results/qualitative/{args.dataset}/direct_results_{args.model}_{args.seed}_{args.max_imgs}.json"

        # if file folder does not exists, create it
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        rtpt.step(subtitle=f"it={bp_id}")

    print(f"Average accuracy: {sum(all_accs) / len(all_accs)}")
    # get std
    print(f"Standard deviation: {torch.std(torch.tensor(all_accs)).item()}")

    # plot times 100 and latex format
    all_accs = [x * 100 for x in all_accs]
    print(
        f"Mean accuracy: {sum(all_accs) / len(all_accs):.2f} \\pm {torch.std(torch.tensor(all_accs)).item():.2f}"
    )

    # All tokens used:
    tokens = prompter.get_produced_tokens()
    print(f"Total tokens used: {tokens}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate Bongard Problems")
    parser.add_argument(
        "--model",
        type=str,
        default="InternVL3-8B",
        help="Name of the model to use for evaluation",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CLEVR-Hans3-unconfounded",
        help="Dataset to use for evaluation",
    )
    parser.add_argument(
        "--xil",
        action="store_true",
        help="Use XIL for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for initialization",
    )

    parser.add_argument(
        "--limit_imgs",
        action="store_true",
        help="Use limited amount of imgs",
    )

    parser.add_argument(
        "--max_imgs",
        type=int,
        default=6,
        help="Max number of images to use for training",
    )

    parser.add_argument(
        "--think",
        action="store_true",
        help="Use thinking steps (only for Ovis)",
    )

    parser.add_argument(
        "--no_sampling",
        action="store_true",
    )

    args = parser.parse_args()
    # args.xil = True

    print(f"\nSTARTING EVALUATION WITH MODEL {args.model} ON DATASET {args.dataset}\n")

    torch.cuda.empty_cache()

    main(args)
