import json
from eval_qualitative import plot_train_imgs
from eval import params_per_dataset
from utils.args import parse_args
from utils.dataset_utils import load_data


def main(args):

    dataset = args.dataset
    model = args.model
    seed = args.seed

    program_depth = 4 if dataset in ["bongard-op", "bongard-hoi"] else 6
    n_objects = params_per_dataset[dataset]["n_objects"]
    n_properties = params_per_dataset[dataset]["n_properties"]
    n_actions = params_per_dataset[dataset]["n_actions"]
    n_imgs = params_per_dataset[dataset]["max_imgs"]

    # results from baseline experiments
    # results/qualitative/CLEVR-Hans3-unconfounded/direct_results_Qwen3-VL-30B-A3B-Instruct_0_10.json
    baseline_path = (
        f"results/qualitative/{dataset}/direct_results_{model}_{seed}_{n_imgs}.json"
    )
    # load baseline results
    with open(baseline_path, "r") as f:
        baseline_results = json.load(f)

    # results from program experiments
    vlp_path = f"results/{dataset}/discovered_programs_{model}_10_{program_depth}_{n_objects}_{n_properties}_{n_actions}_0_{n_imgs}_naive_weighted_{seed}.json"

    # load vlp results
    with open(vlp_path, "r") as f:
        vlp_results = json.load(f)

    data = load_data(dataset, max_imgs=n_imgs)

    for i in range(len(data)):

        pos_train_img, neg_train_img, pos_test_img, neg_test_img, gt = data[i]

        accuracy_baseline = baseline_results[i]["accuracy"]
        rule_baseline = baseline_results[i]["rule"]

        vlp_result = vlp_results[str(i)][0]
        # turn string into tuple
        vlp_result_tuple = vlp_result.split(",")

        accuracy_vlp = float(vlp_result_tuple[3])
        rule_vlp = vlp_result_tuple[0]

        if accuracy_baseline <= 0.5 and accuracy_vlp >= 0.8:
            print(f"Example {i}:")
            print(f"GT Rule: {gt}")
            print(f"Baseline Acc: {accuracy_baseline}, Rule: {rule_baseline}")
            print(f"VLP Acc: {accuracy_vlp}, Rule: {rule_vlp}")
            print("-----")

        # plot train images
        plot_train_imgs(
            (pos_train_img, neg_train_img, pos_test_img, neg_test_img, gt),
            i,
            dataset,
            f"results/good_examples/good_example_{dataset}_{i}_{model}.pdf",
        )


if __name__ == "__main__":
    args = parse_args()

    args.model = "Qwen3-VL-30B-A3B-Instruct"
    args.dataset = "CLEVR-Hans3-unconfounded"
    args.seed = 0

    main(args)
