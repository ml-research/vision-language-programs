import json
from ast import literal_eval as make_tuple
from httpx import __name
import pandas as pd
import numpy as np


n_tasks_per_dataset = {
    "bongard-op": 200,
    "bongard-hoi": 166,
    "bongard-hoi-max-img": 67,
    "cocologic": 10,
    "cocologic-max-img": 8,
    "CLEVR-Hans3-unconfounded": 3,
    "bongard-rwr": 60,
}

params_per_dataset = {
    "bongard-op": {
        "n_objects": 10,
        "n_properties": 10,
        "n_actions": 3,
        "max_program_depth": 4,
        "max_imgs": 6,
    },
    "bongard-hoi": {
        "n_objects": 10,
        "n_properties": 5,
        "n_actions": 10,
        "n_image_scenery": 0,
        "max_program_depth": 4,
        "max_imgs": 6,
    },
    "bongard-hoi-max-img": {
        "n_objects": 10,
        "n_properties": 5,
        "n_actions": 10,
        "n_image_scenery": 0,
        "max_program_depth": 4,
        "max_imgs": 10,
    },
    "bongard-rwr": {
        "n_objects": 10,
        "n_properties": 10,
        "n_actions": 5,
        "n_image_scenery": 0,
        "max_program_depth": 6,
        "max_imgs": 6,
    },
    "cocologic": {
        "n_objects": 10,
        "n_properties": 10,
        "n_actions": 3,
        "n_image_scenery": 0,
        "max_program_depth": 6,
        "max_imgs": 10,
    },
    "cocologic-max-img": {
        "n_objects": 10,
        "n_properties": 10,
        "n_actions": 3,
        "n_image_scenery": 0,
        "max_program_depth": 6,
        "max_imgs": 30,
    },
    "CLEVR-Hans3-unconfounded": {
        "n_objects": 10,
        "n_properties": 10,
        "n_actions": 0,
        "n_image_scenery": 0,
        "max_program_depth": 6,
        "max_imgs": 10,
    },
}


def eval(file_path, n_tasks):

    # load json from path
    with open(file_path, "r") as f:
        results = json.load(f)

    train_accuracies = []
    test_accuracies = []
    all_tokens = 0

    cur_n_tasks = 0
    for i in range(n_tasks):
        key = str(i)

        if type(results) is dict:
            if key in results.keys():

                result = results[key]
                accs = []
                r = result[0]

                # turn line into tuple
                result_tuple = r.split(",")

                if len(result_tuple) >= 4:
                    result_tuple = result_tuple[:4]
                    p, prob, train_acc, test_acc = result_tuple
                    prob = float(prob.strip())
                    train_acc = float(train_acc.strip())
                    test_acc = float(test_acc.replace(")", "").strip())

                    # assert that values are floats
                    assert type(prob) is float
                    assert type(train_acc) is float
                    assert type(test_acc) is float

                else:
                    print(f"Unexpected result format: {r}")
                    train_acc = 0
                    test_acc = 0
                    prob = 0

            else:
                continue

        else:
            if len(results) != n_tasks:
                print(
                    f"Warning: Expected {n_tasks} tasks, but got {len(results)} in {file_path}"
                )
                return 0
            try:

                current_result = results[i]
                acc = current_result["accuracy"]
                train_acc = 0
                test_acc = acc

            except Exception as e:
                print(f"Error occurred while processing task {i} in {file_path}: {e}")
                continue

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        cur_n_tasks += 1

    if cur_n_tasks != n_tasks:
        print(f"Warning: Expected {n_tasks} tasks, but got {cur_n_tasks}")
        return 0

    if len(train_accuracies) == 0 or len(test_accuracies) == 0:
        print(f"No accuracies found in {file_path}")
        return 0

    # get mean train and test accuracies
    mean_train_acc = 100 * sum(train_accuracies) / len(train_accuracies)
    mean_test_acc = 100 * sum(test_accuracies) / len(test_accuracies)
    # print(f"Mean Train Accuracy: {mean_train_acc:.2f}")
    # print(f"Mean Test Accuracy: {mean_test_acc:.2f}")

    return mean_test_acc


def get_tokens(file_path, n_tasks):

    if "qualitative" in file_path:
        # load json from path
        with open(file_path, "r") as f:
            results = json.load(f)

        all_tokens = 0

        cur_n_tasks = 0
        for i in range(n_tasks):
            key = str(i)

            if type(results) is dict:
                # TODO get tokens from vlp results
                pass

            else:
                if len(results) != n_tasks:
                    print(
                        f"Warning: Expected {n_tasks} tasks, but got {len(results)} in {file_path}"
                    )
                    return 0
                try:
                    current_result = results[i]
                    tokens = current_result["output_tokens"]
                    all_tokens += tokens
                except:
                    continue

            cur_n_tasks += 1

        if cur_n_tasks != n_tasks:
            print(f"Warning: Expected {n_tasks} tasks, but got {cur_n_tasks}")
            return 0

    else:

        # replace .json with used_tokens.json
        file_path = file_path.replace(".json", "_used_tokens.txt")

        # cast string representation to int
        with open(file_path, "r") as f:
            file_content = f.read()

        print(f"File content: {file_content}")
        # replace "Total used tokens: " with ""
        file_content = file_content.replace("Total used tokens: ", "")
        all_tokens = int(file_content.strip())

    return all_tokens


def eval_all(
    incomplete=False,
    no_sampling=False,
    results_folder="results",
    distribution="naive_weighted",
):

    search_timeout = 10

    datasets = [
        "bongard-hoi",
        "bongard-op",
        "bongard-rwr",
        "cocologic",
        "CLEVR-Hans3-unconfounded",
    ]
    models = [
        # "gpt-5-mini",
        # "gpt-4o",
        "InternVL3-8B",
        "InternVL3-14B",
        # "InternVL3-78B",
        "Qwen2.5-VL-7B-Instruct",
        # "Qwen3-VL-30B-A3B-Instruct",
        # "Qwen3-VL-30B-A3B-Thinking",
        # "Molmo-7B",
        # "Ovis2.5-9B",
        "Kimi-VL-A3B-Instruct",
        # "Kimi-VL-A3B-Thinking-2506",
    ]

    df_1 = pd.DataFrame()
    df_for_plotting = pd.DataFrame()

    for dataset in datasets:

        print(f"\nDATASET: {dataset}\n")

        params = params_per_dataset[dataset]
        n_objects = params["n_objects"]
        n_properties = params["n_properties"]
        n_actions = params["n_actions"]
        max_program_depth = params["max_program_depth"]

        max_imgs = params["max_imgs"]
        # max_imgs = 24

        if no_sampling:
            methods = ["base_no_sampling", "no_sampling"]
            seeds = [0]
        else:
            methods = ["base", "no"]
            seeds = [0, 1, 2]

        for baseline in methods:

            for model in models:

                test_accuracies = []
                for seed in seeds:

                    dataset_name = dataset

                    if baseline == "base":
                        file_path = f"results/qualitative/{dataset_name}/direct_results_{model}_{seed}_{max_imgs}.json"
                    elif baseline == "structure":
                        file_path = f"results/baseline_with_structure/{dataset_name}_{model}_{max_imgs}_{seed}.json"
                    elif baseline == "base_no_sampling":
                        file_path = f"results/qualitative/{dataset_name}/no_sampling/direct_results_{model}_{seed}_{max_imgs}.json"
                    elif baseline == "no_sampling":
                        file_path = f"{results_folder}/{dataset}/no_sampling/discovered_programs_{model}_{search_timeout}_{max_program_depth}_{n_objects}_{n_properties}_{n_actions}_0_{max_imgs}_{distribution}_{seed}.json"
                    else:
                        file_path = f"{results_folder}/{dataset_name}/discovered_programs_{model}_{search_timeout}_{max_program_depth}_{n_objects}_{n_properties}_{n_actions}_0_{max_imgs}_{distribution}_{seed}.json"
                        # results_nvd/CLEVR-Hans3-unconfounded/discovered_programs_InternVL3-8B_10_5_10_10_0_10_naive_weighted_0.json
                    try:
                        test_acc = eval(file_path, n_tasks=n_tasks_per_dataset[dataset])
                        if test_acc != 0:
                            test_accuracies.append(test_acc)
                        # elif not incomplete:
                        #     test_accuracies.append(0)
                    except Exception as e:
                        # print(f"Error evaluating {file_path}: {e}")
                        # if not incomplete:
                        #     test_accuracies.append(0)
                        pass

                print(f"Test accuracies {model}:\t\t\t\t {test_accuracies}")

                if len(test_accuracies) == len(seeds) or incomplete:
                    mean_acc = np.mean(test_accuracies)
                    std_acc = np.std(test_accuracies)
                else:
                    mean_acc = 0
                    std_acc = 0

                if baseline == "base":
                    model = model + "\_baseline"
                elif baseline == "base_no_sampling":
                    model = model + "\_baseline\_no\_sampling"
                elif baseline == "structure":
                    model = model + "\_structure"
                elif baseline == "no_sampling":
                    model = model + "\_vlp\_no\_sampling"
                else:
                    model = model + "\_vlp"

                dataset_str = dataset + f"\_{max_imgs}"

                decimals = 1
                if no_sampling:
                    latex_acc = f"${mean_acc:.{decimals}f}$"
                else:
                    # latex_acc = f"${mean_acc:.{decimals}f} \\mbox{{\\tiny $\\pm$ {std_acc:.{decimals}f}}}$"
                    latex_acc = f"${mean_acc:.{decimals}f}$"

                df_1.loc[model, dataset_str] = latex_acc
                df_for_plotting.loc[model, dataset_str] = mean_acc

    # average over datasets for each model
    df_1["Average"] = df_for_plotting.mean(axis=1).apply(lambda x: f"${x:.1f}$")
    df_for_plotting["Average"] = df_for_plotting.mean(axis=1).apply(
        lambda x: f"${x:.1f}$"
    )

    # order columns average, then datasets in datasets list order
    cols = ["Average"]
    for dataset in datasets:
        params = params_per_dataset[dataset]
        max_imgs = params["max_imgs"]
        cols.append(dataset + f"\_{max_imgs}")
    df_1 = df_1[cols]
    df_for_plotting.to_csv("results/all_results.csv")
    print(df_for_plotting)
    print(df_1.to_latex())

    return df_1


def eval_max_imgs(no_sampling):

    distribution = "naive_weighted"
    search_timeout = 10

    datasets = [
        "bongard-hoi-max-img",
        "cocologic-max-img",
        "CLEVR-Hans3-unconfounded",
    ]
    models = [
        "InternVL3-8B",
        "InternVL3-14B",
        "Kimi-VL-A3B-Instruct",
        "Qwen2.5-VL-7B-Instruct",
        "Qwen3-VL-30B-A3B-Instruct",
    ]

    df_1 = pd.DataFrame()
    df_for_plotting = pd.DataFrame()

    n_tasks_more_imgs = {
        "bongard-hoi-max-img": 67,
        "cocologic-max-img": 8,
        "CLEVR-Hans3-unconfounded": 3,
    }

    for dataset in datasets:

        print(f"\nDATASET: {dataset}\n")

        params = params_per_dataset[dataset]
        n_objects = params["n_objects"]
        n_properties = params["n_properties"]
        n_actions = params["n_actions"]
        max_program_depth = params["max_program_depth"]

        if no_sampling:
            seeds = [0]
        else:
            seeds = [0]

        for max_imgs in [10, 20, 30, 50]:

            # for baseline in ["base", "no"]:
            for baseline in ["base", "no"]:

                for model in models:

                    test_accuracies = []
                    for seed in seeds:

                        if no_sampling:
                            if baseline == "base":
                                file_path = f"results/qualitative/{dataset}/no_sampling/direct_results_{model}_{seed}_{max_imgs}.json"
                            else:
                                if (
                                    "bongard-hoi-max-img" == dataset
                                    and baseline == "no"
                                ):
                                    file_path = f"results/{dataset}/no_sampling/discovered_programs_{model}_{search_timeout}_{max_program_depth}_{n_objects}_{n_properties}_{n_actions}_0_{max_imgs}_{distribution}_{seed}.json"
                                    print(f"File path: {file_path}")
                                    print(
                                        "results/bongard-hoi-max-img/no_sampling/discovered_programs_InternVL3-14B_10_4_10_5_10_0_10_naive_weighted_0.json"
                                    )
                                else:
                                    file_path = f"results/{dataset}/no_sampling/discovered_programs_{model}_{search_timeout}_{max_program_depth}_{n_objects}_{n_properties}_{n_actions}_{max_imgs}_{distribution}_{seed}.json"

                        else:
                            if baseline == "base":
                                file_path = f"results/qualitative/{dataset}/direct_results_{model}_{seed}_{max_imgs}.json"
                            else:
                                file_path = f"results/{dataset}/discovered_programs_{model}_{search_timeout}_{max_program_depth}_{n_objects}_{n_properties}_{n_actions}_0_{max_imgs}_{distribution}_{seed}.json"
                                # results_nvd/CLEVR-Hans3-unconfounded/discovered_programs_InternVL3-8B_10_5_10_10_0_10_naive_weighted_0.json
                        try:
                            n_tasks = n_tasks_more_imgs[dataset]
                            test_acc = eval(file_path, n_tasks=n_tasks)
                            if test_acc != 0:
                                test_accuracies.append(test_acc)
                        except Exception as e:
                            pass

                    print(f"Test accuracies {model}:\t\t\t\t {test_accuracies}")

                    if len(test_accuracies) == len(seeds):
                        mean_acc = np.mean(test_accuracies)
                        std_acc = np.std(test_accuracies)
                    else:
                        mean_acc = np.nan
                        std_acc = np.nan

                    if baseline == "base":
                        model = model + "\_baseline"
                    elif baseline == "structure":
                        model = model + "\_structure"
                    else:
                        model = model + "\_vlp"

                    dataset_str = dataset + f"\_{max_imgs}"

                    decimals = 1
                    latex_acc = f"${mean_acc:.{decimals}f}$"

                    df_1.loc[model, dataset_str] = latex_acc
                    df_for_plotting.loc[model, dataset_str] = mean_acc

    # print df for plotting
    print(df_for_plotting)
    df_for_plotting.to_csv("results/all_results_max_imgs.csv")

    df_for_plotting.to_csv("results/all_results_max_imgs.csv")
    print(df_1.to_latex())


def eval_thinking(incomplete=False):

    distribution = "naive_weighted"
    search_timeout = 10

    datasets = [
        "cocologic",
        "cocologic-new",
        "CLEVR-Hans3-unconfounded",
    ]
    models = [
        "gpt-5",
        "gpt-5-chat-latest",
        "gpt-4o",
        "Kimi-VL-A3B-Instruct",
        "Qwen3-VL-30B-A3B-Instruct",
        "Kimi-VL-A3B-Thinking-2506",
        "Qwen3-VL-30B-A3B-Thinking",
    ]

    df_1 = pd.DataFrame()
    df_for_plotting = pd.DataFrame()

    for dataset in datasets:

        print(f"\nDATASET: {dataset}\n")

        params = params_per_dataset[dataset]
        n_objects = params["n_objects"]
        n_properties = params["n_properties"]
        n_actions = params["n_actions"]
        max_program_depth = params["max_program_depth"]
        max_imgs = params["max_imgs"]

        for baseline in ["base", "no"]:

            for model in models:

                test_accuracies = []
                for seed in [0]:

                    if baseline == "base":
                        if "Think" in model or "gpt-5" in model:
                            file_path = f"results/qualitative/{dataset}/direct_results_{model}_{seed}_{max_imgs}_think.json"
                        else:
                            file_path = f"results/qualitative/{dataset}/direct_results_{model}_{seed}_{max_imgs}.json"
                    elif baseline == "structure":
                        file_path = f"results/baseline_with_structure/{dataset}_{model}_{max_imgs}_{seed}.json"
                    elif baseline == "base_no_sampling":
                        file_path = f"results/qualitative/{dataset}/no_sampling/direct_results_{model}_{seed}_{max_imgs}.json"
                        if "Think" in model or "gpt-5" in model:
                            file_path = f"results/qualitative/{dataset}/no_sampling/direct_results_{model}_{seed}_{max_imgs}_think.json"
                    elif baseline == "no_sampling":
                        file_path = f"results/{dataset}/no_sampling/discovered_programs_{model}_{search_timeout}_{max_program_depth}_{n_objects}_{n_properties}_{n_actions}_0_{max_imgs}_{distribution}_{seed}.json"
                    else:
                        file_path = f"results/{dataset}/discovered_programs_{model}_{search_timeout}_{max_program_depth}_{n_objects}_{n_properties}_{n_actions}_0_{max_imgs}_{distribution}_{seed}.json"
                    try:
                        test_acc = eval(file_path, n_tasks=n_tasks_per_dataset[dataset])
                        tokens = get_tokens(
                            file_path, n_tasks=n_tasks_per_dataset[dataset]
                        )
                        print(f"Tokens used by {model}: {tokens}")

                        if test_acc != 0:
                            test_accuracies.append(test_acc)
                        # elif not incomplete:
                        #     test_accuracies.append(0)
                    except Exception as e:
                        # print(f"Error evaluating {file_path}: {e}")
                        # if not incomplete:
                        #     test_accuracies.append(0)
                        tokens = 0
                        pass

                print(f"Test accuracies {model}:\t\t\t\t {test_accuracies}")

                if len(test_accuracies) == 3 or incomplete:
                    mean_acc = np.mean(test_accuracies)
                    std_acc = np.std(test_accuracies)
                else:
                    mean_acc = 0
                    std_acc = 0

                if baseline == "base":
                    model = model + "\_baseline"
                elif baseline == "base_no_sampling":
                    model = model + "\_baseline\_no\_sampling"
                elif baseline == "structure":
                    model = model + "\_structure"
                elif baseline == "no_sampling":
                    model = model + "\_vlp\_no\_sampling"
                else:
                    model = model + "\_vlp"

                dataset_str = dataset + f"\_{max_imgs}"

                decimals = 1

                latex_acc = f"${mean_acc:.{decimals}f}$"

                df_1.loc[model, dataset_str] = latex_acc
                df_for_plotting.loc[model, dataset_str] = mean_acc

                dataset_tokens = dataset + " Tokens"
                df_1.loc[model, dataset_tokens] = f"${tokens}$"
                df_for_plotting.loc[model, dataset_tokens] = tokens

    df_for_plotting.to_csv("results/thinking_results.csv")
    print(df_for_plotting)
    print(df_1.to_latex())


if __name__ == "__main__":

    df_uniform = eval_all(
        incomplete=False,
        no_sampling=False,
        results_folder="results",
        distribution="uniform",
    )

    df_naive_weighted = eval_all(
        incomplete=False,
        no_sampling=False,
        results_folder="results",
        distribution="naive_weighted",
    )

    # calculate difference between the two dataframes
    df_difference = df_uniform.copy()
    for col in df_uniform.columns:
        for row in df_uniform.index:
            val_uniform = df_uniform.loc[row, col]
            val_naive = df_naive_weighted.loc[row, col]
            if val_uniform == "$0.0$" or val_naive == "$0.0$":
                df_difference.loc[row, col] = "$-$"
                continue
            try:
                num_uniform = float(val_uniform.replace("$", ""))
                num_naive = float(val_naive.replace("$", ""))
                diff = num_naive - num_uniform
                df_difference.loc[row, col] = f"${diff:.1f}$"
                # add a + if there is a positive difference
                if diff > 0:
                    df_difference.loc[row, col] = f"$+{diff:.1f}$"
            except:
                df_difference.loc[row, col] = "$-$"

    print("Difference between uniform and naive weighted:")
    print(df_difference.to_latex())

    eval_confounded(no_sampling=False)
    eval_max_imgs(no_sampling=False)
    eval_thinking(incomplete=True)
