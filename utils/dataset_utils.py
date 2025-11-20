import os
import json
from datasets import load_dataset
from PIL import Image

N_TEST_IMAGES_CLEVR = 10


# get index of first element with rule
def _get_index_of_rule(data, rule):
    for i, item in enumerate(data):
        pos_imgs, neg_imgs, r = item
        if r == rule:
            return i
    return -1


def check_train_test_exclusivity(data):
    for i, task in enumerate(data):
        pos_imgs, neg_imgs, pos_test_imgs, neg_test_imgs, _ = task
        train_set = set(pos_imgs + neg_imgs)
        test_set = set(pos_test_imgs + neg_test_imgs)
        intersection = train_set.intersection(test_set)
        if len(intersection) > 0:
            print(f"Task {i}:")
            print("Train and test sets are not exclusive!")
            print("Intersection:", intersection)
            return False
    print("Train and test sets are exclusive.")
    return True


def load_data(dataset, max_imgs=6):
    if dataset == "bongard-op":

        # load bongard open world dataset
        ds = load_dataset("rujiewu/Bongard-OpenWorld")
        data = ds["test"]
        image_top_path = "data/bongard-op"

        prepared_data = []

        for i, sample in enumerate(data):

            gt_rule = sample["concept"]
            image_paths = sample["imageFiles"]
            image_paths = [os.path.join(image_top_path, x) for x in image_paths]

            pos_imgs_paths_all = image_paths[:7]  # first 7 are positive
            neg_imgs_paths_all = image_paths[7:]  # next 7 are negative

            # remove last image (test image)
            pos_imgs_paths = pos_imgs_paths_all[:-1]
            neg_imgs_paths = neg_imgs_paths_all[:-1]

            pos_test_imgs_paths = [pos_imgs_paths_all[-1]]
            neg_test_imgs_paths = [neg_imgs_paths_all[-1]]

            assert len(pos_imgs_paths) == 6
            assert len(neg_imgs_paths) == 6
            assert len(pos_test_imgs_paths) == 1
            assert len(neg_test_imgs_paths) == 1

            for img_path in pos_imgs_paths:
                assert "pos" in img_path
            for img_path in neg_imgs_paths:
                assert "neg" in img_path

            assert "pos" in pos_test_imgs_paths[0]
            assert "neg" in neg_test_imgs_paths[0]

            prepared_data.append(
                (
                    pos_imgs_paths,
                    neg_imgs_paths,
                    pos_test_imgs_paths,
                    neg_test_imgs_paths,
                    gt_rule,
                )
            )

        assert check_train_test_exclusivity(prepared_data)
        return prepared_data

    elif dataset == "bongard-rwr":

        dataset_folder = "data/bongard-rwr/dataset"
        # get all folders in dataset_folder
        bps = [
            f
            for f in os.listdir(dataset_folder)
            if os.path.isdir(os.path.join(dataset_folder, f))
        ]

        gt_rules_path = "data/bongard-rwr/gt_rules.json"
        gt_rules = json.load(open(gt_rules_path, "r"))

        # convert numeric name to three digit string
        # bps = [f"{int(f):03d}" for f in bps]
        # join path
        bps = [os.path.join(dataset_folder, f) for f in bps]
        # sort bps by name
        bps.sort()

        prepared_data = []

        for sample in bps:
            print(f"Processing {sample}...")

            gt_rule = gt_rules[os.path.basename(sample)][0]

            pos_imgs_paths = []
            neg_imgs_paths = []
            pos_test_imgs_paths = []
            neg_test_imgs_paths = []

            pos_folder = os.path.join(sample, "left")
            neg_folder = os.path.join(sample, "right")

            # get all images in pos_folder
            for img in os.listdir(pos_folder):
                if (
                    img.endswith(".png")
                    or img.endswith(".jpg")
                    or img.endswith(".jpeg")
                ):
                    pos_imgs_paths.append(os.path.join(pos_folder, img))
            # get all images in neg_folder
            for img in os.listdir(neg_folder):
                if (
                    img.endswith(".png")
                    or img.endswith(".jpg")
                    or img.endswith(".jpeg")
                ):
                    neg_imgs_paths.append(os.path.join(neg_folder, img))

            pos_train_imgs_paths = pos_imgs_paths[:-1]
            neg_train_imgs_paths = neg_imgs_paths[:-1]
            pos_test_imgs_paths = [pos_imgs_paths[-1]]
            neg_test_imgs_paths = [neg_imgs_paths[-1]]

            prepared_data.append(
                (
                    pos_train_imgs_paths,
                    neg_train_imgs_paths,
                    pos_test_imgs_paths,
                    neg_test_imgs_paths,
                    gt_rule,
                )
            )

        assert check_train_test_exclusivity(prepared_data)
        print(f"Loaded {len(prepared_data)} Bongard tasks from {dataset_folder}")
        return prepared_data

    elif "cocologic" in dataset:

        json_path_train = "data/cocologic/binary_coco_logic_train.json"
        json_path_val = "data/cocologic/binary_coco_logic_val.json"

        tasks = json.load(open(json_path_train, "r"))
        tasks_val = json.load(open(json_path_val, "r"))

        prepared_data = []

        # iterate over dictionary
        for key, sample in tasks.items():

            # print(sample)
            val_sample = tasks_val[key]

            pos_imgs_paths = sample["pos"]
            neg_imgs_paths = sample["neg"]

            pos_test_imgs_paths = val_sample["pos"][:N_TEST_IMAGES_CLEVR]
            neg_test_imgs_paths = val_sample["neg"][:N_TEST_IMAGES_CLEVR]

            # report number of images before limiting
            print(
                f"Task {key}: Pos total: {len(pos_imgs_paths)}, Neg total: {len(neg_imgs_paths)}"
            )

            if len(pos_imgs_paths) < (max_imgs):
                print(
                    f"Skipping task {key} due to insufficient positive images ({len(pos_imgs_paths)} < {max_imgs})"
                )
                continue

            if dataset == "cocologic-max-img" and len(pos_imgs_paths) < 50:
                print(
                    f"Skipping task {key} due to insufficient positive images for max-img setting ({len(pos_imgs_paths)} < 50)"
                )
                continue

            # get number of images
            if len(pos_imgs_paths) > max_imgs:
                pos_imgs_paths = pos_imgs_paths[:max_imgs]
            if len(neg_imgs_paths) > max_imgs:
                neg_imgs_paths = neg_imgs_paths[:max_imgs]

            # report number of images
            print(
                f"  Pos train: {len(pos_imgs_paths)}, Neg train: {len(neg_imgs_paths)}, Pos test: {len(pos_test_imgs_paths)}, Neg test: {len(neg_test_imgs_paths)}"
            )

            prepared_data.append(
                (
                    pos_imgs_paths,
                    neg_imgs_paths,
                    pos_test_imgs_paths,
                    neg_test_imgs_paths,
                    key,
                )
            )

        assert check_train_test_exclusivity(prepared_data)

        return prepared_data

    elif dataset == "CLEVR-Hans3-unconfounded":
        train_folder = "data/CLEVR-Hans3/train/images"
        if dataset == "CLEVR-Hans3-unconfounded":
            test_folder = "data/CLEVR-Hans3/val/images"
        else:
            test_folder = "data/CLEVR-Hans3/test/images"
        prepared_data = []

        for cls in range(3):

            pos_imgs_paths = [
                f"{train_folder}/CLEVR_Hans_classid_{cls}_{id:06d}.png"
                for id in range(50)
            ]
            pos_test_imgs_paths = [
                f"{test_folder}/CLEVR_Hans_classid_{cls}_{id:06d}.png"
                for id in range(50)
            ]

            n_false_support_img_per_class = (max_imgs + 1) // 2

            neg_imgs_paths = []
            neg_test_imgs_paths = []

            for neg_cls in range(3):
                if neg_cls == cls:
                    continue

                # train
                neg_imgs_paths = neg_imgs_paths + [
                    f"{train_folder}/CLEVR_Hans_classid_{neg_cls}_{id:06d}.png"
                    for id in range(n_false_support_img_per_class)
                ]
                # test
                neg_test_imgs_paths = neg_test_imgs_paths + [
                    f"{test_folder}/CLEVR_Hans_classid_{neg_cls}_{id:06d}.png"
                    for id in range(N_TEST_IMAGES_CLEVR)
                ]

            if len(pos_imgs_paths) > max_imgs:
                pos_imgs_paths = pos_imgs_paths[:max_imgs]
                neg_imgs_paths = neg_imgs_paths[:max_imgs]

                pos_test_imgs_paths = pos_test_imgs_paths[:N_TEST_IMAGES_CLEVR]
                neg_test_imgs_paths = neg_test_imgs_paths[:N_TEST_IMAGES_CLEVR]

            # report number of images
            print(
                f"Class {cls}: Pos train: {len(pos_imgs_paths)}, Neg train: {len(neg_imgs_paths)}, Pos test: {len(pos_test_imgs_paths)}, Neg test: {len(neg_test_imgs_paths)}"
            )

            prepared_data.append(
                (
                    pos_imgs_paths,
                    neg_imgs_paths,
                    pos_test_imgs_paths,
                    neg_test_imgs_paths,
                    "Class_" + str(cls),
                )
            )

        assert check_train_test_exclusivity(prepared_data)

        print(f"Loaded {len(prepared_data)} CLEVR-Hans tasks from {train_folder}")
        return prepared_data

    elif dataset == "bongard-hoi" or dataset == "bongard-hoi-max-img":

        test_set_paths = [
            "data/bongard-hoi/bongard_hoi_release/bongard_hoi_test_seen_obj_seen_act.json",
            "data/bongard-hoi/bongard_hoi_release/bongard_hoi_test_seen_obj_unseen_act.json",
            "data/bongard-hoi/bongard_hoi_release/bongard_hoi_test_unseen_obj_seen_act.json",
            "data/bongard-hoi/bongard_hoi_release/bongard_hoi_test_unseen_obj_unseen_act.json",
        ]

        top_path = "data/bongard-hoi/hake"

        prepared_data = []
        all_rules = []
        expected_number_of_concepts = 0
        task_dict = {}

        for test_set_path in test_set_paths:
            with open(test_set_path, "r") as f:
                test_set = json.load(f)

            # preserve rule order within each file
            rules = []
            for _, _, rule in test_set:
                if rule not in rules:
                    rules.append(rule)

            print(f"Total unique rules: {len(rules)}")
            expected_number_of_concepts += len(rules)

            for rule in rules:
                if rule not in task_dict:
                    task_dict[rule] = {
                        "rule": rule,
                        "pos": [],
                        "neg": [],
                        "pos_test": [],
                        "neg_test": [],
                    }

            for pos_imgs, neg_imgs, rule in test_set:
                pos_imgs_paths = [
                    os.path.join(top_path, img["im_path"]) for img in pos_imgs
                ]
                neg_imgs_paths = [
                    os.path.join(top_path, img["im_path"]) for img in neg_imgs
                ]

                pos_train_imgs = pos_imgs_paths[:-1]
                neg_train_imgs = neg_imgs_paths[:-1]
                pos_test_imgs = [pos_imgs_paths[-1]]
                neg_test_imgs = [neg_imgs_paths[-1]]

                assert len(pos_train_imgs) + len(pos_test_imgs) == 7
                assert len(neg_train_imgs) + len(neg_test_imgs) == 7

                task_dict[rule]["pos"].extend(pos_train_imgs)
                task_dict[rule]["neg"].extend(neg_train_imgs)
                task_dict[rule]["pos_test"].extend(pos_test_imgs)
                task_dict[rule]["neg_test"].extend(neg_test_imgs)

        # sort alphabetically for consistency
        task_dict = dict(sorted(task_dict.items()))
        all_rules = list(task_dict.keys())

        assert len(all_rules) == 166

        # build prepared_data cleanly and only once
        prepared_data = []

        for key, value in task_dict.items():
            if len(value["pos"]) < 50 and dataset == "bongard-hoi-max-img":
                print(f"Skipping rule {key} due to insufficient images.")
                expected_number_of_concepts -= 1
                continue

            # print(f"Rule: {key}")
            # print(
            #     f"  Pos train: {len(value['pos'])}, Neg train: {len(value['neg'])}, "
            #     f"Pos test: {len(value['pos_test'])}, Neg test: {len(value['neg_test'])}"
            # )

            # limit number of training/test images if needed
            pos_train_imgs = value["pos"]
            neg_train_imgs = value["neg"]
            pos_test_imgs = value["pos_test"]
            neg_test_imgs = value["neg_test"]

            all_pos_train_imgs = pos_train_imgs[:50]
            all_neg_train_imgs = neg_train_imgs[:50]

            pos_train_imgs = pos_train_imgs[:max_imgs]
            neg_train_imgs = neg_train_imgs[:max_imgs]

            n_test_imgs = 4 if dataset == "bongard-hoi-max-img" else 1

            if "bongard-hoi-max-img" in dataset:
                pos_test_imgs = [
                    img for img in pos_test_imgs if img not in all_pos_train_imgs
                ][:n_test_imgs]
                neg_test_imgs = [
                    img for img in neg_test_imgs if img not in all_neg_train_imgs
                ][:n_test_imgs]

            else:
                pos_test_imgs = [
                    img for img in pos_test_imgs if img not in pos_train_imgs
                ][:n_test_imgs]
                neg_test_imgs = [
                    img for img in neg_test_imgs if img not in neg_train_imgs
                ][:n_test_imgs]

            if len(pos_test_imgs) < n_test_imgs or len(neg_test_imgs) < n_test_imgs:
                print(
                    f"  Skipping rule {key} due to insufficient test images after train-test exclusivity filtering."
                )
                expected_number_of_concepts -= 1
                continue

            # report number of images

            print(
                f"  Final Pos train: {len(pos_train_imgs)}, Neg train: {len(neg_train_imgs)}, "
                f"Pos test: {len(pos_test_imgs)}, Neg test: {len(neg_test_imgs)}"
            )

            prepared_data.append(
                (pos_train_imgs, neg_train_imgs, pos_test_imgs, neg_test_imgs, key)
            )

        print(len(prepared_data))
        print(expected_number_of_concepts)
        assert len(prepared_data) == expected_number_of_concepts
        if dataset == "bongard-hoi":
            assert len(prepared_data) == 166

        # build and save rule dict
        rule_dict = {idx: rule for idx, rule in enumerate(all_rules)}
        with open(f"data/bongard-hoi/rules_{dataset}.json", "w") as f:
            json.dump(rule_dict, f)

        assert check_train_test_exclusivity(prepared_data)

        return prepared_data

    else:
        raise ValueError(f"Dataset {dataset} not supported.")


if __name__ == "__main__":
    dataset = "cocologic"

    data = load_data(dataset, max_imgs=6)
