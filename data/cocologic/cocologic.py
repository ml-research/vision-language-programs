from torch.utils.data import Dataset
import numpy as np
import os
import json
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision import transforms
from collections import defaultdict, Counter


def load_category_mapping(annotation_file):
    with open(annotation_file, "r") as f:
        coco_data = json.load(f)
    categories = coco_data["categories"]
    return {cat["id"]: cat["name"] for cat in categories}


cocologic_objects_from_gt = {
    "Ambiguous Pairs (Pet vs Ride Paradox)": ["dog", "cat", "bicycle", "motorcycle"],
    "Pair of Pets": ["cat", "dog", "bird"],
    "Rural Animal Scene": ["cow", "horse", "sheep"],
    "Conflicted Companions (Leash vs Licence)": ["dog", "car"],
    "Animal Meets Traffic": ["horse", "cow", "sheep", "car", "bus", "traffic light"],
    "Occupied Interior": ["couch", "chair", "person"],
    "Empty Seat": ["couch", "chair", "person"],
    "Odd Ride Out": ["bicycle", "motorcycle", "car", "bus"],
    "Personal Transport XOR Car": ["person", "motorcycle", "car"],
    "Unlikely Breakfast Guests": ["bowl", "dog", "cat", "horse", "cow", "sheep"],
}

PATH_TO_COCO = "data/cocologic/coco/"


class COCOLogicDataset(Dataset):
    def __init__(
        self,
        annotation_file,
        image_dir,
        category_id_to_name,
        transform=None,
        filter_no_labels=True,
        exclusive_label=True,
        exclusive_match_only=True,
        log_statistics=False,
        version=10,
    ):
        """
        annotation_file: path to COCO annotations JSON
        image_dir: path to the images folder
        category_id_to_name: dict mapping COCO category id to names
        transform: torchvision transforms
        filter_no_labels: if True, drops images that satisfy no logical classes
        exclusive_label: if True, only assign first matching class as 1 (others 0)
        exclusive_match_only: if True, only include images that match exactly one logical class

            exclusive_match_only | exclusive_label | Resulting Effect
            False | False | Multi-label dataset, overlapping classes allowed.
            False | True | Overlapping images included, but only first class is used in label.
            True | False | Only images with one class are included, label is still multi-hot (only one 1).
            True | True | Clean single-class dataset, label is one-hot — ideal for multi-class classification.
        """

        self.image_dir = image_dir
        self.transform = transform
        self.exclusive_label = exclusive_label
        self.exclusive_match_only = exclusive_match_only
        self.version = version

        # Load COCO annotations
        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        self.imgs = {img["id"]: img for img in coco_data["images"]}
        self.annotations = coco_data["annotations"]

        self.image_to_categories = {}
        category_frequency = Counter()
        for ann in self.annotations:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            cat_name = category_id_to_name[cat_id]
            self.image_to_categories.setdefault(img_id, set()).add(cat_name)
            category_frequency[cat_name] += 1

        # # logical class definitions
        if self.version == 10:
            self.logical_classes = [
                # 1. Ambiguous Pairs (Pet vs Ride Paradox). The image includes either a cat or a dog (but not both),
                # and either a bicycle or a motorcycle (but not both).
                (
                    "Ambiguous Pairs (Pet vs Ride Paradox)",
                    lambda cats: (("cat" in cats) ^ ("dog" in cats))
                    and (("bicycle" in cats) ^ ("motorcycle" in cats)),
                ),
                # 2. Pair of Pets. Exactly two of the following animal categories are present: a cat, a dog, or a bird.
                (
                    "Pair of Pets",
                    lambda cats: sum(c in cats for c in ["cat", "dog", "bird"]) == 2,
                ),
                # 3. Rural Animal Scene. The image includes one or more rural animals (cow, horse, or sheep) and no people.
                (
                    "Rural Animal Scene",
                    lambda cats: any(c in cats for c in ["cow", "horse", "sheep"])
                    and "person" not in cats,
                ),
                # 4. Conflicted Companions (Leash vs Licence). An image features either a dog or a car, but not both.
                (
                    "Conflicted Companions (Leash vs Licence)",
                    lambda cats: ("dog" in cats) ^ ("car" in cats),
                ),
                # 5. Animal Meet Traffic. The image contains a rural animal (horse, cow, or sheep) and a
                # traffic-related object (car, bus, or traffic light).
                (
                    "Animal Meets Traffic",
                    lambda cats: any(c in cats for c in ["horse", "cow", "sheep"])
                    and any(c in cats for c in ["car", "bus", "traffic light"]),
                ),
                # 6. Occupied Interior. The image includes furniture (a couch or chair) and at least one person.
                (
                    "Occupied Interior",
                    lambda cats: any(c in cats for c in ["couch", "chair"])
                    and "person" in cats
                    and sum(c == "person" for c in cats) == 1,
                ),
                # 7. Empty Seat. The image includes indoor furniture (a couch or chair) but no person is present.
                (
                    "Empty Seat",
                    lambda cats: any(c in cats for c in ["couch", "chair"])
                    and "person" not in cats,
                ),
                # 8. Odd Ride Out. Exactly one of the following categories is present: a bicycle, motorcycle, car, or bus.
                (
                    "Odd Ride Out",
                    lambda cats: sum(
                        c in cats for c in ["bicycle", "motorcycle", "bus", "car"]
                    )
                    == 1,
                ),
                # 9. Personal Transport XOR Car. A person is present alongside either a bicycle or a car — but not both.
                (
                    "Personal Transport XOR Car",
                    lambda cats: "person" in cats
                    and (("bicycle" in cats) ^ ("car" in cats)),
                ),
                # 10. Unlikely Breakfast Guests. The image shows a bowl (suggesting food) and at least one animal (dog, cat, horse, cow, or sheep).
                (
                    "Unlikely Breakfast Guests",
                    lambda cats: "bowl" in cats
                    and any(c in cats for c in ["dog", "cat", "horse", "cow", "sheep"]),
                ),
            ]
        else:
            raise Exception("COCOLogic other than 10 is not implemented")

        total_images = len(self.imgs)
        kept_images = 0
        class_counts = defaultdict(int)
        class_cooccurrence = Counter()

        self.image_ids = []
        for img_id in self.imgs:
            cats = self.image_to_categories.get(img_id, set())
            labels = [int(fn(cats)) for _, fn in self.logical_classes]

            if filter_no_labels and not any(labels):
                continue

            if exclusive_match_only and sum(labels) != 1:
                continue

            self.image_ids.append(img_id)
            kept_images += 1

            label_tuple = tuple(labels)
            class_cooccurrence[label_tuple] += 1

            for i, val in enumerate(labels):
                if val:
                    class_counts[self.logical_classes[i][0]] += 1

        if log_statistics:
            print(f"\nLogicalCOCODataset: Loaded {total_images} images.")
            print(
                f"Filtered to {kept_images} images after applying logical class filters.\n"
            )

            print("Per-Class Image Count:")
            for class_name, _ in self.logical_classes:
                print(f" - {class_name:<25}: {class_counts[class_name]}")

            print("\nTop 20 Class Co-occurrence Patterns:")
            for pattern, count in class_cooccurrence.most_common(20):
                pattern_str = ", ".join(
                    [self.logical_classes[i][0] for i, v in enumerate(pattern) if v]
                )
                print(f" - [{pattern_str or 'None'}] : {count} images")

            print("\nTop 20 Category Frequencies:")
            for cat, count in category_frequency.most_common(20):
                print(f" - {cat:<20}: {count} annotations")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.imgs[img_id]
        img_path = os.path.join(self.image_dir, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        categories = self.image_to_categories.get(img_id, set())
        labels = [int(fn(categories)) for _, fn in self.logical_classes]

        if self.exclusive_label:
            exclusive = [0] * len(labels)
            for i, val in enumerate(labels):
                if val:
                    exclusive[i] = 1
                    break
            labels = exclusive

        label_index = torch.tensor(labels.index(1), dtype=torch.long)
        # return labels as one.hot and twice for consistency
        return image, torch.tensor(labels), categories, img_id, img_path


def sort_img_paths_to_cover_categories(
    i, img_paths, img_path_to_categories, target_categories
):
    sorted_imgs = []
    covered_categories = set()

    for j, img_path in enumerate(img_paths):
        img_cats = img_path_to_categories.get(img_path, set())
        new_cats = img_cats & target_categories - covered_categories
        if new_cats:
            sorted_imgs.append(img_path)
            covered_categories.update(new_cats)
        if covered_categories == target_categories:
            print(
                f"Class {i}: All target categories covered (iteration {j}, img n {len(sorted_imgs)})."
            )
            break

    # if not all target categories are covered, print a warning
    if covered_categories != target_categories:
        missing = target_categories - covered_categories
        print(f"Warning: Class {i}: Missing categories {missing}")

    # append the remaining images
    for img_path in img_paths:
        if img_path not in sorted_imgs:
            sorted_imgs.append(img_path)

    assert len(img_paths) == len(sorted_imgs)
    return sorted_imgs


if __name__ == "__main__":

    for split in ["train", "val"]:

        category_map = load_category_mapping(
            f"{PATH_TO_COCO}/annotations/instances_{split}2017.json"
        )

        dataset = COCOLogicDataset(
            annotation_file=f"{PATH_TO_COCO}/annotations/instances_{split}2017.json",
            image_dir=f"{PATH_TO_COCO}/images/{split}2017/",
            category_id_to_name=category_map,
            transform=None,
            filter_no_labels=True,
            exclusive_label=True,
            exclusive_match_only=True,
            log_statistics=True,
        )

        print(f"Dataset size: {len(dataset)}")

        # create pos / neg dataset based on coco logic
        binary_coco_logic = {}
        classes = [x[0] for x in dataset.logical_classes]

        img_path_to_categories = {}

        for cls in classes:
            binary_coco_logic[cls] = {
                "pos": [],
                "neg": [],
            }
        for i in range(len(dataset)):
            # if i > 10000:
            #     break
            _, labels, categories, img_id, image_path = dataset[i]

            img_path_to_categories[image_path] = categories

            # get index of the first 1 in labels
            j = torch.where(labels == 1)[0][0].item() if 1 in labels else -1
            binary_coco_logic[classes[j]]["pos"].append(
                image_path,
            )
            for k in range(len(labels)):
                if k != j:
                    binary_coco_logic[classes[k]]["neg"].append(
                        image_path,
                    )

        for j in range(len(classes)):
            sorted_imgs = sort_img_paths_to_cover_categories(
                j,
                binary_coco_logic[classes[j]]["pos"],
                img_path_to_categories,
                set(cocologic_objects_from_gt[classes[j]]),
            )
            binary_coco_logic[classes[j]]["pos"] = sorted_imgs

        # make sure all lists are of the same length
        min_length = min(
            len(binary_coco_logic[cls]["pos"]) for cls in binary_coco_logic
        )
        max_length = max(
            len(binary_coco_logic[cls]["pos"]) for cls in binary_coco_logic
        )
        print(f"Maximum length of positive samples: {max_length}")
        print(f"Minimum length of positive samples: {min_length}")

        min_length = 100

        # truncate all lists to the minimum length
        for cls in binary_coco_logic:
            binary_coco_logic[cls]["pos"] = binary_coco_logic[cls]["pos"][:min_length]
            binary_coco_logic[cls]["neg"] = binary_coco_logic[cls]["neg"][:min_length]

        # save to json
        with open(f"data/cocologic/binary_coco_logic_{split}.json", "w") as f:
            json.dump(binary_coco_logic, f, indent=4)
        print(
            f"Binary COCO Logic dataset saved to data/cocologic/binary_coco_logic_{split}.json"
        )
        print("Done!")
