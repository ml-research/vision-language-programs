from tqdm import tqdm
from method.type_system import *
import json
import pandas as pd
import torch


def pre_collect_vlm_answers(
    examples, objects, properties, actions, semantics, primitive_types
):
    """Pre-collect VLM answers for functions with IMG input.
    TODO: implement objects, actions, properties
    """
    # get name of functions with IMG input
    vlm_functions = []

    object_functions = []
    property_functions = []
    object_property_functions = []
    object_action_functions = []

    for k, v in primitive_types.items():
        # test if v has attribute type_in
        if not hasattr(v, "type_in"):
            continue
        if v.type_in == IMG:
            vlm_functions.append(k)
            if v.type_out.type_in == OBJECT:
                try:
                    is_object_action = v.type_out.type_out.type_in == ACTION
                    is_object_property = v.type_out.type_out.type_in == PROPERTY
                except AttributeError:
                    is_object_action = False
                    is_object_property = False
                if is_object_action:
                    object_action_functions.append(k)
                elif is_object_property:
                    object_property_functions.append(k)
                else:
                    object_functions.append(k)

            elif v.type_out.type_in == PROPERTY:
                property_functions.append(k)

    vlm_functions = [semantics[k] for k in vlm_functions]

    object_functions = [semantics[k] for k in object_functions]
    property_functions = [semantics[k] for k in property_functions]
    object_property_functions = [semantics[k] for k in object_property_functions]
    object_action_functions = [semantics[k] for k in object_action_functions]

    # number of prompts to make
    n_prompts = len(examples) * (
        len(object_functions) * len(objects)
        + len(property_functions) * len(properties)
        + len(object_property_functions) * len(objects) * len(properties)
        + len(object_action_functions) * len(objects) * len(actions)
    )

    progress = tqdm(total=n_prompts)

    for f in object_functions:
        for e in examples:
            input = e[0][0]

            for o in objects:
                _ = f(input)(o)
                progress.update(1)

    for f in property_functions:
        for e in examples:
            input = e[0][0]

            for p in properties:
                _ = f(input)(p)
                progress.update(1)

    for f in object_property_functions:
        for e in examples:
            input = e[0][0]

            for o in objects:
                for p in properties:
                    _ = f(input)(o)(p)
                    progress.update(1)

    for f in object_action_functions:
        for e in examples:
            input = e[0][0]

            for o in objects:
                for a in actions:
                    _ = f(input)(o)(a)
                    progress.update(1)

    print("Pre-collection of VLM answers done.")


def pre_collect_img_representations(examples, variables, semantics, primitive_types):
    """Pre-collect image representations for functions with IMG input."""

    def convert_to_string(element):
        if isinstance(element, str):
            return element
        else:
            try:
                element = str(element)
                return element
            except Exception as e:
                print(f"Error converting to string: {e}")
                return None

    objects_from_img = semantics["get_objects"]
    try:
        actions_from_img = semantics["get_actions"]
    except KeyError:
        actions_from_img = None

    try:
        exists_small = semantics["exists_object_small_in_img"]
    except KeyError:
        exists_small = None

    try:
        exists_large = semantics["exists_object_large_in_img"]
    except KeyError:
        exists_large = None

    try:
        exists_with_property_small = semantics[
            "exists_object_with_property_small_in_img"
        ]
    except KeyError:
        exists_with_property_small = None

    try:
        exists_with_property_large = semantics[
            "exists_object_with_property_large_in_img"
        ]
    except KeyError:
        exists_with_property_large = None

    img_object_representations = []
    img_action_representations = []

    for e in examples:
        input = e[0][0]
        output = objects_from_img(input)
        # print(output)
        img_object_representations.append(output)

        if actions_from_img is not None and len(variables.get("actions", [])) > 0:

            output = actions_from_img(input)
            # print(output)
            img_action_representations.append(output)

        if exists_small is not None:
            for o in variables.get("objects", []):
                _ = exists_small(input)(o)
        if exists_large is not None:
            for o in variables.get("objects", []):
                _ = exists_large(input)(o)
        if exists_with_property_small is not None:
            for o in variables.get("objects", []):
                for p in variables.get("properties", []):
                    _ = exists_with_property_small(input)(o)(p)
        if exists_with_property_large is not None:
            for o in variables.get("objects", []):
                for p in variables.get("properties", []):
                    _ = exists_with_property_large(input)(o)(p)

    objects = []
    properties = []
    actions = []

    # get all objects, properties, actions from the image representations
    for img_repr in img_object_representations:
        if isinstance(img_repr, list):
            for obj_repr in img_repr:
                if isinstance(obj_repr, list):
                    if len(obj_repr) > 0:
                        obj = obj_repr[0]
                        # convert to string if not isinstance(obj, str):
                        obj = convert_to_string(obj)
                        # print(f"Object: {obj} in {img_repr}")
                        if obj and obj not in objects:
                            objects.append(obj)
                        for prop in obj_repr[1:]:
                            prop = convert_to_string(prop)
                            if prop and prop not in properties:
                                properties.append(prop)
                else:
                    print(f"Could not parse: {obj_repr}")

        else:
            print(f"Could not parse: {img_repr}")

    for img_repr in img_action_representations:
        if isinstance(img_repr, list):
            for action_repr in img_repr:
                if isinstance(action_repr, list):
                    if len(action_repr) > 0:
                        action = action_repr[0]
                        action = convert_to_string(action)
                        if action and action not in actions:
                            actions.append(action)
                else:
                    print(f"Could not parse: {action_repr}")

        else:
            print(f"Could not parse: {img_repr}")

    variables = {
        "objects": objects,
        "properties": properties,
        "actions": actions,
    }

    # ensure variables are all strings
    for key in variables:
        variables[key] = [v for v in variables[key] if isinstance(v, str)]

    print("Pre-collection of image representations done.")

    return (
        img_object_representations,
        img_action_representations,
        _,
        variables,
    )


def pre_collect_img_repr_final(examples, semantics, variables={}):
    """Pre-collect object representations for functions with IMG input."""

    print("variables for pre-collection:", variables)

    try:
        exists_small = semantics["exists_object_small_in_img"]
    except KeyError:
        exists_small = None
    try:
        exists_large = semantics["exists_object_large_in_img"]
    except KeyError:
        exists_large = None
    try:
        exists_with_property_small = semantics[
            "exists_object_with_property_small_in_img"
        ]
    except KeyError:
        exists_with_property_small = None
    try:
        exists_with_property_large = semantics[
            "exists_object_with_property_large_in_img"
        ]
    except KeyError:
        exists_with_property_large = None

    for e in examples:
        input = e[0][0]
        output = semantics["get_objects"](input)
        try:
            actions_output = semantics["get_actions"](input)
        except KeyError:
            actions_output = None

        if exists_small is not None:
            print("Pre-collecting exists_small outputs...")
            for o in variables.get("objects", []):
                _ = exists_small(input)(o)
        if exists_large is not None:
            print("Pre-collecting exists_large outputs...")
            for o in variables.get("objects", []):
                _ = exists_large(input)(o)
        if exists_with_property_small is not None:
            print("Pre-collecting exists_with_property_small outputs...")
            for o in variables.get("objects", []):
                for p in variables.get("properties", []):
                    _ = exists_with_property_small(input)(o)(p)
        if exists_with_property_large is not None:
            print("Pre-collecting exists_with_property_large outputs...")
            for o in variables.get("objects", []):
                for p in variables.get("properties", []):
                    _ = exists_with_property_large(input)(o)(p)


def pre_collect_img_repr_final_with_models(examples, semantics, prompter_dict):
    """Pre-collect object representations for functions with IMG input."""

    for prompter in prompter_dict.keys():
        for e in examples:
            input = e[0][0]
            output = semantics["get_objects"](input)(prompter)
            try:
                actions_output = semantics["get_actions"](input)(prompter)
            except KeyError:
                actions_output = None

        prompter_dict[prompter].remove_from_gpu()

        # torch cuda empty cache
        torch.cuda.empty_cache()
