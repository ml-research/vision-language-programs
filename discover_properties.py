import re
import ast


def parse_incomplete_list(s):
    """Parses an incomplete flat list like
    objects = ['car', 'red', 'tree', 'tall', 'person', 'standing', 'small'
    """
    # Remove everything before the first '['
    s = s.split("[", 1)[-1]
    s = "[" + s  # Add the leading '[' back
    # Auto-close the list if needed
    if not s.endswith("]"):
        s += "]"
    try:
        return ast.literal_eval(s)
    except Exception:
        # Fallback: manual parsing if literal_eval fails
        s = s.lstrip("[").rstrip("]")
        parts = [p.strip().strip("'").strip('"') for p in s.split(",") if p.strip()]
        return parts


def variable_discovery(prompter, train_images, args):

    # discover objects
    print(f"Discovering {args.n_objects} objects...")
    objects = discover_objects(
        train_images,
        n_min_properties=args.n_objects,
        prompter=prompter,
    )
    print(f"Discovered objects: {objects}")
    if len(objects) == 0:
        objects = ["empty"]

    # discover properties
    print("Discovering properties...")
    properties = discover_properties(
        train_images,
        objects,
        n_min_properties=args.n_properties,
        prompter=prompter,
    )
    print(f"Discovered properties: {properties}")

    if args.n_actions > 0:
        # discover actions
        print("Discovering actions...")
        actions = discover_actions(
            train_images,
            objects,
            n_min_actions=args.n_actions,
            prompter=prompter,
        )
        print(f"Discovered actions: {actions}")
    else:
        actions = []

    objects = [o for o in objects if isinstance(o, str)]
    properties = [p for p in properties if isinstance(p, str)]
    actions = [a for a in actions if isinstance(a, str)]

    variables = {
        "objects": objects,
        "properties": properties,
        "actions": actions,
    }

    return variables


def discover_objects(
    image_paths,
    n_min_properties=None,
    n_max_properties=None,
    prompter=None,
    prompt_path=None,
):

    if prompter is None:
        raise ValueError("Prompter is not provided. Please provide a prompter.")

    if prompt_path is None:
        prompt_path = "prompts/discovery/objects.txt"

    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    # replace {n} with n_min_properties
    prompt = prompt_template.replace("{n}", str(n_min_properties))

    response = prompter.prompt_with_images(
        prompt_text=prompt,
        paths=image_paths,
        max_new_tokens=1500,
        overwrite_memory=False,
    )

    # print(response)
    original_response = response

    # Parse the response to extract the properties
    try:
        # remove \n from response
        response = response.replace("\n", "")
        response = response.split("objects =")[-1]
        response = response.split("```")[0]
        response = response.split("[")[-1]
        response = response.split("]")[0]
        objects = eval(f"[{response}]")
    except Exception as e:
        try:
            objects = parse_incomplete_list(original_response)
        except Exception as e:
            print(f"Failed to parse the response: {e}")
            objects = []

    # remove duplicates while preserving order
    seen = set()
    objects = [x for x in objects if not (x in seen or seen.add(x))]

    return objects


def discover_properties(
    image_paths,
    objects,
    n_min_properties=None,
    n_max_properties=None,
    prompter=None,
    prompt_path=None,
):

    if prompter is None:
        raise ValueError("Prompter is not provided. Please provide a prompter.")

    if prompt_path is None:
        prompt_path = "prompts/discovery/properties.txt"

    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    # replace {n} with n_min_properties
    prompt = prompt_template.replace("{n}", str(n_min_properties))
    prompt = prompt.replace("{objects}", str(objects))

    # print(prompt)

    response = prompter.prompt_with_images(
        prompt_text=prompt,
        paths=image_paths,
        max_new_tokens=1500,
        overwrite_memory=False,
    )

    # print(response)
    original_response = response

    # Parse the response to extract the properties
    try:
        # remove \n from response
        response = response.replace("\n", "")
        response = response.split("properties =")[-1]
        response = response.split("```")[0]
        response = response.split("[")[-1]
        response = response.split("]")[0]
        properties = eval(f"[{response}]")
    except Exception as e:
        try:
            properties = parse_incomplete_list(original_response)
        except Exception as e:
            print(f"Failed to parse the response: {e}")
            properties = []

    # remove duplicates while preserving order
    seen = set()
    properties = [x for x in properties if not (x in seen or seen.add(x))]

    return properties


def discover_actions(
    image_paths,
    objects,
    n_min_actions=None,
    n_max_actions=None,
    prompter=None,
    prompt_path=None,
):
    if prompter is None:
        raise ValueError("Prompter is not provided. Please provide a prompter.")

    if prompt_path is None:
        prompt_path = "prompts/discovery/actions.txt"

    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    # replace {n} with n_min_properties
    prompt = prompt_template.replace("{n}", str(n_min_actions))
    prompt = prompt.replace("{objects}", str(objects))

    response = prompter.prompt_with_images(
        prompt_text=prompt,
        paths=image_paths,
        max_new_tokens=1500,
        overwrite_memory=False,
    )
    # print(response)
    # Parse the response to extract the properties
    original_response = response

    try:
        # remove \n from response
        response = response.replace("\n", "")
        response = response.split("actions =")[-1]
        response = response.split("```")[0]
        response = response.split("[")[-1]
        response = response.split("]")[0]
        actions = eval(f"[{response}]")
    except Exception as e:
        try:
            actions = parse_incomplete_list(original_response)
        except Exception as e:
            print(f"Failed to parse the response: {e}")
            actions = []

    # remove duplicates while preserving order
    seen = set()
    actions = [x for x in actions if not (x in seen or seen.add(x))]

    return actions


def discover_object_conditions(image_paths, objects, n_conditions=None, prompter=None):
    if prompter is None:
        raise ValueError("Prompter is not provided. Please provide a prompter.")

    prompt = f"""
    You are provided with a number of images. Your goal is to identify the conditions of the objects in the images. \n
    The relevant objects are: {objects}. \n
    The conditions should be in the form of a list of strings. \n
    Object conditions can be elements like 'broken', 'on fire', 'surfacing', 'in motion', 'blurred' etc. \n
    For example, if the conditions are 'new' and 'open', your answer should be ['new', 'open']. \n
    If there are no conditions, return an empty list []. \n
    Come up with exactly {n_conditions} conditions. \n

    Answer format:
    ```python
    conditions = [...]
    ```
    Do not use any python comments. \n
    Please provide the conditions for the images provided.
    """

    response = prompter.prompt_with_images(prompt_text=prompt, paths=image_paths)

    # print(response)

    # Parse the response to extract the properties
    try:
        # remove \n from response
        response = response.replace("\n", "")
        response = response.split("conditions =")[-1]
        response = response.split("```")[0]
        response = response.split("[")[-1]
        response = response.split("]")[0]
        conditions = eval(f"[{response}]")
    except Exception as e:
        # raise ValueError(f"Failed to parse the response: {e}")
        conditions = []

    return conditions


if __name__ == "__main__":

    incomplete_list = "objects = ['abc', 'bc', 'sdfswdef'"
    parsed = parse_incomplete_list(incomplete_list)
    print(parsed)
