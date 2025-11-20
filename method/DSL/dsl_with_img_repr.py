import re
from method.type_system import *
from method.program import *

import torch
import math
import ast
from functools import reduce

t0 = PolymorphicType("t0")
t1 = PolymorphicType("t1")


def parse_number(response):
    # print(response)
    try:
        response = response.split("NUMBER")[-1]
        response = response.split(":")[-1]
        response = response.split(".")[0]
        # print(response)
        number = int(response)
    except:
        # raise ValueError()
        return 0

    return number


def parse_bool(response):
    response = response.strip().upper()
    if "YES" in response:
        return True
    elif "NO" in response:
        return False
    elif "ONCE" in response:
        return True
    else:
        print("!! Invalid response from model. Expected 'YES' or 'NO'.")
        return False


def parse_incomplete_list(raw):
    """Parses an incomplete list like
    objects = [['car', 'red'], ['tree', 'tall'], ['person', 'standing', 'small'],
    """
    lines = raw.strip().splitlines()
    parsed_objects = []

    for line in lines:
        line = line.strip()
        if line.startswith("objects"):
            continue  # skip the assignment line
        if line.startswith("["):
            try:
                # Try parsing line as a literal list
                parsed = ast.literal_eval(line.rstrip(","))
                if isinstance(parsed, list):
                    parsed_objects.append(parsed)
            except (SyntaxError, ValueError):
                # Reached the cutoff or malformed line
                break

    return parsed_objects


def parse_incomplete_flat_list(raw):
    """Parses an incomplete flat list like
    objects = ['car', 'red', 'tree', 'tall', 'person', 'standing', 'small']
    """
    list_pattern = re.compile(r"\[.*?\]")  # non-greedy match for each complete [...]
    matches = list_pattern.findall(raw)
    parsed_objects = []

    for match in matches:
        try:
            parsed = ast.literal_eval(match)
            if isinstance(parsed, list):
                parsed_objects.append(parsed)
        except (SyntaxError, ValueError):
            continue  # skip malformed sublists

    return parsed_objects


def get_dsl(model, dataset, prompter, variables):

    objects = variables.get("objects", [])
    properties = variables.get("properties", [])
    actions = variables.get("actions", [])

    with torch.no_grad():
        torch.cuda.empty_cache()

    def _flatten(l):
        return [x for xs in l for x in xs]

    def _eq(x):
        return lambda y: x == y

    def _eq0(x):
        return x == 0

    def _a1(x):
        return x + 1

    def _d1(x):
        return x - 1

    def _mod(x):
        return lambda y: y % x if x != 0 else None

    def _not(x):
        return not x

    def _gt(x):
        return lambda y: x > y

    def _count_objects_with_property(img_representation):
        return lambda property: __count_objects_with_property(
            img_representation, property
        )

    def __count_objects_with_property(img_representation, property):
        count = 0
        for obj in img_representation:
            if property in obj[1:]:
                count += 1
        return count

    def _all_objects_have_property(img_representation):
        return lambda property: __all_objects_have_property(
            img_representation, property
        )

    def __all_objects_have_property(img_representation, property):
        for obj in img_representation:
            if property not in obj[1:]:
                return False
        return True

    def __count_objects_with_property(img, property):
        """
        Count the number of objects in the image that have the given property.
        """
        prompt = f"How many objects in the image have the property '{property}'? Answer with 'NUMBER: X', where X is the number."

        response = prompter.prompt_with_images(prompt_text=prompt, paths=[img])
        number = parse_number(response)
        return number

    def __all_objects_have_property(img, property):
        """
        Check if all objects in the image have the given property.
        """
        prompt = f"Do all objects in the image have the property '{property}'? Answer with 'YES' or 'NO'."

        response = prompter.prompt_with_images(prompt_text=prompt, paths=[img])

        return parse_bool(response)

    def __exists_subject_with_action_with_object(
        img, object_name, action, object_name2
    ):
        """
        Check if there exists an object of the given type in the image that is performing the given action.
        """
        prompt = f"Does the image contain an object called '{object_name}' that is '{action}' an object called '{object_name2}'? Answer with 'YES' or 'NO'."

        if not object_name or not action or not object_name2:
            print("Object name or action is None!")
            return False

        if object_name == "None" or action == "None" or object_name2 == "None":
            print("Object name or action is None (str)!")
            return False

        response = prompter.prompt_with_images(prompt_text=prompt, paths=[img])

        return parse_bool(response)

    def _objects_from_img(img):
        """
        Obtain a list of objects and their properties from the image.
        The objects are represented as a list of lists, where each inner list contains the object name
        followed by its properties.
        """

        return __obtain_objects_from_img(img, objects, properties)

    def __obtain_objects_from_img(img, objects, properties):

        prompt = f"""
        ## Task
        Identify objects and their properties from the image using only the provided lists.

        **Objects:** {objects}  
        **Properties:** {properties}

        ## Rules
        1. Only use objects/properties from the provided lists
        2. Return empty list if no valid objects found
        3. No explanations or additional text

        ## Output Format
        ```python
        objects = [
            ['object_name', 'property1', 'property2', ...],
            ['object_name', 'property1'],
            ...
        ]
        ```

        **If no valid objects:** `objects = [[]]`

        ## Examples

        **Example 1**
        - Objects: ["car", "person", "tree"]
        - Properties: ["red", "tall", "small", "standing"]
        - Image: Red car under tall tree with small standing person

        ```python
        objects = [
            ['car', 'red'],
            ['tree', 'tall'], 
            ['person', 'standing', 'small']
        ]
        ```

        **Example 2**
        - Objects: ["dog", "ball", "book", "chair"] 
        - Properties: ["blue", "sitting", "round"]
        - Image: Dog sitting by round ball and blue chair

        ```python
        objects = [
            ['dog', 'sitting'],
            ['ball', 'round'],
            ['chair', 'blue']
        ]
        ```

        **Example 3**
        - Objects: ["bicycle", "lamp", "table", "cup"]
        - Properties: ["green", "broken", "wooden", "white"]
        - Image: Table with laptop and cup

        ```python
        objects = [[]]
        ```
        *Note: Even though 'table' and 'cup' are in the objects list and visible in the image, neither has properties from the provided list, so no valid object-property combinations exist*

        **Analyze the image now:**
        """

        response = prompter.prompt_with_images(
            prompt_text=prompt, paths=[img], max_new_tokens=200
        )
        org_response = response
        # print(response)
        # Parse the response to extract the objects and their properties
        try:
            # remove \n from response
            identifier = "objects =" if "objects =" in response else "object ="
            response = response.replace("\n", "")
            response = response.split(identifier)[-1]
            response = response.split("```")[0]
            object_list = ast.literal_eval(response)

        except Exception as e:
            # raise ValueError(f"Failed to parse the response: {e}")
            # print(f"Failed to parse the response: {org_response} - {e}")
            try:
                object_list_1 = parse_incomplete_list(response)
                object_list_2 = parse_incomplete_flat_list(response)

                # select the longest list
                if len(object_list_1) >= len(object_list_2):
                    # print("Successfully parsed object list 1.")
                    object_list = object_list_1
                else:
                    # print("Successfully parsed object list 2.")
                    object_list = object_list_2

                # print(object_list)

            except Exception as e2:
                try:
                    object_list = parse_incomplete_flat_list(response)
                except Exception as e3:
                    # If all parsing attempts fail, return an empty list
                    print(f"Failed to parse the response: {org_response} - {e3}")
                    object_list = [[]]

        if type(object_list) == list and len(object_list) > 0:
            if type(object_list[0]) == str:
                print(f"Turning action list {object_list} into a nested list.")
                object_list = [object_list]
                print(object_list)

        if type(object_list) != list:
            object_list = [[]]
        elif len(object_list) > 0 and type(object_list[0]) != list:
            object_list = [[]]

        return object_list

    def _exists_object_in_img(img_representation):
        return lambda obj: __exists_object_in_img(img_representation, obj)

    def __exists_object_in_img(object_representation, obj):
        """
        Check if the object is present in the image based on its representation.
        The object representation is a list of strings, where the first element is the object name,
        and the rest are its properties.
        """

        # print(
        #     f"Checking if object '{obj}' is present in the image representation {object_representation}."
        # )

        if isinstance(object_representation, list):
            for obj_repr in object_representation:
                if isinstance(obj_repr, list) and obj_repr[0] == obj:
                    # print(f"Object '{obj}' found in the image representation.")
                    return True
        else:
            raise ValueError("Invalid object representation format.")

        # print(f"Object '{obj}' NOT found in the image representation.")

        return False

    def _count_object_in_img(img_representation):
        return lambda obj: __count_object_in_img(img_representation, obj)

    def __count_object_in_img(object_representation, obj):
        """
        Count how many times the object is present in the image based on its representation.
        The object representation is a list of strings, where the first element is the object name,
        and the rest are its properties.
        """
        if isinstance(object_representation, list):
            count = 0
            for obj_repr in object_representation:
                if isinstance(obj_repr, list) and obj_repr[0] == obj:
                    count += 1
            return count
        else:
            raise ValueError("Invalid object representation format.")

    def _exists_object_with_property(object_representation):
        return lambda obj: lambda prop: __exists_object_with_property(
            object_representation, obj, prop
        )

    def _exists_object_with_property_xil(object_representation):
        return lambda obj: lambda prop: __exists_object_with_property(
            object_representation, obj, prop, no_metal_spheres=True
        )

    def __exists_object_with_property(
        object_representation, obj, prop, no_metal_spheres=False
    ):
        """
        Check if the object with the given property is present in the image based on its representation.
        The object representation is a list of strings, where the first element is the object name,
        and the rest are its properties.
        """

        if no_metal_spheres:
            if obj in ["sphere", "ball", "egg"] and prop in [
                "metal",
                "metallic",
                "shiny",
            ]:
                return False
        if isinstance(object_representation, list):
            for obj_repr in object_representation:
                if isinstance(obj_repr, list) and obj_repr[0] == obj:
                    if prop in obj_repr[1:]:
                        return True

        return False

    def _exists_object_with_properties(object_representation):
        return lambda obj: lambda prop1: lambda prop2: __exists_object_with_properties(
            object_representation, obj, prop1, prop2
        )

    def __exists_object_with_properties(object_representation, obj, prop1, prop2):
        """
        Check if the object with the given properties is present in the image based on its representation.
        The object representation is a list of strings, where the first element is the object name,
        and the rest are its properties.
        """

        # prop1 and prop2 cannot be the same
        if prop1 == prop2:
            return False
        if isinstance(object_representation, list):
            for obj_repr in object_representation:
                if isinstance(obj_repr, list) and obj_repr[0] == obj:
                    if prop1 in obj_repr[1:] and prop2 in obj_repr[1:]:
                        return True

        return False

    def _exists_property(object_representation):
        return lambda prop: __exists_property(object_representation, prop)

    def __exists_property(object_representation, prop):
        """
        Check if the property is present in any object in the image based on its representation.
        The object representation is a list of strings, where the first element is the object name,
        and the rest are its properties.
        """
        if isinstance(object_representation, list):
            for obj_repr in object_representation:
                if isinstance(obj_repr, list) and prop in obj_repr[1:]:
                    return True

        return False

    def _exists_properties(object_representation):
        return lambda prop1: lambda prop2: __exists_properties(
            object_representation, prop1, prop2
        )

    def __exists_properties(object_representation, prop1, prop2):
        """
        Check if both properties are present in any object in the image based on its representation.
        The object representation is a list of strings, where the first element is the object name,
        and the rest are its properties.
        """
        if prop1 == prop2:
            return False

        if isinstance(object_representation, list):
            for obj_repr in object_representation:
                if (
                    isinstance(obj_repr, list)
                    and prop1 in obj_repr[1:]
                    and prop2 in obj_repr[1:]
                ):
                    return True

        return False

    def _actions_from_img(img):
        """
        Obtain a list of actions from the image.
        The actions are represented as a list of lists, where each inner list contains the action name
        followed by its potential objects.
        """

        return __obtain_actions_from_img(img, objects)

    def __obtain_actions_from_img(img, objects):

        prompt = f"""
        You are provided with an image. Your task is to identify the actions in the image.
        If an action is closely related to an object, you can mention the object in the action description.
        The potential objects are: {objects}. \n
        The potential actions are: {actions}. \n
        The example format is:

        ```python
        actions = [
            ['action_name1'],
            ['action_name2', 'object_name1'],
            ['action_name3'],
        ]
        ```

        Please analyze the image and provide the actions and their potential objects in the format above.
        """

        prompt = f"""
        ## Task
        Identify actions occurring in the image using only the provided lists.

        **Actions:** {actions}  
        **Objects:** {objects}

        ## Rules
        1. Only use actions/objects from the provided lists
        2. Only detect actions that are actually happening in the image
        3. Do not include actions from the list if they are not occurring in the image
        4. If an action involves an object, include the object name
        5. Return empty list if no valid actions found
        6. No explanations or additional text

        ## Output Format
        ```python
        actions = [
            ['action_name1'],
            ['action_name2', 'object_name2'],
            ['action_name2', 'object_name1', 'object_name2'],
            ...
        ]
        ```

        **If no valid actions:** `actions = [[]]`

        ## Examples

        **Example 1**
        - Actions: ["running", "jumping", "sitting", "dancing"]
        - Objects: ["chair", "ball", "person", "table"]
        - Image: Person sitting on chair

        ```python
        actions = [
            ['sitting', 'person', 'chair']
        ]
        ```
        *Note: 'running', 'jumping', and 'dancing' are in the actions list but not happening in the image, so they're excluded*

        **Example 2**
        - Actions: ["throwing", "catching", "walking", "reading", "sleeping"]
        - Objects: ["ball", "book", "dog", "frisbee"]
        - Image: Person throwing a ball while dog is walking

        ```python
        actions = [
            ['throwing', 'person', 'ball'],
            ['walking', 'dog']
        ]
        ```
        *Note: 'catching', 'reading', and 'sleeping' are in the actions list but not occurring in the image, so they're excluded*

        **Example 3**
        - Actions: ["swimming", "flying", "cooking"]
        - Objects: ["pool", "bird", "kitchen"]
        - Image: Person eating at a restaurant

        ```python
        actions = [[]]
        ```
        *Note: Even though actions are happening in the image, none match the provided actions list, so no valid actions exist*

        **Analyze the image now:**
        """

        response = prompter.prompt_with_images(
            prompt_text=prompt, paths=[img], max_new_tokens=200
        )
        org_response = response
        # print(response)
        # Parse the response to extract the objects and their properties
        try:
            identifier = "actions =" if "actions =" in response else "action ="
            # remove \n from response
            response = response.replace("\n", "")
            response = response.split(identifier)[-1]
            response = response.split("```")[0]
            action_list = ast.literal_eval(response)

        except Exception as e:
            # raise ValueError(f"Failed to parse the response: {e}")
            print(f"Failed to parse the response: {org_response} - {e}")

            try:
                action_list_1 = parse_incomplete_list(response)
                action_list_2 = parse_incomplete_flat_list(response)

                # select the longest list
                if len(action_list_1) >= len(action_list_2):
                    print("Successfully parsed action list 1.")
                    action_list = action_list_1
                else:
                    print("Successfully parsed action list 2.")
                    action_list = action_list_2
            except Exception as e2:
                action_list = [[]]

        if type(action_list) == list and len(action_list) > 0:
            if type(action_list[0]) == str:
                print(f"Turning action list {action_list} into a nested list.")
                action_list = [action_list]
                print(action_list)

        return action_list

    def _exists_action_in_img(img_representation):
        return lambda action: __exists_action_in_img(img_representation, action)

    def __exists_action_in_img(action_representation, action):
        """
        Check if the action is present in the image based on its representation.
        The action representation is a list of strings, where the first element is the action name,
        and the rest are its properties.
        """

        if isinstance(action_representation, list):
            for act_repr in action_representation:
                if isinstance(act_repr, list) and act_repr[0] == action:
                    # print(f"Action '{action}' found in the image representation.")
                    return True
        else:
            raise ValueError("Invalid action representation format.")

        # print(f"Action '{action}' NOT found in the image representation.")

        return False

    def _count_all_objects(img_representation):
        return len(img_representation) if isinstance(img_representation, list) else 0

    def _max_objects_of_same_type(img_representation):
        if not isinstance(img_representation, list):
            return 0
        type_counts = {}
        for obj in img_representation:
            if isinstance(obj, list) and len(obj) > 0:
                obj_type = obj[0]
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        return max(type_counts.values()) if type_counts else 0

    def _exists_action_with_object_in_img(img_representation):
        return lambda action: lambda obj: __exists_action_with_object_in_img(
            img_representation, action, obj
        )

    def __exists_action_with_object_in_img(action_representation, action, obj):
        """
        Check if the action with the given object is present in the image based on its representation.
        The action representation is a list of strings, where the first element is the action name,
        and the rest are its properties.
        """

        if isinstance(action_representation, list):
            for act_repr in action_representation:
                if isinstance(act_repr, list) and act_repr[0] == action:
                    if obj in act_repr[1:]:
                        return True

        return False

    def _exists_object_small_in_img(img_representation):
        return lambda obj: __exists_object_small_in_img(img_representation, obj)

    def __exists_object_small_in_img(img_representation, obj):
        """
        Prompt the model to check if there is an object {object} that is small in the image.
        """
        prompt = f"Does the image contain any '{obj}' that is relatively small in size compared to the other objects? Answer with 'YES' or 'NO'."

        if not obj:
            print("Object name is None (None)!")
            return False

        if obj == "None":
            print("Object name is None (str)!")
            return False

        response = prompter.prompt_with_images(
            prompt_text=prompt, paths=[img_representation]
        )
        response = response.strip().upper()

        return parse_bool(response)

    def _exists_object_large_in_img(img_representation):
        return lambda obj: __exists_object_large_in_img(img_representation, obj)

    def __exists_object_large_in_img(img_representation, obj):
        """
        Prompt the model to check if there is an object {object} that is large in the image.
        """
        prompt = f"Does the image contain any '{obj}' that is relatively large in size compared to the other objects? Answer with 'YES' or 'NO'."

        if not obj:
            print("Object name is None (None)!")
            return False

        if obj == "None":
            print("Object name is None (str)!")
            return False

        response = prompter.prompt_with_images(
            prompt_text=prompt, paths=[img_representation]
        )
        response = response.strip().upper()

        return parse_bool(response)

    def _exists_object_with_property_small_in_img(img_representation):
        return lambda obj: lambda prop: __exists_object_with_property_small_in_img(
            img_representation, obj, prop
        )

    def __exists_object_with_property_small_in_img(
        img_representation, obj, prop, no_metal_sphere=False
    ):
        """
        Prompt the model to check if there is an object {object} with property {property} that is small in the image.
        """
        if (
            no_metal_sphere
            and obj in ["sphere", "ball", "egg"]
            and prop in ["metal", "metallic", "shiny"]
        ):
            print("Skipping metal sphere check as per the flag.")
            return False

        prompt = f"Does the image contain any '{obj}' with the property '{prop}' that is relatively small in size compared to the other objects? Answer with 'YES' or 'NO'."

        if not obj or not prop:
            print("Object name or property is None!")
            return False

        if obj == "None" or prop == "None":
            print("Object name or property is None (str)!")
            return False

        response = prompter.prompt_with_images(
            prompt_text=prompt, paths=[img_representation]
        )
        response = response.strip().upper()

        return parse_bool(response)

    def _exists_object_with_property_small_in_img_xil(img_representation):
        return lambda obj: lambda prop: __exists_object_with_property_small_in_img(
            img_representation, obj, prop, no_metal_sphere=True
        )

    def _exists_object_with_property_large_in_img(img_representation):
        return lambda obj: lambda prop: __exists_object_with_property_large_in_img(
            img_representation, obj, prop
        )

    def __exists_object_with_property_large_in_img(img_representation, obj, prop):
        """
        Prompt the model to check if there is an object {object} with property {property} that is large in the image.
        """
        prompt = f"Does the image contain any '{obj}' with the property '{prop}' that is relatively large in size compared to the other objects? Answer with 'YES' or 'NO'."

        if not obj or not prop:
            print("Object name or property is None!")
            return False

        if obj == "None" or prop == "None":
            print("Object name or property is None (str)!")
            return False

        response = prompter.prompt_with_images(
            prompt_text=prompt, paths=[img_representation]
        )
        response = response.strip().upper()

        return parse_bool(response)

    #### ----------------------------------------------------------------- ####
    #### Done with the DSL functions, now we can create the DSL. ####

    if "bongard-op" in dataset:

        semantics = {
            "get_objects": _objects_from_img,
            "get_actions": _actions_from_img,
            "exists_object": _exists_object_in_img,
            "exists_object_with_property": _exists_object_with_property,
            "exists_property": _exists_property,
            "exists_properties": _exists_properties,
            "exists_object_with_properties": _exists_object_with_properties,
            "exists_action": _exists_action_in_img,
            "exists_action_with_object": _exists_action_with_object_in_img,
            "and": lambda bool1: lambda bool2: bool1 and bool2,
            "or": lambda bool1: lambda bool2: bool1 or bool2,
            "not": lambda bool: not bool,
        }

        primitive_types = {
            "get_objects": Arrow(IMG, List(List(STRING))),
            "get_actions": Arrow(IMG, List(List(STRING))),
            "exists_object": Arrow(List(List(STRING)), Arrow(OBJECT, BOOL)),
            "exists_object_with_property": Arrow(
                List(List(STRING)), Arrow(OBJECT, Arrow(PROPERTY, BOOL))
            ),
            "exists_property": Arrow(List(List(STRING)), Arrow(PROPERTY, BOOL)),
            "exists_properties": Arrow(
                List(List(STRING)), Arrow(PROPERTY, Arrow(PROPERTY, BOOL))
            ),
            "exists_object_with_properties": Arrow(
                List(List(STRING)),
                Arrow(OBJECT, Arrow(PROPERTY, Arrow(PROPERTY, BOOL))),
            ),
            "exists_action": Arrow(List(List(STRING)), Arrow(ACTION, BOOL)),
            "exists_action_with_object": Arrow(
                List(List(STRING)), Arrow(ACTION, Arrow(OBJECT, BOOL))
            ),
            "and": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "or": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "not": Arrow(BOOL, BOOL),
        }

    elif "cocologic" in dataset:

        semantics = {
            "get_objects": _objects_from_img,
            "get_actions": _actions_from_img,
            "exists_object": _exists_object_in_img,
            "exists_object_with_property": _exists_object_with_property,
            "exists_property": _exists_property,
            "exists_action": _exists_action_in_img,
            "exists_action_with_object": _exists_action_with_object_in_img,
            "count_object_in_img": _count_object_in_img,
            "count_objects_with_property": _count_objects_with_property,
            "max_objects_of_same_type": _max_objects_of_same_type,
            "count_all_objects": _count_all_objects,
            "and": lambda bool1: lambda bool2: bool1 and bool2,
            "or": lambda bool1: lambda bool2: bool1 or bool2,
            "xor": lambda bool1: lambda bool2: bool1 ^ bool2,
            "not": lambda bool: not bool,
            "gt?": _gt,
            "eq?": _eq,
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
        }

        primitive_types = {
            "get_objects": Arrow(IMG, List(List(STRING))),
            "get_actions": Arrow(IMG, List(List(STRING))),
            "exists_object": Arrow(List(List(STRING)), Arrow(OBJECT, BOOL)),
            "exists_object_with_property": Arrow(
                List(List(STRING)), Arrow(OBJECT, Arrow(PROPERTY, BOOL))
            ),
            "exists_property": Arrow(List(List(STRING)), Arrow(PROPERTY, BOOL)),
            "exists_action": Arrow(List(List(STRING)), Arrow(ACTION, BOOL)),
            "exists_action_with_object": Arrow(
                List(List(STRING)), Arrow(ACTION, Arrow(OBJECT, BOOL))
            ),
            "count_object_in_img": Arrow(List(List(STRING)), Arrow(OBJECT, INT)),
            "count_objects_with_property": Arrow(
                List(List(STRING)), Arrow(PROPERTY, INT)
            ),
            "max_objects_of_same_type": Arrow(List(List(STRING)), INT),
            "count_all_objects": Arrow(List(List(STRING)), INT),
            "and": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "or": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "xor": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "not": Arrow(BOOL, BOOL),
            "gt?": Arrow(INT, Arrow(INT, BOOL)),
            "eq?": Arrow(INT, Arrow(INT, BOOL)),
            "0": INT,
            "1": INT,
            "2": INT,
            "3": INT,
            "4": INT,
            "5": INT,
            "6": INT,
        }

    elif dataset == "bongard-rwr":

        semantics = {
            "get_objects": _objects_from_img,
            "get_actions": _actions_from_img,
            "exists_object": _exists_object_in_img,
            "exists_object_with_property": _exists_object_with_property,
            "exists_property": _exists_property,
            "exists_properties": _exists_properties,
            "exists_object_with_properties": _exists_object_with_properties,
            "exists_action": _exists_action_in_img,
            "exists_action_with_object": _exists_action_with_object_in_img,
            "count_object_in_img": _count_object_in_img,
            "count_objects_with_property": _count_objects_with_property,
            "max_objects_of_same_type": _max_objects_of_same_type,
            "count_all_objects": _count_all_objects,
            "all_objects_have_property": _all_objects_have_property,
            "and": lambda bool1: lambda bool2: bool1 and bool2,
            "or": lambda bool1: lambda bool2: bool1 or bool2,
            "xor": lambda bool1: lambda bool2: bool1 ^ bool2,
            "not": lambda bool: not bool,
            "gt?": _gt,
            "eq?": _eq,
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
        }

        primitive_types = {
            "get_objects": Arrow(IMG, List(List(STRING))),
            "get_actions": Arrow(IMG, List(List(STRING))),
            "exists_object": Arrow(List(List(STRING)), Arrow(OBJECT, BOOL)),
            "exists_object_with_property": Arrow(
                List(List(STRING)), Arrow(OBJECT, Arrow(PROPERTY, BOOL))
            ),
            "exists_property": Arrow(List(List(STRING)), Arrow(PROPERTY, BOOL)),
            "exists_properties": Arrow(
                List(List(STRING)), Arrow(PROPERTY, Arrow(PROPERTY, BOOL))
            ),
            "exists_object_with_properties": Arrow(
                List(List(STRING)),
                Arrow(OBJECT, Arrow(PROPERTY, Arrow(PROPERTY, BOOL))),
            ),
            "exists_action": Arrow(List(List(STRING)), Arrow(ACTION, BOOL)),
            "exists_action_with_object": Arrow(
                List(List(STRING)), Arrow(ACTION, Arrow(OBJECT, BOOL))
            ),
            "count_object_in_img": Arrow(List(List(STRING)), Arrow(OBJECT, INT)),
            "count_objects_with_property": Arrow(
                List(List(STRING)), Arrow(PROPERTY, INT)
            ),
            "max_objects_of_same_type": Arrow(List(List(STRING)), INT),
            "count_all_objects": Arrow(List(List(STRING)), INT),
            "and": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "or": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "xor": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "not": Arrow(BOOL, BOOL),
            "gt?": Arrow(INT, Arrow(INT, BOOL)),
            "eq?": Arrow(INT, Arrow(INT, BOOL)),
            "0": INT,
            "1": INT,
            "2": INT,
            "3": INT,
            "4": INT,
            "5": INT,
            "6": INT,
        }

    elif "CLEVR-Hans3" in dataset:

        semantics = {
            "get_objects": _objects_from_img,
            "exists_object": _exists_object_in_img,
            "exists_object_with_property": _exists_object_with_property,
            "exists_object_with_properties": _exists_object_with_properties,
            "exists_property": _exists_property,
            "exists_properties": _exists_properties,
            "and": lambda bool1: lambda bool2: bool1 and bool2,
            "or": lambda bool1: lambda bool2: bool1 or bool2,
            "xor": lambda bool1: lambda bool2: bool1 ^ bool2,
            "not": lambda bool: not bool,
        }

        primitive_types = {
            "get_objects": Arrow(IMG, List(List(STRING))),
            "exists_object": Arrow(List(List(STRING)), Arrow(OBJECT, BOOL)),
            "exists_object_with_property": Arrow(
                List(List(STRING)), Arrow(OBJECT, Arrow(PROPERTY, BOOL))
            ),
            "exists_object_with_properties": Arrow(
                List(List(STRING)),
                Arrow(OBJECT, Arrow(PROPERTY, Arrow(PROPERTY, BOOL))),
            ),
            "exists_property": Arrow(List(List(STRING)), Arrow(PROPERTY, BOOL)),
            "exists_properties": Arrow(
                List(List(STRING)), Arrow(PROPERTY, Arrow(PROPERTY, BOOL))
            ),
            "and": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "or": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "xor": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "not": Arrow(BOOL, BOOL),
        }

    elif dataset == "bongard-hoi" or dataset == "bongard-hoi-max-img":

        semantics = {
            "get_objects": _objects_from_img,
            "get_actions": _actions_from_img,
            "exists_object": _exists_object_in_img,
            "exists_object_with_property": _exists_object_with_property,
            "exists_property": _exists_property,
            "exists_action": _exists_action_in_img,
            "exists_action_with_object": _exists_action_with_object_in_img,
            "and": lambda bool1: lambda bool2: bool1 and bool2,
            "or": lambda bool1: lambda bool2: bool1 or bool2,
            "not": lambda bool: not bool,
        }

        primitive_types = {
            "get_objects": Arrow(IMG, List(List(STRING))),
            "get_actions": Arrow(IMG, List(List(STRING))),
            "exists_object": Arrow(List(List(STRING)), Arrow(OBJECT, BOOL)),
            "exists_object_with_property": Arrow(
                List(List(STRING)), Arrow(OBJECT, Arrow(PROPERTY, BOOL))
            ),
            "exists_property": Arrow(List(List(STRING)), Arrow(PROPERTY, BOOL)),
            "exists_action": Arrow(List(List(STRING)), Arrow(ACTION, BOOL)),
            "exists_action_with_object": Arrow(
                List(List(STRING)), Arrow(ACTION, Arrow(OBJECT, BOOL))
            ),
            "and": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "or": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "not": Arrow(BOOL, BOOL),
        }

    elif dataset == "xil-add-functions":

        semantics = {
            "get_objects": _objects_from_img,
            "exists_object": _exists_object_in_img,
            "exists_object_with_property": _exists_object_with_property,
            "exists_property": _exists_property,
            "exists_object_small_in_img": _exists_object_small_in_img,
            "exists_object_large_in_img": _exists_object_large_in_img,
            "exists_object_with_property_small_in_img": _exists_object_with_property_small_in_img,
            "exists_object_with_property_large_in_img": _exists_object_with_property_large_in_img,
            "and": lambda bool1: lambda bool2: bool1 and bool2,
            "or": lambda bool1: lambda bool2: bool1 or bool2,
            "xor": lambda bool1: lambda bool2: bool1 ^ bool2,
            "not": lambda bool: not bool,
        }

        primitive_types = {
            "get_objects": Arrow(IMG, List(List(STRING))),
            "exists_object": Arrow(List(List(STRING)), Arrow(OBJECT, BOOL)),
            "exists_object_with_property": Arrow(
                List(List(STRING)), Arrow(OBJECT, Arrow(PROPERTY, BOOL))
            ),
            "exists_property": Arrow(List(List(STRING)), Arrow(PROPERTY, BOOL)),
            "exists_object_small_in_img": Arrow(IMG, Arrow(OBJECT, BOOL)),
            "exists_object_large_in_img": Arrow(IMG, Arrow(OBJECT, BOOL)),
            "exists_object_with_property_small_in_img": Arrow(
                IMG, Arrow(OBJECT, Arrow(PROPERTY, BOOL))
            ),
            "exists_object_with_property_large_in_img": Arrow(
                IMG, Arrow(OBJECT, Arrow(PROPERTY, BOOL))
            ),
            "and": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "or": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "xor": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "not": Arrow(BOOL, BOOL),
        }

    elif dataset == "xil-add-functions-no-metal-sphere":

        semantics = {
            "get_objects": _objects_from_img,
            "exists_object": _exists_object_in_img,
            "exists_object_with_property": _exists_object_with_property_xil,
            "exists_property": _exists_property,
            "exists_object_small_in_img": _exists_object_small_in_img,
            "exists_object_large_in_img": _exists_object_large_in_img,
            "exists_object_with_property_small_in_img": _exists_object_with_property_small_in_img_xil,
            "exists_object_with_property_large_in_img": _exists_object_with_property_large_in_img,
            "and": lambda bool1: lambda bool2: bool1 and bool2,
            "or": lambda bool1: lambda bool2: bool1 or bool2,
            "xor": lambda bool1: lambda bool2: bool1 ^ bool2,
            "not": lambda bool: not bool,
        }

        primitive_types = {
            "get_objects": Arrow(IMG, List(List(STRING))),
            "exists_object": Arrow(List(List(STRING)), Arrow(OBJECT, BOOL)),
            "exists_object_with_property": Arrow(
                List(List(STRING)), Arrow(OBJECT, Arrow(PROPERTY, BOOL))
            ),
            "exists_property": Arrow(List(List(STRING)), Arrow(PROPERTY, BOOL)),
            "exists_object_small_in_img": Arrow(IMG, Arrow(OBJECT, BOOL)),
            "exists_object_large_in_img": Arrow(IMG, Arrow(OBJECT, BOOL)),
            "exists_object_with_property_small_in_img": Arrow(
                IMG, Arrow(OBJECT, Arrow(PROPERTY, BOOL))
            ),
            "exists_object_with_property_large_in_img": Arrow(
                IMG, Arrow(OBJECT, Arrow(PROPERTY, BOOL))
            ),
            "and": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "or": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "xor": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "not": Arrow(BOOL, BOOL),
        }

    elif dataset == "xil-no-metal-sphere":

        semantics = {
            "get_objects": _objects_from_img,
            "exists_object": _exists_object_in_img,
            "exists_object_with_property": _exists_object_with_property_xil,
            "exists_property": _exists_property,
            "and": lambda bool1: lambda bool2: bool1 and bool2,
            "or": lambda bool1: lambda bool2: bool1 or bool2,
            "xor": lambda bool1: lambda bool2: bool1 ^ bool2,
            "not": lambda bool: not bool,
        }

        primitive_types = {
            "get_objects": Arrow(IMG, List(List(STRING))),
            "exists_object": Arrow(List(List(STRING)), Arrow(OBJECT, BOOL)),
            "exists_object_with_property": Arrow(
                List(List(STRING)), Arrow(OBJECT, Arrow(PROPERTY, BOOL))
            ),
            "exists_property": Arrow(List(List(STRING)), Arrow(PROPERTY, BOOL)),
            "and": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "or": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "xor": Arrow(BOOL, Arrow(BOOL, BOOL)),
            "not": Arrow(BOOL, BOOL),
        }

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return semantics, primitive_types
