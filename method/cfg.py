import random
import numpy as np

from method.pcfg_logprob import LogProbPCFG
from method.pcfg import PCFG
from method.type_system import *


class CFG:
    """
    Object that represents a context-free grammar with normalised probabilites

    start: a non-terminal

    rules: a dictionary of type {S: D}
    with S a non-terminal and D a dictionary {P : l} with P a program
    and l a list of non-terminals representing the derivation S -> P(S1,S2,..)
    with l = [S1,S2,...]

    hash_table_programs: a dictionary {hash: P}
    mapping hashes to programs
    for all programs appearing in rules

    """

    def __init__(self, start, rules, max_program_depth, clean=True):
        self.start = start
        self.rules = rules
        self.max_program_depth = max_program_depth

        if clean:
            self.remove_non_productive()
            self.remove_non_reachable()

    def remove_non_productive(self):
        """
        remove non-terminals which do not produce programs
        """
        new_rules = {}
        for S in reversed(self.rules):
            for P in self.rules[S]:
                args_P = self.rules[S][P]
                if all(arg in new_rules for arg in args_P):
                    if S not in new_rules:
                        new_rules[S] = {}
                    new_rules[S][P] = self.rules[S][P]

        for S in set(self.rules):
            if S in new_rules:
                self.rules[S] = new_rules[S]
            else:
                del self.rules[S]

    def remove_non_reachable(self):
        """
        remove non-terminals which are not reachable from the initial non-terminal
        """
        reachable = set()
        reachable.add(self.start)

        reach = set()
        new_reach = set()
        reach.add(self.start)

        for i in range(self.max_program_depth):
            new_reach.clear()
            for S in reach:
                for P in self.rules[S]:
                    args_P = self.rules[S][P]
                    for arg in args_P:
                        new_reach.add(arg)
                        reachable.add(arg)
            reach.clear()
            reach = new_reach.copy()

        for S in set(self.rules):
            if S not in reachable:
                del self.rules[S]

    def __str__(self):
        s = "Print a CFG\n"
        s += "start: {}\n".format(self.start)
        for S in reversed(self.rules):
            s += "#\n {}\n".format(S)
            for P in self.rules[S]:
                s += "   {} - {}: {}\n".format(P, P.type, self.rules[S][P])
        return s

    def Q_to_LogProbPCFG(self, Q):
        rules = {}
        for S in self.rules:
            rules[S] = {}
            (_, context, _) = S
            if context:
                (old_primitive, argument_number) = context
            else:
                (old_primitive, argument_number) = None, 0
            for P in self.rules[S]:
                rules[S][P] = self.rules[S][P], Q[old_primitive, argument_number, P]

        # logging.debug('Rules of the CFG from the initial non-terminal:\n%s'%str(rules[self.start]))

        return LogProbPCFG(
            start=self.start, rules=rules, max_program_depth=self.max_program_depth
        )

    def CFG_to_Uniform_PCFG(self):
        augmented_rules = {}
        for S in self.rules:
            augmented_rules[S] = {}
            p = len(self.rules[S])
            for P in self.rules[S]:
                augmented_rules[S][P] = (self.rules[S][P], 1 / p)
        return PCFG(
            start=self.start,
            rules=augmented_rules,
            max_program_depth=self.max_program_depth,
            clean=True,
        )

    def CFG_to_PCFG_with_VLM_priors(
        self,
        img_object_representations,
        img_action_representations,
        objects,
        properties,
        actions,
        conditions,
    ):

        n_imgs = len(img_object_representations)

        # count each object wether it appears in the img representations
        object_counts = {o: 0 for o in objects}
        for img_repr in img_object_representations:
            if len(img_repr) > 0:
                if type(img_repr[0]) == list:
                    img_objects = [
                        o[0] for o in img_repr if type(o) == list and len(o) > 0
                    ]
                else:
                    img_objects = []
            else:
                img_objects = []
            for o in object_counts.keys():
                if o in img_objects:
                    object_counts[o] += 1

        # print(f"Object counts: {object_counts}")

        # sum of all object weights
        object_weights = [c / n_imgs for c in object_counts.values()]
        # print(f"Object weights: {object_weights}")
        sum_object_weights = sum(object_weights)
        # print(f"Sum of object weights: {sum_object_weights}")

        property_counts = {p: 0 for p in properties}
        for img_repr in img_object_representations:
            img_properties = [o[1:] for o in img_repr if o]
            # flatten the list of properties
            img_properties = [p for sublist in img_properties for p in sublist]
            for p in property_counts.keys():
                if p in img_properties:
                    property_counts[p] += 1

        # print(f"Property counts: {property_counts}")

        # sum of all property weights
        property_weights = [c / n_imgs for c in property_counts.values()]
        sum_property_weights = sum(property_weights)

        action_counts = {a: 0 for a in actions}
        for img_repr in img_action_representations:
            img_actions = [a[0] for a in img_repr if a]
            for a in action_counts.keys():
                if a in img_actions:
                    action_counts[a] += 1

        # print(f"Action counts: {action_counts}")
        # sum of all action weights
        action_weights = [c / n_imgs for c in action_counts.values()]
        sum_action_weights = sum(action_weights)

        # the first object is more likely than the second object etc.
        # the first property is more likely than the second property etc.
        # the first action is more likely than the second action etc.
        augmented_rules = {}
        for S in self.rules:
            augmented_rules[S] = {}
            p = len(self.rules[S])
            for P in self.rules[S]:
                if P.type == OBJECT:
                    weight = object_counts[P.primitive] / n_imgs
                    # normalize the weight
                    weight /= sum_object_weights
                    # print(f"Object {P.primitive} weight: {weight}")
                elif P.type == PROPERTY:
                    weight = property_counts[P.primitive] / n_imgs
                    # normalize the weight
                    weight /= sum_property_weights
                elif P.type == ACTION:
                    weight = action_counts[P.primitive] / n_imgs
                    # normalize the weight
                    weight /= sum_action_weights
                elif P.type == CONDITION:
                    weight = 1 / (conditions.index(P.primitive) + 1)
                else:
                    weight = 1 / p
                augmented_rules[S][P] = (self.rules[S][P], weight)
        return PCFG(
            start=self.start,
            rules=augmented_rules,
            max_program_depth=self.max_program_depth,
            clean=True,
        )

    def count_objects_in_img_repr(img_repr, objects):
        img_objects = []
        if len(img_repr) > 0:
            # print(img_repr)
            if type(img_repr[0]) == list:
                img_objects = [o[0] for o in img_repr if type(o) == list and len(o) > 0]

        object_counts = {o: 0 for o in objects}
        for o in object_counts.keys():
            if o in img_objects:
                object_counts[o] += 1

        return object_counts

    def count_properties_in_img_repr(img_repr, properties):
        img_properties = []
        if len(img_repr) > 0:
            if type(img_repr[0]) == list:
                img_properties = [
                    o[1:] for o in img_repr if type(o) == list and len(o) > 1
                ]
            img_properties = [p for sublist in img_properties for p in sublist]

        property_counts = {p: 0 for p in properties}
        for p in property_counts.keys():
            if p in img_properties:
                property_counts[p] += 1
        return property_counts

    def count_actions_in_img_repr(img_repr, actions):
        img_actions = []
        if len(img_repr) > 0 and type(img_repr) == list:
            if type(img_repr[0]) == list:
                img_actions = [o[0] for o in img_repr if type(o) == list and len(o) > 0]
        action_counts = {a: 0 for a in actions}
        for a in action_counts.keys():
            if a in img_actions:
                action_counts[a] += 1
        return action_counts

    def likelihood_ratio(pos_counts, neg_counts, n_pos, n_neg):
        # avoid division by zero
        if pos_counts + neg_counts == 0:
            return 0
        return (pos_counts / n_pos) / ((pos_counts / n_pos) + (neg_counts / n_neg))

    def positive_ratio(pos_counts, neg_counts, n_pos, n_neg):
        if pos_counts == 0:
            return 1e-2  # epsilon to avoid zero likelihood
        return pos_counts / n_pos

    def likelihood_ratio_weighted(pos_counts, neg_counts, n_pos, n_neg):
        epsilon = 1e-2
        # avoid division by zero
        if pos_counts + neg_counts == 0:
            return epsilon
        if pos_counts == 0:
            return epsilon
        ratio = pos_counts / (pos_counts + neg_counts)
        weight = pos_counts / n_pos
        return weight * ratio

    def CFG_to_PCFG_with_naive_frequency_ratio(
        self,
        img_object_representations,
        img_action_representations,
        objects,
        properties,
        actions,
        conditions,
    ):
        # TODO: case when pos and neg are not same size
        n_imgs = len(img_object_representations)
        pos_img_representations = img_object_representations[: n_imgs // 2]
        neg_img_representations = img_object_representations[n_imgs // 2 :]

        pos_imgs_object_counts = {o: 0 for o in objects}
        neg_imgs_object_counts = {o: 0 for o in objects}

        # count each object wether it appears in the img representations
        for img_repr in pos_img_representations:
            img_object_counts = CFG.count_objects_in_img_repr(img_repr, objects)
            for o in objects:
                pos_imgs_object_counts[o] += img_object_counts[o]

        for img_repr in neg_img_representations:
            img_object_counts = CFG.count_objects_in_img_repr(img_repr, objects)
            for o in objects:
                neg_imgs_object_counts[o] += img_object_counts[o]

        object_likelihoods = {
            o: CFG.likelihood_ratio(
                pos_imgs_object_counts[o],
                neg_imgs_object_counts[o],
                len(pos_img_representations),
                len(neg_img_representations),
            )
            for o in objects
        }
        sum_object_likelihoods = sum(object_likelihoods.values())

        ## Properties

        positive_property_counts = {p: 0 for p in properties}
        negative_property_counts = {p: 0 for p in properties}

        for img_repr in pos_img_representations:
            img_object_counts = CFG.count_properties_in_img_repr(img_repr, properties)
            for p in properties:
                positive_property_counts[p] += img_object_counts[p]

        for img_repr in neg_img_representations:
            img_object_counts = CFG.count_properties_in_img_repr(img_repr, properties)
            for p in properties:
                negative_property_counts[p] += img_object_counts[p]

        property_likelihoods = {
            p: CFG.likelihood_ratio(
                positive_property_counts[p],
                negative_property_counts[p],
                len(pos_img_representations),
                len(neg_img_representations),
            )
            for p in properties
        }
        sum_property_likelihoods = sum(property_likelihoods.values())

        ## Actions

        pos_action_representations = img_action_representations[: n_imgs // 2]
        neg_action_representations = img_action_representations[n_imgs // 2 :]

        positive_action_counts = {a: 0 for a in actions}
        negative_action_counts = {a: 0 for a in actions}

        for img_repr in pos_action_representations:
            action_counts = CFG.count_actions_in_img_repr(img_repr, actions)
            for a in positive_action_counts.keys():
                if a in action_counts:
                    positive_action_counts[a] += action_counts[a]

        for img_repr in neg_action_representations:
            action_counts = CFG.count_actions_in_img_repr(img_repr, actions)
            for a in negative_action_counts.keys():
                if a in action_counts:
                    negative_action_counts[a] += action_counts[a]

        action_likelihoods = {
            a: CFG.likelihood_ratio(
                positive_action_counts[a],
                negative_action_counts[a],
                len(pos_action_representations),
                len(neg_action_representations),
            )
            for a in actions
        }
        sum_action_likelihoods = sum(action_likelihoods.values())

        # print(object_likelihoods)
        # print(property_likelihoods)
        # print(action_likelihoods)

        # the first object is more likely than the second object etc.
        # the first property is more likely than the second property etc.
        # the first action is more likely than the second action etc.
        augmented_rules = {}
        for S in self.rules:
            augmented_rules[S] = {}
            p = len(self.rules[S])
            for P in self.rules[S]:
                if P.type == OBJECT:
                    if sum_object_likelihoods == 0:
                        weight = 1 / p
                    else:
                        # weight = object_counts[P.primitive] / n_imgs
                        weight = (
                            object_likelihoods[P.primitive] / sum_object_likelihoods
                        )
                        # normalize the weight
                        weight /= sum_object_likelihoods
                    # print(f"Object {P.primitive} weight: {weight}")
                elif P.type == PROPERTY:
                    if sum_property_likelihoods == 0:
                        weight = 1 / p
                    else:
                        weight = (
                            property_likelihoods[P.primitive] / sum_property_likelihoods
                        )
                    # normalize the weight
                    weight /= sum_property_likelihoods
                elif P.type == ACTION:
                    if sum_action_likelihoods == 0:
                        weight = 1 / p
                    else:
                        weight = (
                            positive_action_counts[P.primitive] / sum_action_likelihoods
                        )
                        # normalize the weight
                        weight /= sum_action_likelihoods
                elif P.type == CONDITION:
                    weight = 1 / (conditions.index(P.primitive) + 1)
                else:
                    weight = 1 / p
                augmented_rules[S][P] = (self.rules[S][P], weight)
        return PCFG(
            start=self.start,
            rules=augmented_rules,
            max_program_depth=self.max_program_depth,
            clean=True,
        )

    def get_likelihoods(
        self,
        img_representations_for_variable,
        variables,
        variable_type,
        mode="likelihood_ratio_weighted",
    ):

        # TODO: case when pos and neg are not same size
        n_imgs = len(img_representations_for_variable)
        pos_img_representations = img_representations_for_variable[: n_imgs // 2]
        neg_img_representations = img_representations_for_variable[n_imgs // 2 :]

        pos_imgs_variable_counts = {o: 0 for o in variables}
        neg_imgs_variable_counts = {o: 0 for o in variables}

        # count each object wether it appears in the img representations
        for img_repr in pos_img_representations:
            if variable_type == "objects":
                img_object_counts = CFG.count_objects_in_img_repr(img_repr, variables)
            elif variable_type == "properties":
                img_object_counts = CFG.count_properties_in_img_repr(
                    img_repr, variables
                )
            elif variable_type == "actions":
                img_object_counts = CFG.count_actions_in_img_repr(img_repr, variables)
            for o in variables:
                pos_imgs_variable_counts[o] += img_object_counts[o]

        for img_repr in neg_img_representations:
            if variable_type == "objects":
                img_object_counts = CFG.count_objects_in_img_repr(img_repr, variables)
            elif variable_type == "properties":
                img_object_counts = CFG.count_properties_in_img_repr(
                    img_repr, variables
                )
            elif variable_type == "actions":
                img_object_counts = CFG.count_actions_in_img_repr(img_repr, variables)
            for o in variables:
                neg_imgs_variable_counts[o] += img_object_counts[o]

        if mode == "likelihood_ratio_weighted":
            variable_likelihoods = {
                o: CFG.likelihood_ratio_weighted(
                    pos_imgs_variable_counts[o],
                    neg_imgs_variable_counts[o],
                    len(pos_img_representations),
                    len(neg_img_representations),
                )
                for o in variables
            }
        elif mode == "positive_ratio":
            variable_likelihoods = {
                o: CFG.positive_ratio(
                    pos_imgs_variable_counts[o],
                    neg_imgs_variable_counts[o],
                    len(pos_img_representations),
                    len(neg_img_representations),
                )
                for o in variables
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")

        sum_variable_likelihoods = sum(variable_likelihoods.values())

        # print(f"{variable_type} likelihoods: {variable_likelihoods}")

        return variable_likelihoods, sum_variable_likelihoods

    def CFG_to_PCFG_with_positives_only(self, img_representations, variables):

        likelihoods_for_variables = {}

        for variable_type, variables in variables.items():
            img_representations_for_variable = img_representations[variable_type]

            variable_likelihoods, sum_variable_likelihoods = self.get_likelihoods(
                img_representations_for_variable,
                variables,
                variable_type,
                mode="positive_ratio",
            )

            likelihoods_for_variables[variable_type] = (
                variable_likelihoods,
                sum_variable_likelihoods,
            )

        augmented_rules = {}
        for S in self.rules:
            augmented_rules[S] = {}
            p = len(self.rules[S])
            for P in self.rules[S]:
                if P.type == OBJECT:
                    object_likelihoods, sum_object_likelihoods = (
                        likelihoods_for_variables["objects"]
                    )
                    if sum_object_likelihoods == 0:
                        weight = 1 / p
                    else:
                        weight = object_likelihoods[P.primitive]
                        # normalize the weight
                        # weight /= sum_object_likelihoods
                elif P.type == PROPERTY:
                    property_likelihoods, sum_property_likelihoods = (
                        likelihoods_for_variables["properties"]
                    )
                    if sum_property_likelihoods == 0:
                        weight = 1 / p
                    else:
                        weight = property_likelihoods[P.primitive]
                        # normalize the weight
                        # weight /= sum_property_likelihoods
                elif P.type == ACTION:
                    action_likelihoods, sum_action_likelihoods = (
                        likelihoods_for_variables["actions"]
                    )
                    if sum_action_likelihoods == 0:
                        weight = 1 / p
                    else:
                        weight = action_likelihoods[P.primitive]
                        # normalize the weight
                        # weight /= sum_action_likelihoods
                else:
                    weight = 1 / p
                augmented_rules[S][P] = (self.rules[S][P], weight)
        return PCFG(
            start=self.start,
            rules=augmented_rules,
            max_program_depth=self.max_program_depth,
            clean=True,
        )

    def CFG_to_PCFG_with_naive_weighted(self, img_representations, variables):

        likelihoods_for_variables = {}

        for variable_type, variables in variables.items():
            img_representations_for_variable = img_representations[variable_type]

            variable_likelihoods, sum_variable_likelihoods = self.get_likelihoods(
                img_representations_for_variable,
                variables,
                variable_type,
                mode="likelihood_ratio_weighted",
            )

            likelihoods_for_variables[variable_type] = (
                variable_likelihoods,
                sum_variable_likelihoods,
            )

        augmented_rules = {}
        for S in self.rules:
            augmented_rules[S] = {}
            p = len(self.rules[S])
            for P in self.rules[S]:
                if P.type == OBJECT:
                    object_likelihoods, sum_object_likelihoods = (
                        likelihoods_for_variables["objects"]
                    )
                    if sum_object_likelihoods == 0:
                        weight = 1 / p
                    else:
                        weight = object_likelihoods[P.primitive]
                        # normalize the weight
                        # weight /= sum_object_likelihoods
                elif P.type == PROPERTY:
                    property_likelihoods, sum_property_likelihoods = (
                        likelihoods_for_variables["properties"]
                    )
                    if sum_property_likelihoods == 0:
                        weight = 1 / p
                    else:
                        weight = property_likelihoods[P.primitive]
                        # normalize the weight
                        # weight /= sum_property_likelihoods
                elif P.type == ACTION:
                    action_likelihoods, sum_action_likelihoods = (
                        likelihoods_for_variables["actions"]
                    )
                    if sum_action_likelihoods == 0:
                        weight = 1 / p
                    else:
                        weight = action_likelihoods[P.primitive]
                        # normalize the weight
                        # weight /= sum_action_likelihoods
                else:
                    weight = 1 / p
                augmented_rules[S][P] = (self.rules[S][P], weight)
        return PCFG(
            start=self.start,
            rules=augmented_rules,
            max_program_depth=self.max_program_depth,
            clean=True,
        )

    def CFG_to_Random_PCFG(self, alpha=0.7):
        new_rules = {}
        for S in self.rules:
            out_degree = len(self.rules[S])
            # weights with alpha-exponential decrease
            weights = [random.random() * (alpha**i) for i in range(out_degree)]
            s = sum(weights)
            # normalization
            weights = [w / s for w in weights]
            random_permutation = list(
                np.random.permutation([i for i in range(out_degree)])
            )
            new_rules[S] = {}
            for i, P in enumerate(self.rules[S]):
                new_rules[S][P] = (self.rules[S][P], weights[random_permutation[i]])
        return PCFG(
            start=self.start,
            rules=new_rules,
            max_program_depth=self.max_program_depth,
            clean=True,
        )
