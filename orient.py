#!/usr/local/bin/python3
import pickle
import sys
import numpy as np
from random import shuffle
import random
import pandas as pd

# col_pairs = [(i, j) for i in range(1, 193) for j in range(1, 193) if i != j]

class Node:
    def __init__(self):
        self.orientation = None
        self.c1 = None
        self.c2 = None
        self.right = None
        self.left = None
        self.purity = None
        self.class_max = None


def entropy_formula(count, size):
    if count == 0:
        return 0.00
    return -(count / size) * np.log((count / size))


def calc_purity(frame):
    if frame.shape[0] == 0:
        return 0, None

    count_0 = frame.loc[frame.iloc[:, 0] == 0].shape[0]
    count_90 = frame.loc[frame.iloc[:, 0] == 90].shape[0]
    count_180 = frame.loc[frame.iloc[:, 0] == 180].shape[0]
    count_270 = frame.loc[frame.iloc[:, 0] == 270].shape[0]

    purity_0 = count_0 / (count_0 + count_90 + count_180 + count_270)
    purity_90 = count_90 / (count_0 + count_90 + count_180 + count_270)
    purity_180 = count_180 / (count_0 + count_90 + count_180 + count_270)
    purity_270 = count_270 / (count_0 + count_90 + count_180 + count_270)
    list_classes = [purity_0, purity_90, purity_180, purity_270]
    clf_class = np.argmax(list_classes)
    if clf_class == 1:
        clf_class = 90
    if clf_class == 2:
        clf_class = 180
    if clf_class == 3:
        clf_class = 270

    return max(list_classes) * 100, clf_class


def calc_entopy(frame):
    classes = [0, 90, 180, 270]
    size = frame.shape[0]
    entropy = 0

    for class_i in classes:
        count = frame.loc[frame.iloc[:, 0] == class_i].shape[0]
        entropy += entropy_formula(count, size)
    return entropy


def split_data_entropy(frame, node):
    print("recursion")

    best_gain = 0
    best_c1 = None
    best_c2 = None
    parent_entropy = calc_entopy(frame)
    for c1, c2 in columns:
        # print(c1, c2)
        left_split = frame.loc[frame.iloc[:, c1] < frame.iloc[:, c2]]
        left_entropy = calc_entopy(left_split)

        right_split = frame.loc[frame.iloc[:, c1] >= frame.iloc[:, c2]]
        right_entropy = calc_entopy(right_split)

        if left_split.shape[0] == 0 or right_split.shape[0] == 0:
            # print("check")
            continue

        scaling_left = left_split.shape[0] / frame.shape[0]
        scaling_right = right_split.shape[0] / frame.shape[0]

        weighted_entropy = scaling_left * left_entropy + scaling_right * right_entropy

        gain = parent_entropy - weighted_entropy
        # print("gain:", gain)
        if gain > best_gain:
            best_gain = gain
            best_c1 = c1
            best_c2 = c2

    print("best gain:", best_gain)
    if not best_c1 or not best_c2:
        print("bad split")
        return

    print("check", best_c1, best_c2)
    best_left_split = frame.loc[frame.iloc[:, best_c1] < frame.iloc[:, best_c2]]
    best_right_split = frame.loc[frame.iloc[:, best_c1] >= frame.iloc[:, best_c2]]

    node.c1 = best_c1
    node.c2 = best_c2

    purity, class_ = calc_purity(best_left_split)

    if purity >= 60:
        print("95")
        new_node = Node()
        node.left = new_node
        new_node.orientation = class_
        new_node.purity = purity
    else:
        new_node = Node()
        node.left = new_node
        new_node.purity = purity
        split_data_entropy(best_left_split, new_node)

    purity, class_ = calc_purity(best_right_split)

    if purity >= 60:
        print("95")
        new_node = Node()
        node.right = new_node
        new_node.orientation = class_
        new_node.purity = purity
    else:
        new_node = Node()
        node.right = new_node
        new_node.purity = purity
        split_data_entropy(best_right_split, new_node)


def split_data_gain(frame, node):
    print("recursion")

    best_disorder = np.Inf
    best_c1 = None
    best_c2 = None
    for c1, c2 in columns:
        # print(c1, c2)
        left_split = frame.loc[frame.iloc[:, c1] < frame.iloc[:, c2]]
        left_entropy = calc_entopy(left_split)

        right_split = frame.loc[frame.iloc[:, c1] >= frame.iloc[:, c2]]
        right_entropy = calc_entopy(right_split)

        if left_split.shape[0] == 0 or right_split.shape[0] == 0:
            # print("found an empty split")
            continue

        scaling_left = left_split.shape[0] / frame.shape[0]
        scaling_right = right_split.shape[0] / frame.shape[0]

        avg_disorder = scaling_left * left_entropy + scaling_right * right_entropy

        if avg_disorder < best_disorder:
            best_disorder = avg_disorder
            best_c1 = c1
            best_c2 = c2

    if not best_c1 or not best_c2:
        # print("bad split")
        return

    # print("check", best_c1, best_c2)
    best_left_split = frame.loc[frame.iloc[:, best_c1] < frame.iloc[:, best_c2]]
    best_right_split = frame.loc[frame.iloc[:, best_c1] >= frame.iloc[:, best_c2]]

    node.c1 = best_c1
    node.c2 = best_c2

    purity, class_ = calc_purity(best_left_split)

    if purity >= 60:
        print("95")
        new_node = Node()
        node.left = new_node
        new_node.orientation = class_
        new_node.purity = purity
    else:
        new_node = Node()
        node.left = new_node
        new_node.purity = purity
        split_data_gain(best_left_split, new_node)

    purity, class_ = calc_purity(best_right_split)

    if purity >= 60:
        print("95")
        new_node = Node()
        node.right = new_node
        new_node.orientation = class_
        new_node.purity = purity
    else:
        new_node = Node()
        node.right = new_node
        new_node.purity = purity
        split_data_gain(best_right_split, new_node)


def traverse(node):
    if not node:
        return
    else:
        print("node:", (node.orientation, node.c1, node.c2, node.purity))
        traverse(node.left)
        traverse(node.right)


def accuracy(node, frame):
    if node.orientation is not None:
        complete_frame = frame.iloc[:, 0:2]
        count_correct = frame.loc[frame.iloc[:, 1] == node.orientation].shape[0]
        accurac = count_correct / frame.shape[0] * 100
        print("class", node.orientation, accurac)

        complete_frame.iloc[:, 1] = node.orientation

        with open("output.txt", "ab") as file:
            np.savetxt(file, complete_frame.values, fmt='%s')
        return count_correct

    print("c1:", node.c1)
    print("c2:", node.c2)
    print("orientation:", node.orientation)
    left_frame = frame.loc[frame.iloc[:, node.c1 + 1] < frame.iloc[:, node.c2 + 1]]
    correct_left = accuracy(node.left, left_frame)

    right_frame = frame.loc[frame.iloc[:, node.c1 + 1] >= frame.iloc[:, node.c2 + 1]]
    correct_right = accuracy(node.right, right_frame)

    return correct_left + correct_right


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise Exception("Please enter the right number of arguements!")

    operation = sys.argv[1]
    model_file = sys.argv[3]
    model = sys.argv[4]

    if operation == 'train':
        train_file = sys.argv[2]
        train = np.loadtxt(train_file, usecols=range(1, 194))
    else:
        test_file = sys.argv[2]
        test = np.loadtxt(test_file, usecols=range(1, 194))

    if model == 'tree':
        if operation == 'train':
            root = Node()
            x = pd.DataFrame(train)
            split_data_gain(x, root)
            traverse(root)
            write_model_nearest(root)
        else:
            root = read_model_nearest()
            test_data = pd.read_csv(test_file, delimiter=' ', header=None)
            print("test shape", test_data.shape[0])
            print("test acc:", (accuracy(root, test_data) / test_data.shape[0]) * 100)