#!/home/dmer/.pyenv/versions/env3/bin/python
# -*- coding: utf-8 -*-

'''
---------------------------------------------------------------------------
File Name: fpgrowth_val.py
Description: 改编 pyfpgrowth.py，增加
Variables: 
        1. transactions： 交易日志
        2. support_threshold：支持度阈值
        3. TN： 需要评估的交易量，替换原始算法的按记录条数计算
        4. 原始算法有bug，漏掉了190行，造成根节点数据漏掉
Changed by: zhangzhiyong
Changed date: Finally Changed on 2018/5/14
---------------------------------------------------------------------------
'''

import itertools

class FPNode(object):
    """
    A node in the FP tree.
    """

    def __init__(self, value, count, parent):
        """
        Create the node.
        """
        self.value = value
        self.count = count    # 增加一个成员记录交易量 [count, tran_value]
        self.parent = parent   
        self.link = None
        self.children = []

    def has_child(self, value):
        """
        Check if node has a particular child node.
        """
        for node in self.children:
            if node.value == value:
                return True

        return False

    def get_child(self, value):
        """
        Return a child node with a particular value.
        """
        for node in self.children:
            if node.value == value:
                return node

        return None

    def add_child(self, value, tvalue):
        """
        Add a node as a child node.
        """
        child = FPNode(value, [1, tvalue], self)
        self.children.append(child)
        return child


class FPTree(object):
    """
    A frequent pattern tree.
    """

    def __init__(self, transactions, threshold, root_value, root_count, trans_val):
        """
        Initialize the tree.
        """
        self.frequent = self.find_frequent_items(transactions, threshold, trans_val)
        self.headers = self.build_header_table(self.frequent)
        self.root = self.build_fptree(
                        transactions, root_value,
                        root_count, self.frequent, self.headers, trans_val)

    @staticmethod
    def find_frequent_items(transactions, threshold, trans_val):
        """
        Create a dictionary of items with occurrences above the threshold.
        """
        items = {}
        i = 0
        for transaction in transactions:
            for item in transaction:
                if item in items:
                    items[item][0] += 1
                    items[item][1] += trans_val[i]
                else:
                    items[item] = [1, trans_val[i]]
            i +=1

        for key in list(items.keys()):
            if items[key][1] < threshold:
                del items[key]

        return items

    @staticmethod
    def build_header_table(frequent):
        """
        Build the header table.
        """
        headers = {}
        for key in frequent.keys():
            headers[key] = None

        return headers

    def build_fptree(self, transactions, root_value,
                     root_count, frequent, headers, trans_val):
        """
        Build the FP tree and return the root node.
        """
        root = FPNode(root_value, root_count, None) 

        i = 0
        for transaction in transactions:
            sorted_items = [x for x in transaction if x in frequent]
            sorted_items.sort(key=lambda x: frequent[x][1], reverse=True)   # frequent[x]: [count, tvalue]
            if len(sorted_items) > 0:
                self.insert_tree(sorted_items, root, headers, trans_val[i])
            i +=1
        return root

    def insert_tree(self, items, node, headers, tran_val):
        """
        Recursively grow FP tree.
        """
        first = items[0]
        child = node.get_child(first)
        if child is not None:
            child.count[0] += 1
            child.count[1] += tran_val
        else:
            # Add new child.
            child = node.add_child(first, tran_val)

            # Link it to header structure.
            if headers[first] is None:
                headers[first] = child
            else:
                current = headers[first]
                while current.link is not None:
                    current = current.link
                current.link = child

        # Call function recursively.
        remaining_items = items[1:]
        if len(remaining_items) > 0:
            self.insert_tree(remaining_items, child, headers, tran_val)

    def tree_has_single_path(self, node):
        """
        If there is a single path in the tree,
        return True, else return False.
        """
        num_children = len(node.children)
        if num_children > 1:
            return False
        elif num_children == 0:
            return True
        else:
            return True and self.tree_has_single_path(node.children[0])

    def mine_patterns(self, threshold):
        """
        Mine the constructed FP tree for frequent patterns.
        """
        if self.tree_has_single_path(self.root):
            return self.generate_pattern_list()
        else:
            return self.zip_patterns(self.mine_sub_trees(threshold))

    def zip_patterns(self, patterns):
        """
        Append suffix to patterns in dictionary if
        we are in a conditional FP tree.
        """
        suffix = self.root.value

        if suffix is not None:
            # We are in a conditional tree.
            new_patterns = {}
            for key in patterns.keys():
                new_patterns[tuple(sorted(list(key) + [suffix]))] = patterns[key]
            new_patterns[tuple([self.root.value])] = self.root.count[1]     # 原始程序的bug，漏掉这行
            return new_patterns

        return patterns

    def generate_pattern_list(self):
        """
        Generate a list of patterns with support counts.
        """
        patterns = {}
        items = self.frequent.keys()

        # If we are in a conditional tree,
        # the suffix is a pattern on its own.
        if self.root.value is None:
            suffix_value = []
        else:
            suffix_value = [self.root.value]
            patterns[tuple(suffix_value)] = self.root.count[1]

        for i in range(1, len(items) + 1):
            for subset in itertools.combinations(items, i):
                pattern = tuple(sorted(list(subset) + suffix_value))
                patterns[pattern] = \
                    min([self.frequent[x][1] for x in subset])

        # print('-------------', patterns)
        return patterns

    def mine_sub_trees(self, threshold):
        """
        Generate subtrees and mine them for patterns.
        """
        patterns = {}
        mining_order = sorted(self.frequent.keys(),
                              key=lambda x: self.frequent[x][1])
        # print('mining_order: ', mining_order)
        # Get items in tree in reverse order of occurrences.
        for item in mining_order:
            suffixes = []
            conditional_tree_input = []
            cti_trans_val = []
            node = self.headers[item]

            # Follow node links to get a list of
            # all occurrences of a certain item.
            while node is not None:
                suffixes.append(node)
                node = node.link

            # For each occurrence of the item, 
            # trace the path back to the root node.
            for suffix in suffixes:
                frequency = suffix.count
                path = []
                parent = suffix.parent

                while parent.parent is not None:
                    path.append(parent.value)
                    parent = parent.parent

                # for i in range(frequency[0]):     # 使用循环，计算得到的support>1
                conditional_tree_input.append(path)
                cti_trans_val.append(frequency[1])

            # Now we have the input for a subtree,
            # so construct it and grab the patterns.
            # print('conditional_tree_input: ', conditional_tree_input)
            # print('cti_trans_val: ', cti_trans_val)
            # print('item: ', item)
            # print('self.frequent[item]: ', self.frequent[item])
            # exit()

            subtree = FPTree(conditional_tree_input, threshold,
                             item, self.frequent[item], cti_trans_val)
            subtree_patterns = subtree.mine_patterns(threshold)

            # Insert subtree patterns into main patterns dictionary.
            for pattern in subtree_patterns.keys():
                if pattern in patterns:
                    patterns[pattern][0] += subtree_patterns[pattern][0]
                    patterns[pattern][1] += subtree_patterns[pattern][1]
                else:
                    patterns[pattern] = subtree_patterns[pattern]

        return patterns


def find_frequent_patterns(transactions, support_threshold, TN):
    """
    Given a set of transactions, find the patterns in it
    over the specified support threshold.
    """
    tree = FPTree(transactions, support_threshold, None, None, TN)
    return tree.mine_patterns(support_threshold)


def generate_association_rules(patterns, confidence_threshold):
    """
    Given a set of frequent itemsets, return a dict
    of association rules in the form
    {(left): ((right), confidence)}
    """
    rules = {}
    for itemset in patterns.keys():
        upper_support = patterns[itemset]

        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))

                if antecedent in patterns:
                    lower_support = patterns[antecedent]
                    confidence = float(upper_support) / lower_support

                    if confidence >= confidence_threshold:
                        rules[antecedent] = (consequent, confidence)

    return rules
