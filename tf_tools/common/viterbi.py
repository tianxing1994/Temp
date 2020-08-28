#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


class Viterbi(object):
    sep = '-'

    def __init__(self, transition_matrix):
        self.transition_matrix = transition_matrix

    def __call__(self, inputs):
        nodes = self._preprocess(inputs)
        paths = self._viterbi(nodes)
        best = self._get_max_paths(paths)
        return best

    @staticmethod
    def _preprocess(inputs):
        nodes = list()
        for sample in inputs:
            temp = {str(i): e for i, e in enumerate(sample)}
            nodes.append(temp)
        return nodes

    def _viterbi(self, nodes):
        paths = nodes[0]
        for l in range(1, len(nodes)):
            paths_old, paths = paths, dict()
            for n, ns in nodes[l].items():
                max_path, max_score = '', -1e10
                for p, ps in paths_old.items():
                    score = ns + ps + self.transition_matrix[int(p.split(self.sep)[-1]), int(n)]
                    if score > max_score:
                        max_path, max_score = p + self.sep + n, score
                paths[max_path] = max_score
        return paths

    def _get_max_paths(self, paths):
        paths = sorted(paths.items(), key=lambda x: x[1])
        path, _ = paths[-1]
        path = list(map(lambda x: int(x), path.split(self.sep)))
        return path


def demo1():
    crf_weights = np.array(
        [[1, 2, 3, 4],
         [4, 3, 2, 1],
         [1, 2, 2, 1],
         [2, 3, 3, 2]],
        dtype=np.float
    )

    nodes = [
        {'0': 1.0, '1': 2.0, '2': 3.0, '3': 4.0},
        {'0': 4.0, '1': 3.0, '2': 2.0, '3': 1.0},
        {'0': 1.0, '1': 2.0, '2': 2.0, '3': 2.0},
        {'0': 2.0, '1': 6.0, '2': 7.0, '3': 4.0},
        {'0': 3.0, '1': 2.0, '2': 3.0, '3': 1.0},
        {'0': 5.0, '1': 1.0, '2': 1.0, '3': 5.0},
        {'0': 7.0, '1': 5.0, '2': 4.0, '3': 6.0},
    ]

    paths = nodes[0]
    for l in range(1, len(nodes)):
        paths_old, paths = paths, dict()
        for n, ns in nodes[l].items():
            max_path, max_score = "", -1e10
            for p, ps in paths_old.items():
                score = ns + ps + crf_weights[int(p.split('-')[-1]), int(n)]
                if score > max_score:
                    max_path, max_score = p + '-' + n, score
            paths[max_path] = max_score
    print(paths)
    return


def demo2():
    crf_weights = np.array(
        [[1, 2, 3, 4],
         [4, 3, 2, 1],
         [1, 2, 2, 1],
         [2, 3, 3, 2]],
        dtype=np.float
    )

    x = np.array(
        [[1, 2, 3, 4],
         [2, 3, 2, 1],
         [4, 3, 2, 5],
         [1, 4, 5, 2],
         [2, 2, 2, 1],
         [3, 3, 4, 5],
         [5, 3, 2, 2]],
        dtype=np.float
    )

    viterbi = Viterbi(transition_matrix=crf_weights)
    ret = viterbi(x)
    print(ret)

    return


if __name__ == '__main__':
    # demo1()
    demo2()
