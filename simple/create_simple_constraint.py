import numpy as np
import math
from itertools import chain
import pickle
from simple.node import *


def lookup_node(elements, nodes, literals):
    elements = tuple(elements)
    el = nodes.get(elements)
    if not el:

        # For creating the circuit
        n = Node()
        n.elements = []
        for e in elements:
            p, s = e
            if p.type == DECOMPOSITION:
                p = nodes.get(tuple(p.elements))
            else:
                p = literals.get(p.elements)
            if s.type == DECOMPOSITION:
                s = nodes.get(tuple(s.elements))
            else:
                s = literals.get(s.elements)
            n.elements.append((p, s))
        nodes[tuple(n.elements)] = n

        el = n

    return el


def create_exactly_k(n, k):
    literals = dict(list(chain.from_iterable(
        ((i, Node(i, type=LITERAL)), (-i, Node(-i, type=LITERAL))) for i in
        range(1, n + 1))))
    nodes = {}

    dp_prev = np.ndarray((n, k + 1), dtype=Node)
    dp_prev.fill(None)
    for i in range(n):
        for j in range(2):
            dp_prev[i][j] = literals[-(i + 1)] if not j else literals[i + 1]

    for num_arr in (n // (2 ** i) for i in range(1, int(math.log2(n)) + 1)):
        dp_curr = np.ndarray((num_arr, k + 1), dtype=Node)
        dp_curr.fill(None)
        for i in range(0, num_arr):
            for j in range(0, k + 1):
                if n // num_arr < j: break
                l = []
                for jj in range(j + 1):
                    if (dp_prev[(i * 2), jj] and dp_prev[(i * 2) + 1, j - jj]):
                        l.append((dp_prev[(i * 2), jj], dp_prev[(i * 2) + 1, j - jj]))
                dp_curr[i, j] = lookup_node(l, nodes, literals)

        dp_prev = dp_curr

    return dp_curr


def create_and_save(n, k, root):
    alpha = create_exactly_k(n, k)[0][-1]
    with open(f'{root}/{n}C{k}.pkl', 'wb') as out:
        pickle.dump(alpha, out, pickle.HIGHEST_PROTOCOL)
        print(f"{n}C{k} done")
