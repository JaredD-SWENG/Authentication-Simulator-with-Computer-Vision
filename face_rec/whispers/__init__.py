"""
 Provides tools for performing and analyzing the whispers label-propagation algorithm to
 determine how many unique individuals are represented amongst a collection of pictures.
"""
import os
import random
from collections import Counter, defaultdict
from glob import glob
from itertools import groupby, takewhile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from facenet_models import FacenetModel
from facenet_pytorch.models.utils.detect_face import crop_resize
from skimage import io

__all__ = [
    "labeled_pics_to_descriptors",
    "create_adjacency_matrix",
    "plot_adjacency_distr",
    "create_graph",
    "run_propagation",
    "plot_graph",
]


def labeled_pics_to_descriptors(
        data_dir: Union[str, Path], accept_prob: float = 0.95
) -> Tuple[np.ndarray, Tuple[str, ...], np.ndarray]:
    """Given labeled directories of pictures individual people, returns the
    face-descriptors and the corresponding names.

    This function warns when more or less than one face is detected in a picture.

    Parameters
    ----------
    data_dir : str
        Directory containing named-directories of pictures of individuals.

        For example

        data_dir/
           - John/
              - jpgs of John
            ...

    accept_prob : float
        The minimum face-detection probability/confidence that is required to accept
        a detection.

    Returns
    -------
    Tuple[numpy.ndarray, Tuple[str, ...], numpy-ndarray]
        NxD array of N face-descriptors, a length-N list of the names corresponding
        to the descriptors, and an Nx160x160x3 array of cropped and resized faces.
    """

    model = FacenetModel()

    data_dir = Path(data_dir)
    names = sorted(i.split(os.sep)[-1] for i in glob(str(data_dir / "*")))
    pics = [
        (name, pic)
        for name in names
        for pic in glob(str(data_dir / name / "*"))
        if pic.split(".")[-1].lower() in {"jpg", "png", "jpeg"}
    ]

    # Stores N descriptors -- one for each picture
    all_descriptors = []
    used_names = []
    faces = []
    for name, pic in pics:
        img = io.imread(pic)
        boxes, probs, _ = model.detect(img)

        # each picture should contain one face
        num_faces_in_pic = (np.array(probs) > accept_prob).sum()
        if num_faces_in_pic != 1:
            print(
                f"Warning: {name} contains a picture with {(np.array(probs) > accept_prob).sum()} faces:\n\t{pic} .. skipping!"
            )
            continue

        all_descriptors += [
            d
            for n, d in enumerate(model.compute_descriptors(img, boxes))
            if probs[n] > accept_prob
        ]

        # For each detected face...
        for box, prob in zip(boxes, probs):
            if prob < accept_prob:
                # skip if the detection score / probability isn't high enough
                continue

            # We can get the cropped face for the purpose of validating our algorithm
            cropped_face = np.array(
                [crop_resize(img, [int(max(0, coord)) for coord in box], 160)]
            )

            used_names.append(name)
            faces.append(cropped_face)

    return np.vstack(all_descriptors), tuple(used_names), np.vstack(faces)


def create_adjacency_matrix(
        descriptors: np.ndarray,
        cutoff: Optional[float] = None,
        weighting_func: Callable[[np.ndarray], np.ndarray] = None,
):
    """Produce an adjacency matrix of cosine distances between descriptor
    vectors. A cutoff value can be provided such that distances exceeding
    the cutoff are set to 0. A weighting function can be supplied to perform
    a mapping on all non-zero weights.

    Parameters
    ----------
    descriptors : numpy.ndarray, shape=(N, D)
        Face-descriptor vectors for N faces.

    cutoff : Optional[float]
        The cutoff above which all distances are set to 0.

    weighting_func : Callable[[numpy.ndarray], numpy.ndarray]
        The mapping applied to all non-zero entries in the adjacency matrix.

    Returns
    -------
    numpy.ndarray, shape=(N, N)
        The adjacency matrix. The indexing corresponds to the order of the descriptors.
    """

    descriptors = descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True)
    similarity = descriptors @ descriptors.T
    cos_dist = 1 - similarity
    np.fill_diagonal(cos_dist, 0.0)

    # set dists above cutoff to 0
    if cutoff is not None:
        keep = np.logical_and(0 < cos_dist, cos_dist <= cutoff)
        thresh = np.where(keep)
        cos_dist[~keep] = 0.0
    else:
        thresh = np.ones(cos_dist.shape, dtype=bool)

    # map non-zero dists to weight values
    if weighting_func is not None:
        # Compute weighted cosine-distance
        cos_dist[thresh] = weighting_func(cos_dist[thresh])
    return cos_dist


def plot_adjacency_distr(descriptors: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a histogram of the adjacency matrix of cosine distances between descriptors.

    Only the upper-triangle of the matrix is binned - no double counting or trivial 0s.

    Parameters
    ----------
    descriptors : numpy.ndarray, shape=(N, D)
        Face-descriptors for N faces

    Returns
    -------
    Tuple[matplotlib.fig.Fig, matplotlib.axis.Axes]
        The figure and axes of the histogram plot.
    """

    adj = create_adjacency_matrix(descriptors)
    p = np.triu(adj)
    p = p[p > 1e-6]

    fig, ax = plt.subplots()
    ax.hist(p, bins=100, label="All People")
    ax.set_title("Distribution of face-descriptor cosine distances")
    ax.set_xlabel("Descriptor distance")
    ax.set_ylabel("Count")
    return fig, ax


class Node:
    def __init__(
            self,
            label: int,
            id_: int,
            neighbors_: Sequence[int],
            truth: Optional[Sequence[str]] = None,
    ):
        """
        Parameters
        ----------
        id : int
            A unique identifier - corresponds to its index in the graph and its index in the adjacency matrix.

        label : int
            The integer-ID that labels the cluster the node belongs to. Initialized to a
            random integer in [0, N)

        neighbors : Sequence[int]
            The IDs for the neighbors of the node.

        truth : Optional[Sequence[str]]
            The name of the person that the node corresponds to. If `None`,
            no truth values are recorded. This is optional and is only used
            to analyze the correctness of the whispers algorithm"""
        self.id = id_
        self.label = label
        self.neighbors = tuple(neighbors_)
        self.truth = truth


def create_graph(adj: np.ndarray, truth: Optional[Sequence[Optional[str]]]) -> Tuple[Node, ...]:
    """Create a graph from the NxN adjacency matrix. See 'Notes' for information.

    Parameters
    ----------
    adj : np.ndarray, shape=(N, N)
        The adjacency matrix for which to construct a graph, where N is the number of nodes. `adj[i, j]` indicates if node-j is a neighbor of node-i. If `adj[i, j]`
        is 0, then they are not neighbors.

        `adj` is always symmetric: `adj[i, j] == `adj[j, i]`

    truth : Optional[Sequential[str]]
        The true labels corresponding to each Node. This is optional and is only used
        if we are analyzing the correctness of our algorithm using precision and recall
        stats.

    Returns
    -------
    Tuple[Node, ...]
        See Notes

    Notes
    -----
    The graph is simply a tuple of N Node-instances, ordered according to the adjacency matrix.

    Each node has the attributes:
        - id : int
            A unique identifier - corresponds to its index in the graph and its index
            in the adjacency matrix.

        - label : int
            The integer-ID that labels the cluster the node belongs to. Initialized to a
            random integer in [0, N)

        - neighbors : Tuple[integer]
            The IDs for the neighbors of the node.

        - truth : Optional[Sequence[str]]]
            The name of the person that the node corresponds to. If `None`,
            no truth values are recorded.
    """

    # stores:  # node-ID -> [neighbor-1-ID, neighbor-2-ID, ...]
    neighbors = defaultdict(list)

    # Find the (node=row, neighbor=col) indices where the
    # adjacency matrix is not zero
    for node, neighbor in zip(*np.where(adj > 0)):
        # Record neighbor
        neighbors[node].append(neighbor)

    # If no truth is provided, we just pass in a list of `None`
    if truth is None:
        truth = [None for i in range(adj.shape[0])]

    labels = list(range(adj.shape[0]))
    random.shuffle(labels)

    return tuple(
        Node(labels[n], n, neighbors[n], truth[n]) for n in range(adj.shape[0])
    )


def _compute_pairwise_metrics(graph: Tuple[Node, ...]) -> Tuple[float, float]:
    """Compute the pair-precision and pair-recall accuracies for a graph.

    Parameters
    ----------
    graph : Tuple[Node, ...]

    Returns
    -------
    Tuple[float, float]
        Precision and recall accuracies, respectively.

    Notes
    -----
    precision = (same_truth & same_class) / [(same_truth & same_class) + (diff_truth & same_class)]

      - High precision: clustered conservatively

      - Low precision: aggressive clustering

    recall = (same_truth & same_class) / [(same_truth & same_class) + (same_truth & diff_class)]

      - High recall: inclusive clusters

      - Low recall: dispersive clusters
    """
    from collections import defaultdict
    from itertools import combinations

    categories = defaultdict(int)
    for n1, n2 in combinations(graph, 2):
        t = "truth" if n1.truth == n2.truth else "~truth"
        l = "label" if n1.label == n2.label else "~label"
        categories[f"{t},{l}"] += 1

    pair_prec = categories["truth,label"] / (
            categories["truth,label"] + categories["truth,~label"]
    )
    pair_recall = categories["truth,label"] / (
            categories["truth,label"] + categories["~truth,label"]
    )
    return pair_prec, pair_recall


def _propagate_label(
        node: Node,
        graph: Tuple[Node, ...],
        weights: Optional[np.ndarray],
):
    """Returns most-whispered label for the given node.

    Checks the neighbors of the node for the most common label –
    or the label with the heighest weight if `weights` is specified –
    and returns that label

    Parameters
    ----------
    node: Node
        The node whose label is being updated.

    graph : Tuple[Node, ... ]
        The graph of N nodes to propagate labels through.

    weights : Union[None, numpy.ndarray]
        `None` or an NxN adjacency matrix of weights to be used for label propagation.

    Returns
    -------
    new_node_label : int
        The most common or heavily weighted label from among the node's neighbors"""

    def first(x):
        return x[0]

    if weights is not None:
        # Accumulate total weights for each label.
        # The label with max weight is propagated to node

        # get (label, weight) for each neighbor of the noded
        label_wght = zip(
            (graph[i].label for i in node.neighbors), weights[node.id, node.neighbors])

        # Stores [(label, weight), ... for each neighbor] sorted by label.
        #
        # E.g. [(label-1, 2.0), (label-2, 1.0), (label-2, 2.5), (label-3, 1.5)]
        label_wght = sorted(label_wght, key=first)

        # Computes the total weight for each label:
        #    [(label-1, total-weight), (label-2, total-weight) ...]
        #
        # E.g. [(label-1, 2.0), (label-2, 3.5), (label-3, 1.5)]
        most_common = (
            (k, sum(amt for _, amt in v)) for k, v in groupby(label_wght, key=first)
        )

        # [(label, total_weight), ...] sorted by descending weight
        #
        # E.g. [(label-2, 3.5), (label-1, 2.0), (label-3, 1.5)]
        most_common = sorted(most_common, key=lambda x: x[1], reverse=True)
    else:
        # No weights are provided. We just tally up the most common labels

        # Stores [(label, label-count), ... tallied over all neighbors]
        #  -- sorted in descending count order
        # E.g. [(label-3, 10), (label-1, 4), (label-2, 1)]
        most_common = Counter(graph[i].label for i in node.neighbors).most_common()

    # Stores the heighest count or weight (if weight is available)
    _max_cnt_or_weight = most_common[0][1]

    # Get all of the labels that have the top tally / weight. Randomly select the
    # final propagated label from these.
    #
    # This is a tie-breaker mechanism. If there is only one label with the
    # top tally/weight then this simply returns that label.
    return random.choice(
        [lbl for lbl, _ in takewhile(lambda x: x[1] == _max_cnt_or_weight, most_common)]
    )


def run_propagation(
        graph: Tuple[Node, ...],
        weights: Optional[np.ndarray],
        num_it: int,
        stat_rate: Optional[int],
) -> Tuple[Tuple[Node, ...], Dict[str, List[float]]]:
    """Randomly choose a node and update its label. Update the node's label
    to agree with the most popular labels amongst its neighbors.

    This process is repeated for `num_it` times.

    If weights are supplied, then each neighbor's contribution
    is weighted. The label with the highest accumulated weight
    is adopted.

    The number of unique labels, pairwise precision-accuracy, and
    pairwise recall-accuracy are recorded during the run.

    Parameters
    ----------
    graph : Tuple[Node, ... ]
        The graph of N nodes to propagate labels through.

    weights : Union[None, numpy.ndarray]
        `None` or an NxN adjacency matrix of weights to be used for label propagation.

    num_it : int
        The number of iterations for which to perform the label-propagation process.

    stat_rate : Union[None, int]
        The frequency with which matching stats will be recorded. If `None`, no stats
        are recorded.

    Returns
    -------
    Tuple[Tuple[Node, ...], Dict[str, List[float]]]
        The graph post-propagation, and the recorded stats.

        The stat-dictionary's keys are 'num_labels', 'precision', and 'recall'.
    """

    # This is used to keep analysis statistics. It will record the precision and
    # recall of our label assignments, assuming that we included truth data
    # in our graphs nodes.
    #
    # This is not needed for the whispers algorithm to work
    stats = defaultdict(list)

    for it in range(num_it):
        # Randomly pick a node from our graph
        node = graph[random.randint(0, len(graph) - 1)]

        # If the node has neighbors, update the node's label
        # using `_propagate_label`
        if node.neighbors:
            node.label = _propagate_label(node, graph, weights)

        # (optional -- if analyzing performance)
        # Record precision/recall stats and number of distinct labels
        if stat_rate is not None and it % stat_rate == 0:
            stats["num_labels"].append(len(set(n.label for n in graph)))
            prec, recall = _compute_pairwise_metrics(graph)
            stats["precision"].append(prec)
            stats["recall"].append(recall)

    # Return the updated graph and the performance statistics (if any)
    return graph, stats


# This is provided to the students
def plot_graph(graph, adj):
    """Use the package networkx to produce a diagrammatic plot of the graph, with
    the nodes in the graph colored according to their current labels.

    Note that only 20 unique colors are available for the current color map,
    so common colors across nodes may be coincidental.

    Parameters
    ----------
    graph : Tuple[Node, ...]
        The graph to plot
    adj : numpy.ndarray, shape=(N, N)
        The adjacency-matrix for the graph. Nonzero entries indicate
        the presence of edges.

    Returns
    -------
    Tuple[matplotlib.fig.Fig, matplotlib.axis.Axes]
        The figure and axes for the plot.
    """
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    g = nx.Graph()
    for n, node in enumerate(graph):
        g.add_node(n)

    g.add_edges_from(zip(*np.where(np.triu(adj) > 0)))
    pos = nx.spring_layout(g)

    color = list(iter(cm.tab20b(np.linspace(0, 1, len(set(i.label for i in graph))))))
    color_map = dict(zip(sorted(set(i.label for i in graph)), color))
    colors = [color_map[i.label] for i in graph]
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(
        g, pos=pos, ax=ax, nodelist=range(len(graph)), node_color=colors
    )
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=g.edges())
    return fig, ax