import torch
from torch import Tensor

class Graph:
    def __init__(self, otherGraph=None):
        if otherGraph is None:
            self.v1 = torch.randn(3, requires_grad=True)
            self.v2 = torch.randn(3, requires_grad=True)
        else:
            self.v1 = torch.tensor(data=otherGraph.v1.data, requires_grad=True)
            self.v2 = torch.tensor(data=otherGraph.v2.data, requires_grad=True)
        self.intermediate_tensors = []
        self.forward(torch.zeros(3), torch.zeros(3))

    def forward(self, x, y):
        self.intermediate_tensors.append(x * self.v1)
        self.intermediate_tensors.append(self.intermediate_tensors[-1].relu())
        self.intermediate_tensors.append(y * self.v2)
        self.intermediate_tensors.append(self.intermediate_tensors[-1].relu())
        self.intermediate_tensors.append(torch.cat((self.intermediate_tensors[-3], self.intermediate_tensors[-1]), 0))
        self.intermediate_tensors.append(self.intermediate_tensors[-1].sum())

    # Computes backward pass, retains state for this example.
    def backward(self, expected):
        self.intermediate_tensors[-1].backward(expected, True)
        return self.v1.grad.clone(), self.v2.grad.clone()

    def copy_state_from(self, other_graph):
        for (state_from, state_to) in \
                zip(other_graph.intermediate_tensors, self.intermediate_tensors):
            if len(state_from.shape) == 0:
                state_to.data = state_from.data
            else:
                for i in range(state_from.shape[0]):
                    state_to.data[i] = state_from.data[i]

graph_forward = Graph()
graph_backward = Graph(graph_forward)

x = torch.tensor([1,2,3], dtype=torch.float, requires_grad=True)
y = torch.tensor([-1,1,-1], dtype=torch.float, requires_grad=True)
graph_forward.forward(x,y)

ARBITRARY_VALUE = torch.tensor(10, dtype=torch.float)
grads_from_original_network = graph_forward.backward(ARBITRARY_VALUE)
grads_before_copy = graph_backward.backward(ARBITRARY_VALUE)
graph_backward.copy_state_from(graph_forward)
grads_after_copy = graph_backward.backward(ARBITRARY_VALUE)

assert(not torch.all(grads_after_copy[0].eq(grads_before_copy[0])))
assert(not torch.all(grads_after_copy[1].eq(grads_before_copy[1])))

assert(torch.all(torch.eq(grads_after_copy[0], grads_from_original_network[0])))
assert(torch.all(torch.eq(grads_after_copy[1], grads_from_original_network[1])))

'''
I'm experimenting with the idea of separating the computation of the forward pass of a graph from the computation of the backward pass. If this can be done efficiently (which is dubious - to be sure), I think it might unlock some interesting new model parallelism schemes that I'd like to explore.

To accomplish this in PyTorch, what I'd need to do is to:

1. Build the same computational graph inside of two different interpreters
2. Compute forward() on both graphs with dummy data to build up a grad_fn chain.

Then repeat:
1. Compute forward() with real data on one graph
2. Copy the required intermediate state tensors from the graph in (1) to the graph in (3)
3. Compute backwards on the second graph with the given state tensors.
'''