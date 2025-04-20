from typing import Optional, List, NamedTuple, Callable, Union
import numpy as np
from pprint import pprint


class Dependency(NamedTuple):
    """A class to represent a dependency in the computation graph."""

    op: Callable
    inputs: "GradTensor"
    grad_fn: Optional[Callable] = None

    def __repr__(self) -> str:
        return f"Dependency(op={self.op.__name__}, inputs=GradTensor@{id(self.inputs)})"


class GradTensor:
    def __init__(
        self, data: np.ndarray, requires_grad: bool = False, depends_on=None
    ) -> None:
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad: Optional["GradTensor"] = None
        # NOTE leave a placeholder for the dependency
        self.depends_on: Optional[List[Dependency]] = depends_on
        if self.requires_grad:
            self.zero_grad()

    @property
    def shape(self) -> tuple:
        """Return the shape of the tensor."""
        return self.data.shape

    def zero_grad(self) -> None:
        """Zero the gradient of the tensor."""
        assert (
            self.requires_grad
        ), "Cannot zero grad of non-requires_grad tensor"
        self.grad = GradTensor(np.zeros_like(self.data))

    def __add__(self, other: "GradTensor") -> "GradTensor":
        """Add two tensors."""
        return _add(self, other)

    def __neg__(self) -> "GradTensor":
        """Negate the tensor."""
        return _neg(self)

    def __mul__(self, other: "GradTensor") -> "GradTensor":
        """Multiply two tensors."""
        return _mul(self, other)

    def __matmul__(self, other: "GradTensor") -> "GradTensor":
        """Matrix multiply two tensors."""
        return _matmul(self, other)

    def sum(self) -> "GradTensor":
        """Sum the tensor along the given axis."""
        return tensor_sum(self)

    def __repr__(self) -> str:
        if self.requires_grad:
            return f"GradTensor(data={self.data}, requires_grad={self.requires_grad}), depends_on={self.depends_on}, grad={self.grad})"
        else:
            return f"GradTensor(data={self.data}"

    def backward(
        self,
        grad: Optional["GradTensor"] = None,
    ) -> None:
        assert self.requires_grad, "Cannot backward on non-requires_grad tensor"
        if grad is None:
            # NOTE if grad is None, we assume the gradient is 1
            grad = GradTensor(np.ones_like(self.data))
        self.grad += grad  # type: ignore
        pprint(self)
        # print the name of the current variable
        if self.depends_on is not None:
            for dep in self.depends_on:
                # NOTE call the backward function of the dependency
                backward_grad = dep.grad_fn(grad)
                dep.inputs.backward(backward_grad)


def _neg(input: GradTensor) -> GradTensor:
    """Negate the input tensor."""
    data = -input.data
    requires_grad = input.requires_grad
    if requires_grad:
        # NOTE create a new dependency
        grad_fn = lambda grad: -grad
        dep = Dependency(
            op=_neg,
            inputs=input,
            grad_fn=grad_fn,
        )
        output = GradTensor(data, requires_grad, depends_on=[dep])
        return output
    else:
        return GradTensor(data, requires_grad)


def _add(input1: GradTensor, input2: GradTensor) -> GradTensor:
    """Add two tensors."""
    data = input1.data + input2.data
    requires_grad = input1.requires_grad or input2.requires_grad
    if requires_grad:
        # NOTE create a new dependency
        grad_fn = lambda grad: grad
        if input1.requires_grad:
            # NOTE create a new dependency
            dep1 = Dependency(
                op=_add,
                inputs=input1,
                grad_fn=grad_fn,
            )
        else:
            dep1 = None
        if input2.requires_grad:
            # NOTE create a new dependency
            dep2 = Dependency(
                op=_add,
                inputs=input2,
                grad_fn=grad_fn,
            )
        else:
            dep2 = None
        deps = [dep for dep in (dep1, dep2) if dep is not None]
        output = GradTensor(data, requires_grad, depends_on=deps)
        return output
    else:
        return GradTensor(data, requires_grad)


def _mul(input1: GradTensor, input2: GradTensor) -> GradTensor:
    """Multiply two tensors."""
    data = input1.data * input2.data
    requires_grad = input1.requires_grad or input2.requires_grad
    if requires_grad:
        # NOTE create a new dependency
        grad_fn = lambda grad: grad * input2.data
        if input1.requires_grad:
            # NOTE create a new dependency
            dep1 = Dependency(
                op=_mul,
                inputs=input1,
                grad_fn=grad_fn,
            )
        else:
            dep1 = None
        if input2.requires_grad:
            # NOTE create a new dependency
            dep2 = Dependency(
                op=_mul,
                inputs=input2,
                grad_fn=grad_fn,
            )
        else:
            dep2 = None
        deps = [dep for dep in (dep1, dep2) if dep is not None]
        output = GradTensor(data, requires_grad, depends_on=deps)
        return output
    else:
        return GradTensor(data, requires_grad)


def _matmul(input1: GradTensor, input2: GradTensor) -> GradTensor:
    """Matrix multiply two tensors."""
    data = input1.data @ input2.data
    requires_grad = input1.requires_grad or input2.requires_grad
    if requires_grad:
        # NOTE create a new dependency
        if input1.requires_grad:
            grad_fn = lambda grad: GradTensor(grad.data @ input2.data.T)
            dep1 = Dependency(
                op=_matmul,
                inputs=input1,
                grad_fn=grad_fn,
            )
        else:
            dep1 = None
        if input2.requires_grad:
            # NOTE create a new dependency
            grad_fn = lambda grad: GradTensor(input1.data.T @ grad.data)
            dep2 = Dependency(
                op=_matmul,
                inputs=input2,
                grad_fn=grad_fn,
            )
        else:
            dep2 = None
        deps = [dep for dep in (dep1, dep2) if dep is not None]
        output = GradTensor(data, requires_grad, depends_on=deps)
        return output
    else:
        return GradTensor(data, requires_grad)


def tensor_sum(input: GradTensor) -> GradTensor:
    """Sum the tensor on all axes."""
    data = input.data.sum()
    requires_grad = input.requires_grad
    if requires_grad:
        # NOTE create a new dependency
        grad_fn = lambda grad: GradTensor(np.ones_like(input.data)) * grad
        dep = Dependency(
            op=tensor_sum,
            inputs=input,
            grad_fn=grad_fn,
        )
        output = GradTensor(data, requires_grad, depends_on=[dep])
        return output
    else:
        return GradTensor(data, requires_grad)
