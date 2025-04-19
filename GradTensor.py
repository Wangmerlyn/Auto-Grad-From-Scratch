from typing import Optional, List, NamedTuple, Callable, Union
import numpy as np


class Dependency(NamedTuple):
    """A class to represent a dependency in the computation graph."""

    op: Callable
    inputs: List["GradTensor"]
    output: "GradTensor"
    grad_fn: Optional[Callable] = None

    def __repr__(self) -> str:
        return f"Dependency(op={self.op.__name__}, inputs={self.inputs}, output={self.output})"


class GradTensor:
    def __init__(
        self, data: np.ndarray, requires_grad: bool = False, depends_on=None
    ) -> None:
        self.data = np.array(data)
        self.requires_grad = requires_grad
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None

    def zero_grad(self) -> None:
        """Zero the gradient of the tensor."""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
        else:
            raise ValueError(
                "Cannot zero grad of a tensor that does not require grad."
            )
    