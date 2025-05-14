# Aim: train a neural network to learn the solution of the following ODE
#
# df/dx = 5(4x^3 + x^2 - 2x - 0.5)
# f(0) = 0
#

from __future__ import annotations

from typing import Callable

import torch


class FFNN(torch.nn.Module):
    def __init__(self, layers: list[int], activation_function: Callable = torch.nn.GELU) -> None:
        if len(layers) < 2:
            raise ValueError("You must specify at least one input and one output layer.")

        self.layers = layers
        self.activation_function = activation_function

        sequence = []
        for n_i, n_o in zip(self.layers[:-2], self.layers[1:-1]):
            sequence.append(torch.nn.Linear(n_i, n_o))
            sequence.append(self.activation_function())

        sequence.append(torch.nn.Linear(self.layers[-2], self.layers[-1]))
        self.nn = torch.nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)


def main():
    def ode_rhs(x: torch.Tensor) -> torch.Tensor:
        return 5 * (4 * x**3 + x**2 - 2 * x - 0.5)

    model = FFNN(layers=[1, 10, 10, 10, 1])

    def ode_lhs(x: torch.Tensor) -> torch.Tensor:
        grad = torch.autograd.grad(
            outputs=model(x),
            inputs=x,
            grad_outputs=torch.ones_like(x),
            create_graph=True,
            retain_graph=True,
        )[0]
        return grad

    # def loss_function(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
    #     loss, metrics = None, {}
    #     # we need access to individual loss terms - call backward() in loop
    #     for key, metric in metrics.items():
    #         metric.backward(retain_graph=True)
    #         gradients[key] = {
    #             name: torch.clone(param.grad.flatten())
    #             for name, param in model.named_parameters()
    #             if param.grad is not None
    #         }
    #     return loss, metrics


if __name__ == "__main__":
    main()
