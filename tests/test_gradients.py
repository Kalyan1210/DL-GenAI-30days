import torch
import pytest
from src.models.mlp import TwoLayerMLP
from src.manual_grad import forward_manual, backward_manual

torch.manual_seed(0)


@pytest.mark.parametrize("batch", [8])
def test_manual_grads_match_autograd(batch):
    x = torch.randn(batch, 28 * 28)
    y = torch.randint(0, 10, (batch,))

    model = TwoLayerMLP()
    loss, cache = forward_manual(x, y, model)
    dW1_m, db1_m, dW2_m, db2_m = backward_manual(cache, model)

    # autograd
    model.zero_grad()
    logits = model(x)
    loss_auto = torch.nn.functional.cross_entropy(logits, y)
    loss_auto.backward()

    dW1_a = model.net[0].weight.grad.T
    db1_a = model.net[0].bias.grad
    dW2_a = model.net[2].weight.grad.T
    db2_a = model.net[2].bias.grad

    for name, m, a in [
        ("W1", dW1_m, dW1_a),
        ("b1", db1_m, db1_a),
        ("W2", dW2_m, dW2_a),
        ("b2", db2_m, db2_a),
    ]:
        assert torch.allclose(
            m, a, atol=1e-6, rtol=1e-4
        ), f"{name} gradients differ"
