"""Point groups: Cn / Dn / T / O / I operators are rotation groups of the right order (CPU, instant)."""
import pytest
import torch

from celeris2d.abinitio import symmetry_ops


@pytest.mark.parametrize("sym,order", [("C1", 1), ("C8", 8), ("D2", 4), ("D7", 14), ("T", 12), ("O", 24), ("I", 60),
                                       ("I1", 60)])
def test_group_order_and_closure(sym, order):
    ops = symmetry_ops(sym).double()
    assert len(ops) == order
    assert torch.allclose(ops @ ops.transpose(1, 2), torch.eye(3, dtype=ops.dtype).expand_as(ops), atol=1e-5)
    assert torch.allclose(torch.linalg.det(ops), torch.ones(order, dtype=ops.dtype), atol=1e-5)
    prod = (ops[:, None] @ ops[None]).reshape(-1, 3, 3)
    d = (prod[:, None] - ops[None]).abs().amax((2, 3))           # every product is one of the operators
    assert bool((d.min(1).values < 1e-4).all())


def test_relion_axes():
    """RELION's frames: T a 3-fold on z, O a 4-fold on z, I (I2) a 2-fold on z and a 5-fold in the x-z plane."""
    def has(sym, axis, deg):
        k = torch.tensor(axis, dtype=torch.float64)
        k = k / k.norm()
        a = torch.deg2rad(torch.tensor(deg, dtype=torch.float64))
        K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float64) + torch.sin(a) * K + (1 - torch.cos(a)) * (K @ K)
        return bool(((symmetry_ops(sym).double() - R).abs().amax((1, 2)) < 1e-4).any())
    assert has("T", (0, 0, 1), 120) and has("O", (0, 0, 1), 90) and has("I", (0, 0, 1), 180)
    assert has("I", (0.525731112, 0, 0.850650808), 72) and has("I1", (1, 0, 0), 180)


def test_unknown_symmetry():
    with pytest.raises(ValueError):
        symmetry_ops("X3")
