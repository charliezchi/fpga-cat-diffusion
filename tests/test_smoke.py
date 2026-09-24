def test_package_imports():
    import catdiff

    assert catdiff.__version__ == "0.1.0"


def test_torch_cpu_tensor_works():
    import torch

    t = torch.randn(2, 3)
    assert t.shape == (2, 3)
