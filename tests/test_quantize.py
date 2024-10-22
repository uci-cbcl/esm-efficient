import pytest
import torch
from esme.quantization import MatMul8bit, Linear8bit, quantize, \
    dequantize, dequantize_matmul
from conftest import device


def test_quantize_dequantize():
    X = torch.normal(0, 0.5, size=(2048, 4096), device=device).half()
    torch.testing.assert_close(
        dequantize(*quantize(X)),
        X,
        rtol=1e-2, atol=1e-1
    )


def test_dequantize_matmul():
    A = torch.normal(0, .1, size=(4 * 2048, 4096), device=device).half()
    B = torch.normal(0, .1, size=(2048, 4096), device=device).half()

    cA, scaleA = quantize(A)
    cB, scaleB = quantize(B)

    torch.testing.assert_close(
        dequantize_matmul(cA.float() @ cB.t().float(), scaleA, scaleB),
        A @ B.t(),
        rtol=1e-2, atol=1e-1
    )


@pytest.mark.skip(reason="Experimental feature")
def test_MatMul8bit():

    A = torch.normal(0, .5, size=(2048, 256),
                     device=device, requires_grad=True, dtype=torch.float16)
    B = torch.normal(0, .1, size=(512, 256),
                     device=device, requires_grad=True, dtype=torch.float16)

    cB, scaleB = quantize(B)

    C = MatMul8bit.apply(A, cB, scaleB, 2)
    grads = torch.normal(0, .3, size=C.shape,
                         device=device, dtype=torch.float16)
    C.backward(grads)

    torch.testing.assert_close(
        A.float() @ B.t().float(),
        C.float(),
        rtol=1e-2, atol=1e-1
    )

    torch.testing.assert_close(
        A.grad.float(),
        grads.float() @ B.float(),
        rtol=1e-2, atol=1e-1
    )


@pytest.mark.skip(reason="Experimental feature")
def test_Linear8bit():

    A = torch.normal(0, .5, size=(8 * 2048, 4096),
                     device=device, requires_grad=True, dtype=torch.float16)

    _linear = torch.nn.Linear(4096, 1024, dtype=torch.float16, device=device)

    linear = Linear8bit(_linear.weight, _linear.bias,
                        threshold=2, device=device)
    y = linear(A)
    grads = torch.normal(0, .5, size=y.shape,
                         device=device, dtype=torch.float16)
    y.backward(grads)

    torch.testing.assert_close(
        y,
        A @ _linear.weight.t() + _linear.bias,
        rtol=1e-2, atol=1e-1
    )

    torch.testing.assert_close(
        A.grad.float(),
        grads.float() @ _linear.weight.float(),
        rtol=1e-2, atol=1e-1
    )
