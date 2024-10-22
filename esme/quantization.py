'''
Experimental 8-bit quantization for PyTorch. Not recommended for production use.
'''
import torch
from torch import nn


def import_cublas_matmul_int8():
    try:
        from torch_cublas_matmul_int8 import matmul_int8
        return matmul_int8
    except ImportError as e:
        raise ImportError(
            "Please install matmul_int8 with "
            "`conda install nvidia::cuda-toolkit` and"
            "`pip install pytorch_cublas_matmul_int8`"
        ) from e


def quantize(X):
    scale = X.abs().amax(axis=1, keepdim=True)
    return (X * 127 / scale).to(torch.int8), scale.squeeze()


def dequantize(cX, scaleX, dtype=torch.float16):
    return (cX.float() * scaleX[:, None] / 127.).to(dtype)


def dequantize_matmul(cX, scaleA, scaleB, dtype=torch.float16):
    return (
        cX.float()
        * scaleA[:, None] / 127.
        * scaleB[None, :] / 127.
    ).to(dtype)


class MatMul8bit(torch.autograd.Function):

    @staticmethod
    def forward(A, cB, scaleB, threshold=6):
        return MatMul8bit._matmul(A, cB, scaleB, threshold)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        if ctx.needs_input_grad[0]:
            _, cB, scaleB, threshold = inputs
            ctx.save_for_backward(cB, scaleB)
            ctx.threshold = threshold
        if ctx.needs_input_grad[1]:
            raise NotImplementedError("Backward for weights not implemented")

    @staticmethod
    def _matmul(A, cB, scaleB, threshold):
        idx = A.abs().amax(axis=0) > threshold
        outlier_features = idx.any()

        if outlier_features:
            outliers = A[:, idx]
            _A = A.clone()
            _A[:, idx] = 0
        else:
            _A = A

        cA, scaleA = quantize(_A)

        matmul_int8 = import_cublas_matmul_int8()
        output = matmul_int8(cA, cB)
        result = dequantize_matmul(output, scaleA, scaleB, dtype=A.dtype)

        if outlier_features:
            result += outliers @ dequantize(cB[:, idx], scaleB,
                                            dtype=result.dtype).t()
        return result

    @staticmethod
    def backward(ctx, grad):

        cB, scaleB = ctx.saved_tensors

        B = dequantize(cB, scaleB, dtype=grad.dtype)
        cBt, scale_Bt = quantize(B.t())

        grad_A = MatMul8bit._matmul(grad, cBt, scale_Bt, ctx.threshold)
        return grad_A, None, None, None


class Linear8bit(torch.nn.Module):

    def __init__(self, weights, bias, device, threshold=6):
        super().__init__()
        self.out_features, self.in_features, = weights.shape

        self.bias = nn.Parameter(bias.to(device))
        cweight, scale = quantize(weights.to(device))

        self.register_buffer('cweight', cweight.contiguous())
        self.register_buffer('scale', scale.contiguous())
        self.threshold = threshold

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 2
        return MatMul8bit.apply(
            x, self.cweight, self.scale, self.threshold
        ) + self.bias

    def __repr__(self):
        return f"Linear8bit(in_features={self.in_features}, out_features={self.out_features}, threshold={self.threshold})"
