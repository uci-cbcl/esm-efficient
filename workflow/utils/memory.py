import torch


def benchmark_memory(fn, fn_kwargs, device):
    '''

    https://github.com/Dao-AILab/flash-attention/blob/1a2c3e8c25251fa30ebee074c27ecbf69c2bad2b/flash_attn/utils/benchmark.py#L258C1-L268C15
    '''
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(**fn_kwargs)
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated(device=device) / ((2**20) * 1000)
    torch.cuda.empty_cache()
    return mem
