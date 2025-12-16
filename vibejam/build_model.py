# vibejam/build_model.py
from vibejam.model import GPTModel
# labs imported lazily to avoid polluting stable layer

def build_model(arch:str, cfg):
    if arch in ("gpt", "transformer"):
        return GPTModel(cfg)
    elif arch == "rwkv_lite":
        from labs.rwkv_lite import RWKVLiteLM
        return RWKVLiteLM(cfg)
    else: 
        raise ValueError(f"Unknown arch={arch}")