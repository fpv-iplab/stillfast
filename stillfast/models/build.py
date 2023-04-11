from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")

def build_model(cfg):
    name = cfg.MODEL.NAME
    model = MODEL_REGISTRY.get(name)(cfg)
    return model