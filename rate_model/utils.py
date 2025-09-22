import json, numpy as np


def json_safe(obj):
    if isinstance(obj, np.generic):        # np.float32, np.int64, etc.
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    return obj
