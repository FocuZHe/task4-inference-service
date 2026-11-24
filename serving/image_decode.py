import base64
import binascii

import cv2
import numpy as np

try:
    from pybase64 import b64decode as _fast_decode
except ImportError:
    _fast_decode = None


def decode_bgr(value: str) -> np.ndarray:
    if _fast_decode is None:
        raw = base64.b64decode(value)
    else:
        try:
            raw = _fast_decode(value, validate=True)
        except (binascii.Error, ValueError):
            # Preserve the standard decoder's permissive behavior for legacy payloads.
            raw = base64.b64decode(value)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("image could not be decoded")
    return image
