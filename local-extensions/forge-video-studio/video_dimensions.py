"""H3 spatial sizing, independent of the Forge runtime."""
import math
from PIL import ImageOps

AUTO = '入力画像に合わせる'

def dimensions(image, mode, size, scale=1.0):
    scale = float(scale)
    if not math.isfinite(scale) or not 0.125 <= scale <= 2:
        raise ValueError('縮尺は0.125〜2×で指定してください。')
    if size == AUTO:
        if image is None or mode == 'テキスト→動画':
            raise ValueError('開始画像を指定するか、固定解像度を選んでください。')
        w, h = ImageOps.exif_transpose(image).size
        w, h = w * scale, h * scale
    else:
        w, h = map(int, size.split(' × '))
    # Round to H3's 32-pixel grid; never silently cap the requested size.
    return tuple(max(64, int(v / 32 + 0.5) * 32) for v in (w, h))

def describe(image, mode, size, scale):
    try:
        w, h = dimensions(image, mode, size, scale)
        extra = ' 大きな解像度は時間とVRAMを多く使います。' if w*h > 768*768 else ''
        return f'生成サイズ: {w} × {h} px（32px単位に調整）' + extra
    except (ValueError, TypeError) as e:
        return str(e)
