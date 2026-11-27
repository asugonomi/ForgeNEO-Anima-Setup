# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path


def default_output_dir():
    from modules import shared, paths_internal
    root = Path(paths_internal.script_path)
    common = getattr(shared.opts, "outdir_samples", "")
    if common:
        base = Path(common).expanduser()
        base = base if base.is_absolute() else root / base
    else:
        samples = Path(getattr(shared.opts, "outdir_txt2img_samples", "") or "outputs/txt2img-images").expanduser()
        samples = samples if samples.is_absolute() else root / samples
        base = samples.parent
    return (base / "p-h3-images").resolve()


def output_root():
    from modules import shared, paths_internal
    value = getattr(shared.opts, "ph3_image_output_dir", "")
    if not value or not str(value).strip():
        return default_output_dir()
    path = Path(str(value).strip()).expanduser()
    return path.resolve() if path.is_absolute() else (Path(paths_internal.script_path) / path).resolve()


def register_settings():
    from modules import shared
    shared.opts.add_option("ph3_image_output_dir", shared.OptionInfo(
        str(default_output_dir()), "P-H3画像編集の保存先（日付フォルダーをこの下に作成）",
        section=("ph3_image", "P-H3 画像編集")))
