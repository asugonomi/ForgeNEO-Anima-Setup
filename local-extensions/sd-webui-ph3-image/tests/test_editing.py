from dataclasses import replace
import types
import pytest
import torch
from PIL import Image
from ph3_native.contracts import Request, PH3Error, Cancelled
from ph3_native.editing import scaled_size, size_within_area, edit_prompt, open_source, make_encoder_class
from ph3_native.sampling import sample


def test_scaled_dimensions_and_explicit_rounding():
    assert scaled_size(2560, 1440, 1) == (2560, 1440)
    assert scaled_size(2048, 2048, .5) == (1024, 1024)
    assert scaled_size(1000, 700, 1.5) == (1504, 1056)
    assert scaled_size(1040, 1040, 1) == (1056, 1056)
    for scale in (0, -1, float('nan'), float('inf'), 100, True):
        with pytest.raises(PH3Error): scaled_size(1024, 1024, scale)


def test_reference_size_and_prompt():
    assert size_within_area(2048, 2048, 1_000_000) == (992, 992)
    assert size_within_area(512, 512, 4_000_000) == (512, 512)
    assert edit_prompt('Add a vase.') == 'Edit <Picture 1>. Add a vase.'
    assert edit_prompt(' Edit <Picture 1>. Remove the cup.') == 'Edit <Picture 1>. Remove the cup.'


def test_source_rotation_and_request_size(tmp_path):
    p = tmp_path / 'source.png'
    exif = Image.Exif(); exif[274] = 6
    Image.new('RGB', (640, 960)).save(p, exif=exif)
    assert open_source(p).size == (960, 640)
    req = Request('remove object', 'd', 't', 'v', width=480, height=320, source_image=str(p), image_scale=.5)
    assert req.checked(check_files=False) is req
    with pytest.raises(PH3Error, match='一致'):
        replace(req, width=512).checked(check_files=False)


def test_sampler_passes_reference_and_visual_tags():
    class Dit:
        def preprocess_text_embeds(self, x): return x
        def forward_streams(self, x, t, context, transformer_options, minimax_payload):
            assert minimax_payload['refs'][0]['kind'] == 'image'
            assert torch.equal(minimax_payload['cond_video_latents'][0], z)
            assert torch.equal(minimax_payload['text_token_tags'], tags)
            return [torch.zeros_like(tensor) for tensor in x]
    z = torch.randn(1, 24, 1, 2, 2)
    tags = torch.tensor([1, 0, 0, 0, 1])
    refs = [{'kind': 'image', 'latent_h': 2, 'latent_w': 2, 'latent': z}]
    sample(Dit(), torch.zeros(1, 5, 2), 32, 32, 3, 'cpu', refs=refs, token_tags=tags)


def test_image_posterior_uses_fixed_reference_seed():
    class Base(torch.nn.Module):
        def _normalize_pixels(self, x): return x
        def _adaptive_encode(self, x):
            self.cancel_check()
            return torch.cat((torch.ones(1, 24, 1, 2, 2), torch.zeros(1, 24, 1, 2, 2)), dim=1)
    native = types.SimpleNamespace(MiniMaxH3VideoVAE=Base, EncoderFCN3D=lambda **kw: torch.nn.Identity(),
                                  LATENTS_MEAN=[0] * 24, LATENTS_STD=[1] * 24, IMAGENET_MEAN=[0] * 3, IMAGENET_STD=[1] * 3)
    model = make_encoder_class(native)()
    x = torch.zeros(1, 3, 1, 32, 32)
    expected = 1 + torch.randn((1, 24, 1, 2, 2), generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(model.encode(x), expected)
    assert torch.equal(model.encode(x), model.encode(x))
    def cancel(): raise Cancelled('stop')
    model.cancel_check = cancel
    with pytest.raises(Cancelled): model.encode(x)


def test_edit_callback_requires_image():
    from ph3_native.ui import run_edit
    result = run_edit(None, None, 1, 'Add a vase', 'd', 't', 'v', '1', 1, 6, 256, True, True, 1024)
    assert result[0] == [] and '画像を選択' in result[1]


def test_image_input_with_forge_cache_override(tmp_path, monkeypatch):
    import gradio as gr
    from gradio.data_classes import FileData
    from ph3_native.ui import EditImage
    path = tmp_path / 'input.png'
    Image.new('RGB', (256, 256), 'red').save(path)
    # Same signature as this Forge's modules.ui_tempdir.save_pil_to_file.
    def forge_writer(pil_image, cache_dir=None, format='png'):
        raise AssertionError('The extension must reuse the validated upload.')
    monkeypatch.setattr(gr.processing_utils, 'save_pil_to_cache', forge_writer)
    payload = FileData(path=str(path), orig_name='input.png')
    with pytest.raises(TypeError, match='name'):
        gr.Image(type='filepath').preprocess(payload)
    component = EditImage(type='filepath')
    assert component.preprocess(payload) == str(path)
    assert component.preprocess(None) is None
    assert component.get_block_name() == 'image'
    invalid = tmp_path / 'invalid.png'; invalid.write_text('not an image')
    with pytest.raises(Exception): component.preprocess(FileData(path=str(invalid)))


def test_reference_limit_and_main_dimensions(tmp_path):
    p = tmp_path / 'main.png'; Image.new('RGB', (512, 768)).save(p)
    ref = tmp_path / 'reference.png'; Image.new('RGB', (1024, 1024)).save(ref)
    req = Request('Use picture 2', 'd', 't', 'v', width=512, height=768,
                  source_image=str(p), reference_images=(str(ref),))
    assert req.checked(check_files=False) is req
    with pytest.raises(PH3Error, match='最大7枚'):
        replace(req, reference_images=(str(ref),) * 8).checked(check_files=False)


def test_output_setting_uses_sibling_folder(tmp_path, monkeypatch):
    import sys
    from ph3_native.settings import output_root, default_output_dir, register_settings
    opts = types.SimpleNamespace(outdir_samples='', outdir_txt2img_samples=str(tmp_path / 'txt2img-images'),
                                 ph3_image_output_dir='')
    recorded = {}
    opts.add_option = lambda key, value: recorded.update({key: value})
    modules = types.ModuleType('modules')
    modules.shared = types.SimpleNamespace(opts=opts, OptionInfo=lambda *a, **kw: (a, kw))
    modules.paths_internal = types.SimpleNamespace(script_path=str(tmp_path / 'forge'))
    monkeypatch.setitem(sys.modules, 'modules', modules)
    assert default_output_dir() == tmp_path / 'p-h3-images'
    assert output_root() == tmp_path / 'p-h3-images'
    register_settings()
    assert recorded['ph3_image_output_dir'][0][0] == str(tmp_path / 'p-h3-images')
    opts.ph3_image_output_dir = str(tmp_path / 'custom-edits')
    assert output_root() == tmp_path / 'custom-edits'


def test_reference_composition_has_no_main_size_dependency(tmp_path):
    from ph3_native.editing import reference_prompt
    p = tmp_path / 'reference.png'; Image.new('RGB', (640, 960)).save(p)
    req = Request('All three people dancing.', 'd', 't', 'v', mode='reference',
                  reference_images=(str(p),) * 3, width=2048, height=2048)
    assert req.checked(check_files=False) is req
    assert req.source_image is None
    text = reference_prompt(req.prompt, 3)
    assert all(f'<Picture {n}>' in text for n in (1, 2, 3))
    assert not text.startswith('Edit ')
    with pytest.raises(PH3Error): replace(req, source_image=str(p)).checked(check_files=False)
    with pytest.raises(PH3Error): replace(req, reference_images=()).checked(check_files=False)
    with pytest.raises(PH3Error): replace(req, reference_images=(str(p),) * 9).checked(check_files=False)
    with pytest.raises(PH3Error, match='32の倍数'): replace(req, width=2001).checked(check_files=False)


def test_reference_callback_requires_images():
    from ph3_native.ui import run_reference
    result = run_reference(None, 'Three friends.', 2048, 2048, 'd', 't', 'v', '1', 1, 6, 256, True, True, 1024)
    assert result[0] == [] and '1〜8枚' in result[1]


def test_all_canvas_presets_are_valid():
    from ph3_native.ui import CANVAS_PRESETS
    for width, height in CANVAS_PRESETS.values():
        Request('test', 'd', 't', 'v', width=width, height=height).checked(check_files=False)
