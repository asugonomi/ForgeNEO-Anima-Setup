# SPDX-License-Identifier: AGPL-3.0-or-later
# Keep this file ASCII: Gradio's component metaclass reads it in the Windows locale.
import gradio as gr


class EditImage(gr.Image):
    """Reuse a validated upload without invoking Forge's incompatible cache writer."""
    is_template = True

    def preprocess(self, payload):
        if payload is None:
            return None
        from .editing import open_source
        # Gradio Blocks already checked the upload/cache path before this call.
        open_source(payload.path)
        return str(payload.path)
