# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from modules import script_callbacks
from ph3_native.ui import build_tab
from ph3_native.settings import register_settings
script_callbacks.on_ui_settings(register_settings)
script_callbacks.on_ui_tabs(build_tab)
