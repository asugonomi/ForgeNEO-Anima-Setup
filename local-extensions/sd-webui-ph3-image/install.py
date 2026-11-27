# No automatic pip install, model download, git pull or changes to Forge.
from pathlib import Path
if not (Path(__file__).parent / "_vendor" / "h3" / "manifest.json").is_file():
    print("[P-H3 Image] Experimental extension. Run setup_native.bat manually; no automatic installation is performed.")
