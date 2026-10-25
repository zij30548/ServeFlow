"""Start the local ServeFlow backend without importing native modules early."""
from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


def main() -> None:
    configured = os.environ.get("ORT_DISABLE_TELEMETRY")
    if configured not in {None, "1"}:
        raise RuntimeError("ORT_DISABLE_TELEMETRY must be 1 before startup")
    if configured is None and "onnxruntime" in sys.modules:
        raise RuntimeError("Native runtime was initialized before the startup guard")
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)

    from api.main import app
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8816, workers=1, access_log=False)


if __name__ == "__main__":
    main()
