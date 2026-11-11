"""Registration Explorer: research-mode viewer for saved registrations (enabled with CELLREG_RESEARCH_MODE=1)."""
import os

RESEARCH_MODE_ENV = "CELLREG_RESEARCH_MODE"
PAGE_LABEL = "Registration Explorer (Research)"


def research_mode_enabled() -> bool:
    return os.environ.get(RESEARCH_MODE_ENV, "").strip().lower() in ("1", "true", "yes", "on")
