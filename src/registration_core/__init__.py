"""Shared registration engine (research Workflow C): global localization, overlap-restricted refinement,
validation and a common transform artifact.  Used by the Slurm runner, the Registration Explorer and the
Workflow C research mode.  The search / ICP / evaluation code is vendored unchanged from the 2026-10-14/19
analyses (see a2.py, search.py, pipeline.py headers); parity is tested in tests/test_registration_core.py.
"""
from .workflow import (CORE_VERSION, SCHEMA, SOURCE_LABELS, VERDICT_TEXT, Constraints, SideInput, WorkflowConfig,  # noqa: F401
                       run_workflow, similarity_from_landmarks, write_artifact)
