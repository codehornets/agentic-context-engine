"""
ACE Observability Module

Provides production-grade observability for ACE framework using Opik.
Replaces custom explainability implementation with industry-standard tracing.
"""

from .opik_integration import OpikIntegration
from .tracers import ace_track, track_role, track_adaptation

__all__ = [
    "OpikIntegration",
    "ace_track",
    "track_role",
    "track_adaptation",
]