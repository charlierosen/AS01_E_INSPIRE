"""
Pipeline orchestration utilities for the refactored analysis package.
"""

from .clustering import DoRClusteringPipeline, ClusteringArtifacts
from .stacking import SpectralStacker, StackingSummary
from .ppxf import PPXFPipeline
from .full import FullPipeline, PipelineResult

__all__ = [
    "DoRClusteringPipeline",
    "ClusteringArtifacts",
    "SpectralStacker",
    "StackingSummary",
    "PPXFPipeline",
    "FullPipeline",
    "PipelineResult",
]
