"""
Diffusion with Britney — Autoregressive-Guided Video Diffusion, DP Transitional Grammar
& 3-Movement Music Video Continuity Toolkit.
"""

from .presets import (
    GenerativeParadigmComparison,
    WorldStateObject,
    ShotTransitionSpec,
    MusicVideoMovementShot,
    MusicVideoDiffusionPack,
    GENERATIVE_PARADIGMS,
    CHURCH_DOOR_CONTINUITY_CASE,
    MUSIC_VIDEO_PACKS,
    get_mv_pack,
    list_mv_packs,
)
from .engine import (
    AutoregressiveWorldStateLedger,
    simulate_pure_vs_ar_guided_diffusion,
    build_sft_music_video_diffusion_tuples,
    export_diffusion_storyboard_json,
)

__all__ = [
    "GenerativeParadigmComparison",
    "WorldStateObject",
    "ShotTransitionSpec",
    "MusicVideoMovementShot",
    "MusicVideoDiffusionPack",
    "GENERATIVE_PARADIGMS",
    "CHURCH_DOOR_CONTINUITY_CASE",
    "MUSIC_VIDEO_PACKS",
    "get_mv_pack",
    "list_mv_packs",
    "AutoregressiveWorldStateLedger",
    "simulate_pure_vs_ar_guided_diffusion",
    "build_sft_music_video_diffusion_tuples",
    "export_diffusion_storyboard_json",
]
