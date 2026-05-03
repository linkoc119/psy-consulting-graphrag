"""
Utils package for GraphRAG Psychology Chatbot
"""
from .prompts import (
    SYSTEM_PROMPT_COUNSELING,
    SYSTEM_PROMPT_CRISIS,
    TRIAGE_GUIDELINES,
    FEW_SHOT_COUNSELING,
    FEW_SHOT_CRISIS,
    build_triage_prompt,
    build_counseling_prompt,
    build_crisis_prompt
)

__all__ = [
    'SYSTEM_PROMPT_COUNSELING',
    'SYSTEM_PROMPT_CRISIS',
    'TRIAGE_GUIDELINES',
    'FEW_SHOT_COUNSELING',
    'FEW_SHOT_CRISIS',
    'build_triage_prompt',
    'build_counseling_prompt',
    'build_crisis_prompt'
]
