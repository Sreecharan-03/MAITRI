"""
Services module for LLM synthesis and flight surgeon reporting.
Re-exports from app.llm_service for unified access across the codebase.
"""
from app.llm_service import (
    SYSTEM_INSTRUCTION,
    generate_flight_surgeon_report,
    generate_astronaut_report
)

__all__ = [
    "SYSTEM_INSTRUCTION",
    "generate_flight_surgeon_report",
    "generate_astronaut_report"
]
