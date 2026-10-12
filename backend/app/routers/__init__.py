from .step1_face import router as step1_router
from .step2_voice import router as step2_router
from .step3_text import router as step3_router
from .step4_stress import router as step4_router
from .step5_summary import router as step5_router

__all__ = [
    "step1_router",
    "step2_router",
    "step3_router",
    "step4_router",
    "step5_router"
]
