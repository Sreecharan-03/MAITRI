from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any

# Step 1: Face Schema
class FaceResponse(BaseModel):
    dominant_emotion: str = Field(default="Neutral", description="Dominant emotion detected (e.g. Happiness, Neutral, Sadness, etc.)")
    confidence: float = Field(default=0.85, description="Confidence score between 0.0 and 1.0")
    scores: Dict[str, float] = Field(default_factory=dict, description="Probability breakdown across emotion classes")
    face_detected: bool = Field(default=True, description="Whether a face was detected in the frame/stream")
    frames_analyzed: Optional[int] = Field(default=1, description="Total frames processed")
    duration_seconds: Optional[float] = Field(default=None, description="Duration in seconds if video/stream processed")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional frame statistics or bounding boxes")

# Step 2: Voice Schema
class VoiceResponse(BaseModel):
    dominant_emotion: str = Field(default="Calm", description="Dominant vocal emotion (e.g. calm, tense, sadness)")
    confidence: float = Field(default=0.84, description="Confidence score between 0.0 and 1.0")
    pitch_hz: float = Field(default=118.2, description="Mean fundamental pitch in Hz")
    scores: Dict[str, float] = Field(default_factory=dict, description="Probability breakdown across acoustic emotion classes")
    details: Optional[Dict[str, Any]] = Field(default=None)

# Step 3: Text Schema
class TextRequest(BaseModel):
    text: str = Field(
        default="Nominal astronaut shift.",
        example="Feeling slightly fatigued during the EVA airlock cycle, but mission tasks are progressing according to plan."
    )

class TextResponse(BaseModel):
    dominant_emotion: str = Field(default="Reflective", description="Dominant linguistic sentiment/emotion")
    confidence: float = Field(default=0.78, description="Confidence score between 0.0 and 1.0")
    detected_8_classes: Dict[str, float] = Field(default_factory=dict, description="Probability breakdown for 8 GoEmotions classes")
    details: Optional[Dict[str, Any]] = Field(default=None)

# Step 4: Wearable Stress Schema (WESAD-style signals)
class WearableDataRequest(BaseModel):
    heart_rate_bpm: Optional[float] = Field(default=None, example=78.0)
    heart_rate: Optional[float] = Field(default=None)
    heartRate: Optional[float] = Field(default=None)
    
    spo2_percent: Optional[float] = Field(default=None, example=97.0)
    spo2: Optional[float] = Field(default=None)
    
    skin_temp_celsius: Optional[float] = Field(default=None, example=36.5)
    skin_temp: Optional[float] = Field(default=None)
    skinTemp: Optional[float] = Field(default=None)
    
    accelerometer_magnitude: Optional[float] = Field(default=1.02)
    hrv_rmssd: Optional[float] = Field(default=None)
    hrvRmssd: Optional[float] = Field(default=None)
    mean_eda: Optional[float] = Field(default=None)
    meanEda: Optional[float] = Field(default=None)
    scr_count: Optional[float] = Field(default=None)
    scrCount: Optional[float] = Field(default=None)
    respiration_rate: Optional[float] = Field(default=None)
    respirationRate: Optional[float] = Field(default=None)

    def get_hr(self) -> float:
        for v in (self.heart_rate_bpm, self.heart_rate, self.heartRate):
            if v is not None:
                return float(v)
        return 78.0

    def get_spo2(self) -> float:
        for v in (self.spo2_percent, self.spo2):
            if v is not None:
                return float(v)
        return 97.0

    def get_temp(self) -> float:
        for v in (self.skin_temp_celsius, self.skin_temp, self.skinTemp):
            if v is not None:
                return float(v)
        return 36.5

    def get_accel(self) -> float:
        return float(self.accelerometer_magnitude or 1.02)

class StressResponse(BaseModel):
    stress_level: str = Field(default="Normal", description="Stress category (e.g. 'Normal', 'Mild Stress', 'High Stress')")
    stress_score: float = Field(default=0.25, description="Continuous stress index from 0.0 to 1.0")
    discordance_flag: bool = Field(default=False, description="True if signs of suppression or autonomic strain are detected")
    details: Optional[Dict[str, Any]] = Field(default=None)

# Step 5: Final Aggregation & LLM Analysis Request
class AstronautSummaryRequest(BaseModel):
    astronaut_id: Optional[str] = Field(default="ASTRO-01", description="Astronaut crew identifier")
    mission_day: Optional[int] = Field(default=42, description="Mission elapsed day (SOL)")
    
    face_result: Optional[FaceResponse] = None
    face_data: Optional[Any] = None
    face: Optional[Any] = None
    
    voice_result: Optional[VoiceResponse] = None
    voice_data: Optional[Any] = None
    voice: Optional[Any] = None
    
    text_result: Optional[TextResponse] = None
    text_data: Optional[Any] = None
    text: Optional[Any] = None
    
    stress_result: Optional[StressResponse] = None
    stress_data: Optional[Any] = None
    vitals_data: Optional[Any] = None
    vitals: Optional[Any] = None
    
    custom_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    apiKey: Optional[str] = None

class AstronautSummaryResponse(BaseModel):
    overall_wellbeing_score: float = Field(default=7.8, description="Overall score out of 10.0")
    psychological_status: str = Field(default="Stable", description="High-level status summary")
    astronaut_specific_assessment: str = Field(..., description="Comprehensive clinical psychological assessment")
    recommended_actions: List[str] = Field(default_factory=list, description="List of operational countermeasures and recovery protocols")
    discordance_analysis: Optional[str] = Field(default=None, description="Analysis of cross-modal alignment/mismatch")
