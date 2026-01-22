from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, Boolean
from datetime import datetime
from database import Base
from sqlalchemy.orm import relationship
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, nullable=False)
class AIPrediction(Base):
    __tablename__ = "ai_predictions"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("users.id"))
    symptoms = Column(Text)
    condition = Column(String)
    risk = Column(String)
    confidence = Column(Float)  # ✅ ADD THIS
    recommendation = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="New")

class DoctorFeedback(Base):
    __tablename__ = "doctor_feedback"

    id = Column(Integer, primary_key=True)
    doctor_id = Column(Integer, ForeignKey("users.id"))
    patient_id = Column(Integer, ForeignKey("users.id"))
    feedback = Column(Text)

class DoctorRemark(Base):
    __tablename__ = "doctor_remarks"

    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("users.id"))
    patient_id = Column(Integer, ForeignKey("users.id"))
    remark = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
class AIFeedback(Base):
    __tablename__ = "ai_feedback"

    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("users.id"))
    feedback = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
class Prescription(Base):
    __tablename__ = "prescriptions"

    id = Column(Integer, primary_key=True, index=True)

    prediction_id = Column(
        Integer,
        ForeignKey("ai_predictions.id"),
        unique=True,
        nullable=False
    )

    doctor_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False
    )

    status = Column(
        String,
        default="PENDING"   # PENDING / CONFIRMED
    )

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships (optional but clean)
    prediction = relationship("AIPrediction")
    doctor = relationship("User")
class Payment(Base):
    __tablename__ = "payments"

    id = Column(Integer, primary_key=True, index=True)
    prescription_id = Column(Integer, ForeignKey("prescriptions.id"))
    doctor_fee = Column(Integer)
    final_amount = Column(Integer)
    status = Column(String, default="PENDING")

    utr = Column(String, unique=True, nullable=True)
    manager_utr = Column(String, nullable=True)

    utr_verified = Column(Boolean, default=False)  # ✅ correct
