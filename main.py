from fastapi import FastAPI, Request, Form, Depends, Cookie, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
from models import  Payment, Prescription, AIPrediction
from models import DoctorFeedback
from models import AIFeedback
from models import DoctorRemark
import os
os.makedirs("static/uploads", exist_ok=True)
from fastapi import UploadFile, File
import io
import pytesseract
from PIL import Image
import re
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from database import engine, SessionLocal
from models import Base, User
Base.metadata.create_all(bind=engine)

def analyze_symptoms_text(symptom_text: str):
    text = symptom_text.lower()

    if "chest" in text and ("pain" in text or "breath" in text):
        return {
            "condition": "Possible Cardiac Risk",
            "risk": "High",
            "confidence": "85%",
            "recommendation": "Seek immediate medical attention."
        }

    if "fever" in text and "cough" in text:
        return {
            "condition": "Respiratory Infection",
            "risk": "Medium",
            "confidence": "75%",
            "recommendation": "Consult a doctor for further evaluation."
        }

    if "headache" in text or "nausea" in text:
        return {
            "condition": "Migraine or Viral Illness",
            "risk": "Low",
            "confidence": "65%",
            "recommendation": "Rest, hydrate, and monitor symptoms."
        }

    return {
        "condition": "General Symptoms",
        "risk": "Low",
        "confidence": "50%",
        "recommendation": "Monitor symptoms and consult a doctor if they persist."
    }
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="AI Medical Consultancy & Virtual Doctor Platform")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/about")
def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request, "title": "About Us"})

@app.get("/contact")
def contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request, "title": "Contact Us"})

@app.get("/privacy")
def privacy(request: Request):
    return templates.TemplateResponse("privacy.html", {"request": request, "title": "Privacy Policy"})

@app.get("/terms")
def terms(request: Request):
    return templates.TemplateResponse("terms.html", {"request": request, "title": "Terms & Conditions"})

@app.get("/disclaimer")
def disclaimer(request: Request):
    return templates.TemplateResponse("disclaimer.html", {"request": request, "title": "Medical Disclaimer"})

@app.get("/faq")
def faq(request: Request):
    return templates.TemplateResponse("faq.html", {"request": request, "title": "FAQs"})

@app.get("/patient-guidelines")
def patient_guidelines(request: Request):
    return templates.TemplateResponse("patient_guidelines.html", {"request": request, "title": "Patient Guidelines"})

@app.get("/doctor-guidelines")
def doctor_guidelines(request: Request):
    return templates.TemplateResponse("doctor_guidelines.html", {"request": request, "title": "Doctor Guidelines"})

@app.get("/refund-policy")
def refund_policy(request: Request):
    return templates.TemplateResponse("refund_policy.html", {"request": request, "title": "Refund & Cancellation Policy"})

@app.get("/support")
def support(request: Request):
    return templates.TemplateResponse("support.html", {"request": request, "title": "Help & Support"})


# --------------------------------------------------
# PASSWORD HASHING
# --------------------------------------------------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    # bcrypt limit is 72 bytes
    return pwd_context.hash(password[:72])
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password[:72], hashed_password)
@app.on_event("startup")
def create_default_users():
    db = SessionLocal()

    admin = db.query(User).filter(User.email == "admin@gmail.com").first()
    if not admin:
        admin = User(
            name="System Admin",     # ✅ REQUIRED
            age=35,                  # ✅ REQUIRED
            gender="Other",          # ✅ REQUIRED
            email="admin@gmail.com",
            password=hash_password("admin123"),
            role="admin"
        )
        db.add(admin)
        db.commit()

    db.close()


    print("✅ Default users created")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

Base.metadata.create_all(bind=engine)

# --------------------------------------------------
# JWT CONFIG
# --------------------------------------------------

SECRET_KEY = "very-secret-key-change-later"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None

# --------------------------------------------------
# DATABASE SESSION
# --------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --------------------------------------------------
# AUTH DEPENDENCY
# --------------------------------------------------

def get_current_user(access_token: str = Cookie(None)):
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = verify_token(access_token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    return payload

# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def welcome(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------------- LOGIN ---------------- #

@app.get("/login", response_class=HTMLResponse)
def login_page(
    request: Request,
    role: str,
    success: str = None,
    error: str = None
):
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "role": role,
            "success": success,
            "error": error
        }
    )

@app.post("/login")
def login(
    role: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(
        User.email == email,
        User.role == role
    ).first()

    if not user or not verify_password(password, user.password):
        return RedirectResponse(
            url=f"/login?role={role}&error=1",
            status_code=302
        )

    token = create_access_token(
    data={
        "sub": user.email,
        "role": user.role,
        "id": user.id
    }
)


    response = RedirectResponse(
        url=f"/{role}/dashboard",
        status_code=302
    )
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        path="/"
    )
    return response

# ---------------- REGISTER (PATIENT ONLY) ---------------- #

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request, error: str = None):
    return templates.TemplateResponse(
        "register.html",
        {"request": request, "error": error}
    )

@app.post("/register")
def register(
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        return RedirectResponse(
            url="/register?error=exists",
            status_code=302
        )

    new_user = User(
        name=name,
        age=age,
        gender=gender,
        email=email,
        password=hash_password(password),
        role="patient"
    )

    db.add(new_user)
    db.commit()

    return RedirectResponse(
        url="/login?role=patient&success=1",
        status_code=302
    )

# ---------------- DASHBOARDS ---------------- #

@app.get("/patient/dashboard", response_class=HTMLResponse)
def patient_dashboard(
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "patient":
        raise HTTPException(status_code=403)

    remarks = db.query(DoctorRemark).filter(
        DoctorRemark.patient_id == user["id"]
    ).order_by(DoctorRemark.id.desc()).all()

    latest_prediction = db.query(AIPrediction).filter(
        AIPrediction.patient_id == user["id"]
    ).order_by(AIPrediction.id.desc()).first()

    return templates.TemplateResponse(
        "patient_dashboard.html",
        {
            "request": request,
            "remarks": remarks,
            "prediction": latest_prediction
        }
    )

@app.get("/patient/payment-status/{prediction_id}", response_class=HTMLResponse)
def patient_payment_status(
    prediction_id: int,
    request: Request,
    error: str = None,   # 👈 ADD THIS
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):

    if user["role"] != "patient":
        raise HTTPException(status_code=403)

    # Get prediction
    prediction = db.query(AIPrediction).filter(
        AIPrediction.id == prediction_id,
        AIPrediction.patient_id == user["id"]
    ).first()

    if not prediction:
        raise HTTPException(status_code=404)

    # Get prescription
    prescription = db.query(Prescription).filter(
        Prescription.prediction_id == prediction_id
    ).first()

    # Default messages
    message = "Waiting for doctor confirmation"
    payment = None

    if prescription:
        if prescription.status == "CONFIRMED":
            message = "Doctor reviewing fee"

            payment = db.query(Payment).filter(
                Payment.prescription_id == prescription.id
            ).first()

            if payment and payment.status == "SUGGESTED":
                message = "Payment suggested by doctor"

    return templates.TemplateResponse(
    "patient_payment_status.html",
    {
        "request": request,
        "message": message,
        "payment": payment,
        "error": error   # 👈 ADD THIS
    }
)

@app.post("/patient/submit-utr")
async def submit_utr(
    request: Request,
    utr_number: str = Form(...),
    payment_id: int = Form(...),
    screenshot: UploadFile | None = File(None),
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    if user["role"] != "patient":
        raise HTTPException(status_code=403, detail="Unauthorized")

    typed_utr = utr_number.strip()

    # Validate UTR format
    if not typed_utr.startswith("UTR") or not typed_utr[3:].isalnum():
        raise HTTPException(status_code=400, detail="Invalid UTR format")

    # Duplicate UTR check
    exists = db.query(Payment).filter(Payment.utr == typed_utr).first()
    if exists:
        return RedirectResponse(
        url=f"/patient/payment-status/{payment_id}?error=utr_exists",
        status_code=302
    )


    # OCR checking
    ocr_utr = None
    if screenshot:
        contents = await screenshot.read()
        img = Image.open(io.BytesIO(contents))
        ocr_text = pytesseract.image_to_string(img)

        match = re.search(r"(UTR[0-9A-Za-z]+)", ocr_text)
        if match:
            ocr_utr = match.group(1).strip()

    if ocr_utr and ocr_utr != typed_utr:
        return RedirectResponse(
        url=f"/patient/payment-status/{payment_id}?error=utr_mismatch",
        status_code=302
    )

    # Save payment
    payment = db.query(Payment).filter(Payment.id == payment_id).first()
    if not payment:
        raise HTTPException(status_code=404, detail="Payment request not found")

    payment.utr = typed_utr
    payment.status = "PENDING"
    payment.uploaded_at = datetime.now()

    # Save screenshot file
    if screenshot:
        filename = f"utr_{payment_id}_{typed_utr}.png"
        save_path = os.path.join("static", "uploads", filename)
        with open(save_path, "wb") as f:
            f.write(contents)
        payment.screenshot = filename

    db.commit()
    return RedirectResponse(f"/patient/payment-status/{payment.id}", status_code=302)

@app.get("/doctor/dashboard", response_class=HTMLResponse)
def doctor_dashboard(request: Request, user=Depends(get_current_user)):
    if user["role"] != "doctor":
        raise HTTPException(status_code=403)
    return templates.TemplateResponse("doctor_dashboard.html", {"request": request})
@app.get("/doctor/ai-summaries", response_class=HTMLResponse)
def doctor_ai_summaries(
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "doctor":
        raise HTTPException(status_code=403)

    predictions = (
    db.query(AIPrediction, User, Prescription, Payment)
    .join(User, User.id == AIPrediction.patient_id)
    .outerjoin(Prescription, Prescription.prediction_id == AIPrediction.id)
    .outerjoin(Payment, Payment.prescription_id == Prescription.id)
    .all()
)


    return templates.TemplateResponse(
        "doctor_ai_summaries.html",
        {
            "request": request,
            "predictions": predictions
        }
    )
@app.post("/doctor/remark")
def submit_doctor_remark(
    patient_id: int = Form(...),
    remark: str = Form(...),
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "doctor":
        raise HTTPException(status_code=403)

    entry = DoctorRemark(
        doctor_id=user["id"],
        patient_id=patient_id,
        remark=remark
    )

    db.add(entry)
    db.commit()

    return RedirectResponse("/doctor/ai-summaries", status_code=302)

@app.get("/doctor/cases", response_class=HTMLResponse)
def doctor_cases(
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "doctor":
        raise HTTPException(status_code=403)

    predictions = (
        db.query(AIPrediction, User)
        .join(User, AIPrediction.patient_id == User.id)
        .all()
    )

    return templates.TemplateResponse(
    "doctor_ai_summaries.html",
    {
        "request": request,
        "predictions": predictions
    }
)


@app.get("/doctor/feedback", response_class=HTMLResponse)
def feedback_page(
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "doctor":
        raise HTTPException(status_code=403)

    patients = db.query(User).filter(User.role == "patient").all()

    return templates.TemplateResponse(
        "doctor_feedback.html",
        {
            "request": request,
            "patients": patients
        }
    )
@app.post("/doctor/ai-feedback")
def submit_ai_feedback(
    feedback: str = Form(...),
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "doctor":
        raise HTTPException(status_code=403)

    entry = AIFeedback(
        doctor_id=user["id"],
        feedback=feedback
    )

    db.add(entry)
    db.commit()

    return RedirectResponse("/doctor/dashboard", status_code=302)
@app.post("/doctor/patient-remark")
def submit_patient_remark(
    patient_id: int = Form(...),
    remark: str = Form(...),
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "doctor":
        raise HTTPException(status_code=403)

    entry = DoctorRemark(
        doctor_id=user["id"],
        patient_id=patient_id,
        remark=remark
    )

    db.add(entry)
    db.commit()

    # 👇 ADD THIS PART
    db.query(AIPrediction).filter(
        AIPrediction.patient_id == patient_id
    ).update({"status": "Reviewed"})

    db.commit()
    return RedirectResponse("/doctor/feedback?success=1", status_code=302)
@app.post("/doctor/confirm-prescription")
def confirm_prescription(
    prediction_id: int = Form(...),
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "doctor":
        raise HTTPException(status_code=403)

    prediction = db.query(AIPrediction).filter(
        AIPrediction.id == prediction_id
    ).first()

    if not prediction:
        raise HTTPException(status_code=404)

    # ✅ UPDATE AI CASE STATUS
    prediction.status = "REVIEWED"

    prescription = db.query(Prescription).filter(
        Prescription.prediction_id == prediction_id
    ).first()

    if not prescription:
        prescription = Prescription(
            prediction_id=prediction_id,
            doctor_id=user["id"],
            status="CONFIRMED"
        )
        db.add(prescription)
    else:
        prescription.status = "CONFIRMED"

    db.commit()

    return RedirectResponse("/doctor/dashboard", status_code=302)

@app.post("/doctor/suggest-payment")
def suggest_payment(
    prescription_id: int = Form(...),
    doctor_fee: int = Form(...),
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Role check
    if user["role"] != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can suggest payment")

    # Fetch prescription
    prescription = db.query(Prescription).filter(
        Prescription.id == prescription_id
    ).first()

    if not prescription:
        raise HTTPException(status_code=404, detail="Prescription not found")

    # Ensure prescription is confirmed
    if prescription.status != "CONFIRMED":
        raise HTTPException(
            status_code=400,
            detail="Payment can be suggested only after prescription confirmation"
        )

    # Calculate final amount (15% platform charge)
    final_amount = int(doctor_fee + (doctor_fee * 0.15))

    # Check if payment already exists
    payment = db.query(Payment).filter(
        Payment.prescription_id == prescription.id
    ).first()

    if not payment:
        payment = Payment(
            prescription_id=prescription.id,
            doctor_fee=doctor_fee,
            final_amount=final_amount,
            status="SUGGESTED"
        )
        db.add(payment)
    else:
        payment.doctor_fee = doctor_fee
        payment.final_amount = final_amount
        payment.status = "SUGGESTED"

    db.commit()

    return RedirectResponse(
        url="/doctor/dashboard",
        status_code=302
    )

@app.get("/admin/medical-feedback", response_class=HTMLResponse)
def admin_medical_feedback(
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "admin":
        raise HTTPException(status_code=403)

    feedbacks = (
        db.query(AIFeedback, User)
        .join(User, AIFeedback.doctor_id == User.id)
        .all()
    )

    return templates.TemplateResponse(
        "admin_medical_feedback.html",
        {
            "request": request,
            "feedbacks": feedbacks
        }
    )
@app.get("/admin/analytics", response_class=HTMLResponse)
def admin_analytics(
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "admin":
        raise HTTPException(status_code=403)

    total_patients = db.query(User).filter(User.role == "patient").count()
    total_doctors = db.query(User).filter(User.role == "doctor").count()
    total_predictions = db.query(AIPrediction).count()
    total_ai_feedbacks = db.query(DoctorFeedback).count()

    return templates.TemplateResponse(
        "admin_analytics.html",
        {
            "request": request,
            "total_patients": total_patients,
            "total_doctors": total_doctors,
            "total_predictions": total_predictions,
            "total_ai_feedbacks": total_ai_feedbacks
        }
    )


@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard(request: Request, user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403)
    return templates.TemplateResponse("admin_dashboard.html", {"request": request})

@app.get("/transaction/dashboard", response_class=HTMLResponse)
def transaction_dashboard(request: Request, user=Depends(get_current_user)):
    if user["role"] != "transaction":
        raise HTTPException(status_code=403)
    return templates.TemplateResponse("transaction_dashboard.html", {"request": request})
@app.get("/transaction/transactions", response_class=HTMLResponse)
def view_transactions(
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "transaction":
        raise HTTPException(status_code=403)

    payments = (
        db.query(Payment, User)
        .join(Prescription, Payment.prescription_id == Prescription.id)
        .join(AIPrediction, Prescription.prediction_id == AIPrediction.id)
        .join(User, AIPrediction.patient_id == User.id)
        .all()
    )

    return templates.TemplateResponse(
        "transactions.html",
        {
            "request": request,
            "payments": payments
        }
    )


@app.get("/transaction/invoices", response_class=HTMLResponse)
def view_invoices(
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] not in ["transaction", "admin"]:
        raise HTTPException(status_code=403)

    invoices = (
        db.query(Payment, Prescription, AIPrediction, User)
        .join(Prescription, Payment.prescription_id == Prescription.id)
        .join(AIPrediction, Prescription.prediction_id == AIPrediction.id)
        .join(User, AIPrediction.patient_id == User.id)
        .filter(Payment.status == "SUCCESS")
        .all()
    )

    return templates.TemplateResponse(
        "transaction_invoices.html",
        {
            "request": request,
            "invoices": invoices
        }
    )

@app.get("/transaction/pending-payments", response_class=HTMLResponse)
def view_pending_payments(
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] not in ["transaction", "admin"]:
        raise HTTPException(status_code=403)

    payments = (
        db.query(Payment, Prescription, User)
        .join(Prescription, Payment.prescription_id == Prescription.id)
        .join(AIPrediction, Prescription.prediction_id == AIPrediction.id)
        .join(User, AIPrediction.patient_id == User.id)
        .filter(Payment.status == "PENDING")
        .all()
    )

    return templates.TemplateResponse(
        "transaction_pending_payments.html",
        {
            "request": request,
            "payments": payments
        }
    )
@app.post("/transaction/verify-payment")
def verify_payment(
    payment_id: int = Form(...),
    action: str = Form(...),
    utr: str = Form(None),
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] not in ["transaction", "admin"]:
        raise HTTPException(status_code=403)

    payment = db.query(Payment).filter(Payment.id == payment_id).first()
    if not payment:
        raise HTTPException(status_code=404)

    # 🔹 VERIFY UTR
    if action == "verify":
        if not payment.utr:
            raise HTTPException(status_code=400, detail="Patient has not submitted UTR")

        if payment.utr.strip() == utr.strip():
            payment.utr_verified = True
            db.commit()
        else:
            raise HTTPException(status_code=400, detail="UTR mismatch")

    # 🔹 APPROVE / REJECT
    elif action == "approve":
        if not payment.utr_verified:
            raise HTTPException(status_code=400, detail="Verify UTR first")
        payment.status = "SUCCESS"

    elif action == "reject":
        payment.status = "FAILED"

    db.commit()
    return RedirectResponse("/transaction/transactions", status_code=302)
@app.api_route("/tm/verify-utr", methods=["GET", "POST"], response_class=HTMLResponse)
def tm_verify_utr(
    request: Request,
    payment_id: int = Form(None),
    typed_utr: str = Form(None),
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    if user["role"] not in ["transaction", "admin"]:
        return templates.TemplateResponse("access_denied.html", {"request": request})

    payment = None
    ocr_utr = None
    message = None

    # Load payment if selected
    if payment_id:
        payment = db.query(Payment).filter(Payment.id == payment_id).first()

        # OCR only if screenshot exists
        if payment and payment.screenshot:
            import pytesseract, os
            from PIL import Image
            
            path = os.path.join("static", "uploads", payment.screenshot)
            if os.path.exists(path):
                img = Image.open(path)
                raw_text = pytesseract.image_to_string(img)

                import re
                found = re.findall(r"UTR[A-Z0-9]+", raw_text)
                if found:
                    ocr_utr = found[-1].strip()

    # If submitted form
    if request.method == "POST" and payment_id and typed_utr and payment:
        typed = typed_utr.strip().upper()
        ocr = (ocr_utr or "").upper()
        db_utr = (payment.utr or "").upper()

        # Compare all three
        if typed == ocr == db_utr:
            payment.status = "SUCCESS"
            message = "Payment Verified Successfully "
        else:
            payment.status = "FAILED"
            message = "❌ UTR mismatch"

        db.commit()

    payments = db.query(Payment).all()

    return templates.TemplateResponse(
        "tm_verify.html",
        {
            "request": request,
            "payments": payments,
            "payment": payment,
            "ocr_utr": ocr_utr,
            "message": message
        }
    )

# ---------------- PATIENT PROFILE ---------------- #

@app.get("/patient/profile", response_class=HTMLResponse)
def patient_profile(
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "patient":
        raise HTTPException(status_code=403)

    patient = db.query(User).filter(User.email == user["sub"]).first()

    return templates.TemplateResponse(
        "patient_profile.html",
        {
            "request": request,
            "patient": patient
        }
    )
@app.get("/patient/ai-check", response_class=HTMLResponse)
def ai_check_page(
    request: Request,
    user=Depends(get_current_user)
):
    if user["role"] != "patient":
        raise HTTPException(status_code=403)

    return templates.TemplateResponse(
        "ai_prediction.html",
        {"request": request}
    )
@app.post("/patient/ai-result", response_class=HTMLResponse)
def ai_result(
    request: Request,
    symptoms: str = Form(...),
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    if user["role"] != "patient":
        raise HTTPException(status_code=403)

    result = analyze_symptoms_text(symptoms)

    # delete old prediction (keep latest only)
    db.query(AIPrediction).filter(
        AIPrediction.patient_id == user["id"]
    ).delete()

    # save new prediction
    confidence_value = float(result.get("confidence", "0%").replace("%", ""))
    prediction = AIPrediction(
    patient_id=user["id"],
    symptoms=symptoms,   # ✅ FIXED
    condition=result["condition"],
    risk=result["risk"],
    confidence=confidence_value,
    recommendation=result["recommendation"]
    )

    db.add(prediction)
    db.commit()

    return templates.TemplateResponse(
        "ai_result.html",
        {
            "request": request,
            "result": result
        }
    )
@app.get("/patient/ai-results", response_class=HTMLResponse)
def view_ai_results(
    request: Request,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    if user["role"] != "patient":
        raise HTTPException(status_code=403)

    # latest AI prediction
    prediction = db.query(AIPrediction).filter(
        AIPrediction.patient_id == user["id"]
    ).order_by(AIPrediction.id.desc()).first()

    # latest doctor remark for patient
    remark = db.query(DoctorRemark).filter(
        DoctorRemark.patient_id == user["id"]
    ).order_by(DoctorRemark.id.desc()).first()

    return templates.TemplateResponse(
    "ai_result.html",
    {
        "request": request,
        "symptoms": prediction.symptoms if prediction else "N/A",
        "result": {
            "condition": prediction.condition if prediction else "N/A",
            "risk": prediction.risk if prediction else "N/A",
            "confidence": f"{prediction.confidence}%" if prediction else "N/A",
            "recommendation": prediction.recommendation if prediction else "N/A"
        },
        "doctor_feedback": remark.remark if remark else None
    }
)


@app.get("/patient/remarks", response_class=HTMLResponse)
def patient_remarks(
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "patient":
        raise HTTPException(status_code=403)

    remarks = db.query(DoctorRemark).filter(
        DoctorRemark.patient_id == user["id"]
    ).all()

    return templates.TemplateResponse(
        "patient_remarks.html",
        {
            "request": request,
            "remarks": remarks
        }
    )

# ---------------- ADMIN CREATE USER ---------------- #

@app.get("/admin/create-user", response_class=HTMLResponse)
def create_user_page(
    request: Request,
    success: str = None,
    error: str = None,
    user=Depends(get_current_user)
):
    if user["role"] != "admin":
        raise HTTPException(status_code=403)

    return templates.TemplateResponse(
        "admin_create_user.html",
        {
            "request": request,
            "success": success,
            "error": error
        }
    )

@app.post("/admin/create-user")
def create_user(
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    if user["role"] != "admin":
        raise HTTPException(status_code=403)

    if role not in ["doctor", "transaction"]:
        raise HTTPException(status_code=400)

    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        return RedirectResponse(
            url="/admin/create-user?error=exists",
            status_code=302
        )

    new_user = User(
        name=name,
        age=age,
        gender=gender,
        email=email,
        password=hash_password(password),
        role=role
    )

    db.add(new_user)
    db.commit()

    return RedirectResponse(
        url="/admin/create-user?success=1",
        status_code=302
    )

# ---------------- ADMIN VIEW USERS ---------------- #

@app.get("/admin/users", response_class=HTMLResponse)
def view_users(
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "admin":
        raise HTTPException(status_code=403)

    users = db.query(User).all()

    return templates.TemplateResponse(
        "admin_users.html",
        {
            "request": request,
            "users": users
        }
    )

# ---------------- ADMIN EDIT USER ---------------- #

@app.get("/admin/edit-user/{user_id}", response_class=HTMLResponse)
def edit_user_page(
    user_id: int,
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "admin":
        raise HTTPException(status_code=403)

    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=404)

    if target_user.role in ["admin", "patient"]:
        raise HTTPException(status_code=403)

    return templates.TemplateResponse(
        "admin_edit_user.html",
        {
            "request": request,
            "target": target_user
        }
    )

@app.post("/admin/edit-user/{user_id}")
def edit_user(
    user_id: int,
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    email: str = Form(...),
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "admin":
        raise HTTPException(status_code=403)

    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=404)

    if target_user.role in ["admin", "patient"]:
        raise HTTPException(status_code=403)

    target_user.name = name
    target_user.age = age
    target_user.gender = gender
    target_user.email = email

    db.commit()

    return RedirectResponse(
        url="/admin/users",
        status_code=302
    )

# ---------------- ADMIN DELETE USER ---------------- #

@app.post("/admin/delete-user")
def delete_user(
    user_id: int = Form(...),
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "admin":
        raise HTTPException(status_code=403)

    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=404)

    if target_user.role in ["admin", "patient"]:
        raise HTTPException(status_code=403)

    db.delete(target_user)
    db.commit()

    return RedirectResponse(
        url="/admin/users",
        status_code=302
    )
@app.get("/admin/medical-feedback", response_class=HTMLResponse)
def admin_medical_feedback(
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "admin":
        raise HTTPException(status_code=403)

    feedbacks = (
        db.query(DoctorFeedback, User)
        .join(User, DoctorFeedback.doctor_id == User.id)
        .all()
    )

    return templates.TemplateResponse(
        "admin_medical_feedback.html",
        {
            "request": request,
            "feedbacks": feedbacks
        }
    )
@app.get("/admin/medical-feedback", response_class=HTMLResponse)
def admin_medical_feedback(
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user["role"] != "admin":
        raise HTTPException(status_code=403)

    feedbacks = db.query(AIFeedback).all()

    return templates.TemplateResponse(
        "admin_medical_feedback.html",
        {
            "request": request,
            "feedbacks": feedbacks
        }
    )

# ---------------- LOGOUT ---------------- #
from fastapi.responses import PlainTextResponse

@app.exception_handler(401)
def unauthorized_handler(request: Request, exc):
    return templates.TemplateResponse(
        "401.html",
        {"request": request},
        status_code=401
    )

@app.exception_handler(403)
def forbidden_handler(request: Request, exc):
    return templates.TemplateResponse(
        "403.html",
        {"request": request},
        status_code=403
    )

@app.get("/logout")
def logout():
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("access_token", path="/")
    return response
