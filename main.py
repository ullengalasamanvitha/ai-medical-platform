from fastapi import FastAPI, Request, Form, Depends, Cookie, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta

from database import engine, SessionLocal
from models import Base, User

# --------------------------------------------------
# APP SETUP
# --------------------------------------------------

app = FastAPI(title="AI Medical Consultancy & Virtual Doctor Platform")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

Base.metadata.create_all(bind=engine)

# --------------------------------------------------
# PASSWORD HASHING
# --------------------------------------------------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

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
        data={"sub": user.email, "role": user.role}
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
def patient_dashboard(request: Request, user=Depends(get_current_user)):
    if user["role"] != "patient":
        raise HTTPException(status_code=403)
    return templates.TemplateResponse("patient_dashboard.html", {"request": request})

@app.get("/doctor/dashboard", response_class=HTMLResponse)
def doctor_dashboard(request: Request, user=Depends(get_current_user)):
    if user["role"] != "doctor":
        raise HTTPException(status_code=403)
    return templates.TemplateResponse("doctor_dashboard.html", {"request": request})

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
    user=Depends(get_current_user)
):
    if user["role"] != "transaction":
        raise HTTPException(status_code=403)

    # Dummy placeholder data
    transactions = [
        {"id": 1, "patient": "John Doe", "amount": "₹500", "status": "Completed"},
        {"id": 2, "patient": "Anita Sharma", "amount": "₹800", "status": "Pending"},
        {"id": 3, "patient": "Rahul Verma", "amount": "₹1200", "status": "Completed"}
    ]

    return templates.TemplateResponse(
        "transactions.html",
        {
            "request": request,
            "transactions": transactions
        }
    )
@app.get("/transaction/invoices", response_class=HTMLResponse)
def view_invoices(
    request: Request,
    user=Depends(get_current_user)
):
    if user["role"] != "transaction":
        raise HTTPException(status_code=403)

    invoices = [
        {"invoice_id": "INV001", "amount": "₹500", "status": "Paid"},
        {"invoice_id": "INV002", "amount": "₹800", "status": "Unpaid"},
        {"invoice_id": "INV003", "amount": "₹1200", "status": "Paid"}
    ]

    return templates.TemplateResponse(
        "invoices.html",
        {
            "request": request,
            "invoices": invoices
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
