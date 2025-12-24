from database import SessionLocal
from models import User
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    return pwd_context.hash(password)

db = SessionLocal()

users = [
    User(
        name="Doctor User",
        age=35,
        gender="Male",
        email="doctor@platform.com",
        password=hash_password("doctor123"),
        role="doctor"
    ),
    User(
        name="Admin User",
        age=40,
        gender="Female",
        email="admin@platform.com",
        password=hash_password("admin123"),
        role="admin"
    ),
    User(
        name="Finance User",
        age=38,
        gender="Male",
        email="finance@platform.com",
        password=hash_password("finance123"),
        role="transaction"
    )
]

for user in users:
    exists = db.query(User).filter(User.email == user.email).first()
    if not exists:
        db.add(user)

db.commit()
db.close()

print("Default users created successfully.")
