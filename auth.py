import os
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================
# Read variables from environment, using secure defaults for local sandbox runtimes
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "b38fae6347960fc5a87ba81e18ca47402bd2ad15ece5d77bd2b8637731773428")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# Passlib CryptContext configuration for secure Argon2/bcrypt hashing wrappers
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Authorization Header scheme extraction dependency (OAuth2 standard)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


# =====================================================================
# PYDANTIC SCHEMAS / TRANSFER INTERFACES
# =====================================================================
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None

class UserBase(BaseModel):
    email: EmailStr
    username: str

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: str
    role: str
    is_active: bool = True

    class Config:
        from_attributes = True


# =====================================================================
# HASHING UTILITIES
# =====================================================================
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against its hashed representation."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generates a secure salt and hashes the plain password with bcrypt."""
    return pwd_context.hash(password)


# =====================================================================
# JWT MANIPULATION SERVICES
# =====================================================================
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> Token:
    """
    Encrypts custom dictionary payload into a highly secured JWT.
    Enforces time-scoped validation attributes (`exp`).
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Standard claim updates
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return Token(
        access_token=encoded_jwt,
        expires_in=int((expire - datetime.utcnow()).total_seconds())
    )


def decode_access_token(token: str) -> TokenData:
    """
    Decodes the validation payload checking signatures and structural claims.
    Raises HTTPException status-code 401 if validation fails.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        role: str = payload.get("role")
        if email is None:
            raise credentials_exception
        return TokenData(email=email, role=role)
    except JWTError:
        raise credentials_exception


# =====================================================================
# DEPENDENCY INJECTION VERIFIER
# =====================================================================
async def get_current_user_claims(token: str = Depends(oauth2_scheme)) -> TokenData:
    """
    Reusable FastAPI dependency that decodes user authentication cookies/headers
    yielding standard metadata values, automatically rejecting unverified payloads.
    """
    return decode_access_token(token)
