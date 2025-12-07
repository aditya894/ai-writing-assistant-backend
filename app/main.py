import os
import json
from datetime import datetime, timedelta

import razorpay
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .llm_client import improve_text


# --------------------------- FASTAPI APP ---------------------------
app = FastAPI(
    title="AI Writing Assistant API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------- RAZORPAY CONFIG ---------------------------
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
FREE_LICENSE_EMAILS_ENV = os.getenv("FREE_LICENSE_EMAILS", "")

if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
    razor_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
else:
    razor_client = None
    print("WARNING: Razorpay keys not set. Payments will not work.")

FREE_LICENSE_EMAILS = {
    e.strip().lower()
    for e in FREE_LICENSE_EMAILS_ENV.split(",")
    if e.strip()
}

LICENSE_FILE = "licenses.json"


# --------------------------- LICENSE STORAGE ---------------------------
def _load_licenses() -> dict:
    if not os.path.exists(LICENSE_FILE):
        return {}
    try:
        with open(LICENSE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_licenses(data: dict) -> None:
    with open(LICENSE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _set_license(email: str, months: int = 1) -> str:
    email = email.strip().lower()

    # Always free for owner
    if email in FREE_LICENSE_EMAILS:
        expiry_str = "2099-12-31"
        licenses = _load_licenses()
        licenses[email] = {"expiry": expiry_str}
        _save_licenses(licenses)
        return expiry_str

    now = datetime.utcnow().date()
    licenses = _load_licenses()
    base_date = now

    if email in licenses and "expiry" in licenses[email]:
        try:
            old_date = datetime.strptime(licenses[email]["expiry"], "%Y-%m-%d").date()
            if old_date > now:
                base_date = old_date
        except ValueError:
            pass

    new_expiry = base_date + timedelta(days=30 * months)
    expiry_str = new_expiry.strftime("%Y-%m-%d")
    licenses[email] = {"expiry": expiry_str}
    _save_licenses(licenses)
    return expiry_str


def _check_license(email: str) -> tuple[bool, str | None]:
    email = email.strip().lower()

    if email in FREE_LICENSE_EMAILS:
        return True, "2099-12-31"

    licenses = _load_licenses()
    if email not in licenses:
        return False, None

    expiry_str = licenses[email].get("expiry")
    if not expiry_str:
        return False, None

    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    except ValueError:
        return False, expiry_str

    if datetime.utcnow().date() > expiry_date:
        return False, expiry_str

    return True, expiry_str


# --------------------------- REQUEST MODELS ---------------------------
class ImproveRequest(BaseModel):
    text: str
    tone: str | None = None
    language: str | None = "en"


class ImproveResponse(BaseModel):
    improved_text: str


class CreateOrderRequest(BaseModel):
    email: str


class CreateOrderResponse(BaseModel):
    order_id: str
    amount: int
    currency: str
    key_id: str


class ActivateLicenseRequest(BaseModel):
    email: str
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


class LicenseStatusResponse(BaseModel):
    active: bool
    expiry: str | None = None


# --------------------------- ROUTES ---------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/improve_text", response_model=ImproveResponse)
def improve_text_endpoint(payload: ImproveRequest):
    improved = improve_text(
        text=payload.text,
        tone=payload.tone or "neutral professional",
        language=payload.language or "en",
    )
    return ImproveResponse(improved_text=improved)


@app.get("/api/license_status", response_model=LicenseStatusResponse)
def license_status(email: str):
    active, expiry = _check_license(email)
    return LicenseStatusResponse(active=active, expiry=expiry)


@app.post("/api/create_order", response_model=CreateOrderResponse)
def create_order(payload: CreateOrderRequest):
    if razor_client is None:
        raise HTTPException(status_code=500, detail="Payment not configured")

    email = payload.email.strip().lower()

    if email in FREE_LICENSE_EMAILS:
        raise HTTPException(status_code=400, detail="This email does not require payment.")

    amount_paise = 99 * 100
    order = razor_client.order.create(
        {
            "amount": amount_paise,
            "currency": "INR",
            "payment_capture": 1,
            "notes": {"email": email},
        }
    )

    return CreateOrderResponse(
        order_id=order["id"],
        amount=amount_paise,
        currency="INR",
        key_id=RAZORPAY_KEY_ID,
    )


@app.post("/api/activate_license")
def activate_license(payload: ActivateLicenseRequest):
    if razor_client is None:
        raise HTTPException(status_code=500, detail="Payment not configured")

    email = payload.email.strip().lower()

    if email in FREE_LICENSE_EMAILS:
        expiry = _set_license(email, months=12)
        return {"success": True, "expiry": expiry}

    params = {
        "razorpay_order_id": payload.razorpay_order_id,
        "razorpay_payment_id": payload.razorpay_payment_id,
        "razorpay_signature": payload.razorpay_signature,
    }

    try:
        razor_client.utility.verify_payment_signature(params)
    except razorpay.errors.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid payment signature")

    expiry = _set_license(email, months=1)
    return {"success": True, "expiry": expiry}
