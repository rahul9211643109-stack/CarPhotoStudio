import base64
import io
import json
import os
import zipfile
from datetime import date
from pathlib import Path

import pymupdf as fitz
import streamlit as st
from PIL import Image


@st.cache_resource(show_spinner=False)
def get_lama_model():
    """Load LaMa once per Streamlit process; keep model cached between reruns."""
    from simple_lama_inpainting import SimpleLama
    return SimpleLama()

APP_DIR = Path(__file__).resolve().parent
BRAND_LOGO_PATH = APP_DIR / "CarPhotoStudio_Logo_Transparent.png"
FAVICON_PATH = APP_DIR / "CarPhotoStudio_Favicon.png"

st.set_page_config(
    page_title="CarPhotoStudio",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.is_file() else "ðŸš—",
    layout="wide",
)

# Premium showroom UI theme (CSS only; app workflows remain unchanged)
st.markdown(
    """
    <style>
    :root {
        --cps-bg: #0a111b;
        --cps-panel: #111c29;
        --cps-panel-2: #162435;
        --cps-border: rgba(153, 180, 204, .18);
        --cps-text: #edf5fb;
        --cps-muted: #9bb0c2;
        --cps-teal: #2dd4bf;
        --cps-teal-dark: #0f766e;
    }
    .stApp, [data-testid="stAppViewContainer"] {
        background: radial-gradient(ellipse at 18% 0%, rgba(22, 78, 99, .23), transparent 37%),
                    radial-gradient(ellipse at 95% 15%, rgba(13, 148, 136, .12), transparent 28%),
                    var(--cps-bg);
        color: var(--cps-text);
    }
    [data-testid="stHeader"] { background: rgba(10, 17, 27, .76); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1825 0%, #0a121d 100%);
        border-right: 1px solid var(--cps-border);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.15rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--cps-muted); }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: .25rem; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 10px; padding: .42rem .55rem; transition: background .18s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(45, 212, 191, .09); }
    .block-container { max-width: 1500px; padding-top: 1.35rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--cps-text) !important; letter-spacing: -.025em; }
    h1 { font-weight: 780 !important; }
    h2 { font-weight: 720 !important; margin-top: .4rem !important; }
    h3 { font-weight: 680 !important; }
    p, li, label, .stCaption, [data-testid="stMarkdownContainer"] { color: inherit; }
    [data-testid="stCaptionContainer"], .stCaption { color: var(--cps-muted) !important; }
    .cps-brandbar {
        display: flex; align-items: center; justify-content: space-between; gap: 1rem;
        padding: 1rem 1.25rem; margin: 0 0 1.45rem 0;
        border: 1px solid var(--cps-border); border-radius: 20px;
        background: linear-gradient(110deg, rgba(20, 36, 52, .95), rgba(12, 27, 39, .88));
        box-shadow: 0 16px 42px rgba(0,0,0,.18);
    }
    .cps-brand-left { display:flex; align-items:center; gap:.9rem; min-width:0; }
    .cps-logo-mark {
        width: 58px; height: 58px; display:flex; align-items:center; justify-content:center;
        flex: 0 0 58px; border-radius: 15px; font-size: 1.7rem; overflow:hidden;
        background: rgba(255,255,255,.96); border: 1px solid rgba(45,212,191,.32);
    }
    .cps-logo-mark img { width: 100%; height: 100%; object-fit: contain; padding: 3px; }
    .cps-overline { font-size: .66rem; letter-spacing: .19em; color: var(--cps-teal); font-weight: 800; }
    .cps-brand-title { font-size: 1.55rem; font-weight: 850; color: #f4f9fc; line-height: 1.18; letter-spacing: -.045em; }
    .cps-brand-title span { color: var(--cps-teal); }
    .cps-brand-subtitle { font-size: .79rem; color: var(--cps-muted); margin-top: .25rem; }
    .cps-live-pill { white-space: nowrap; padding: .42rem .7rem; border-radius: 999px;
        color: #a7f3d0; background: rgba(16,185,129,.1); border: 1px solid rgba(52,211,153,.24);
        font-size: .72rem; font-weight: 700; }
    [data-testid="stMetric"] {
        padding: 1rem 1.1rem; background: linear-gradient(145deg, rgba(22,36,53,.92), rgba(14,26,39,.92));
        border: 1px solid var(--cps-border); border-radius: 17px;
    }
    [data-testid="stMetricLabel"] { color: var(--cps-muted) !important; font-size: .82rem; }
    [data-testid="stMetricValue"] { color: #f2fbff !important; font-weight: 800; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--cps-border) !important; border-radius: 17px !important; }
    [data-testid="stAlert"] { border-radius: 13px; border: 1px solid var(--cps-border); }
    [data-testid="stFileUploader"] {
        background: rgba(16, 29, 43, .72); border: 1px dashed rgba(45,212,191,.38);
        border-radius: 17px; padding: .65rem;
    }
    [data-testid="stFileUploader"] section { background: transparent; }
    div.stButton > button, div.stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
        border-radius: 11px; border: 1px solid rgba(152,178,198,.25);
        background: #162637; color: #edf8ff; font-weight: 680; min-height: 2.65rem;
        transition: transform .15s ease, border-color .15s ease, background .15s ease;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        border-color: rgba(45,212,191,.7); background: #1a3445; transform: translateY(-1px); color: white;
    }
    div.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
        background: linear-gradient(120deg, #0f9f91, #0f766e); border-color: rgba(45,212,191,.6); color: #fff;
        box-shadow: 0 6px 18px rgba(13,148,136,.15);
    }
    div.stButton > button[kind="primary"]:hover { background: linear-gradient(120deg, #14b8a6, #0f8b82); }
    [data-baseweb="input"] > div, [data-baseweb="textarea"] > div, [data-baseweb="select"] > div {
        background: #0e1b29; border-color: rgba(151,176,197,.25); border-radius: 10px;
    }
    [data-baseweb="input"] input, [data-baseweb="textarea"] textarea { color: #eff8fd; }
    [data-baseweb="select"] div { color: #edf5fb; }
    [data-testid="stExpander"] {
        background: rgba(16,29,43,.64); border: 1px solid var(--cps-border); border-radius: 13px;
    }
    [data-testid="stImage"] img { border-radius: 13px; border: 1px solid rgba(161,185,204,.13); }
    [data-testid="stProgressBar"] > div > div { background: var(--cps-teal); }
    hr { border-color: var(--cps-border); }
    code { color: #a7f3d0; background: rgba(45,212,191,.08); border-radius: 5px; }
    @media (max-width: 700px) {
        .block-container { padding-top: .8rem; }
        .cps-brandbar { padding: .85rem; border-radius: 15px; }
        .cps-logo-mark { width: 42px; height: 42px; flex-basis:42px; border-radius:12px; }
        .cps-brand-title { font-size: 1.2rem; }
        .cps-brand-subtitle { font-size: .7rem; }
        .cps-live-pill { display:none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Google OAuth: Render Secret Files path first; local file fallback for VS Code.
CLIENT_FILE = "/etc/secrets/client_secret.json"
LOCAL_CLIENT_FILE = str(APP_DIR / "client_secret.json")
TOKEN_FILE = str(APP_DIR / "token.json")
APP_URL = os.environ.get("APP_URL", "https://carphotostudio.onrender.com/")
AI_USAGE_FILE = Path(__file__).resolve().parent / "ai_usage.json"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# Use a downsized copy in the header to keep reruns responsive.
if BRAND_LOGO_PATH.is_file():
    try:
        with Image.open(BRAND_LOGO_PATH) as _brand_img:
            _brand_img = _brand_img.convert("RGBA")
            _brand_img.thumbnail((256, 256), Image.Resampling.LANCZOS)
            _brand_buf = io.BytesIO()
            _brand_img.save(_brand_buf, format="PNG", optimize=True)
        _brand_logo_b64 = base64.b64encode(_brand_buf.getvalue()).decode("ascii")
        _brand_mark_html = f'<img src="data:image/png;base64,{_brand_logo_b64}" alt="CarPhotoStudio logo" />'
    except Exception:
        _brand_mark_html = "ðŸš˜"
else:
    _brand_mark_html = "ðŸš˜"

st.markdown(
    f"""
    <div class=\"cps-brandbar\">
      <div class=\"cps-brand-left\">
        <div class=\"cps-logo-mark\">{_brand_mark_html}</div>
        <div>
          <div class=\"cps-overline\">SHOWROOM WORKSPACE</div>
          <div class=\"cps-brand-title\">CarPhoto<span>Studio</span></div>
          <div class=\"cps-brand-subtitle\">PDF tools Â· AI photo studio Â· Listings Â· Drive</div>
        </div>
      </div>
      <div class=\"cps-live-pill\">â— WORKSPACE READY</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "photos" not in st.session_state:
    st.session_state.photos = []
if "original_pdf" not in st.session_state:
    st.session_state.original_pdf = None
if "combined_pdf" not in st.session_state:
    st.session_state.combined_pdf = None
if "drive_folder" not in st.session_state:
    st.session_state.drive_folder = None
if "car_listings" not in st.session_state:
    st.session_state.car_listings = []
if "listing_photo_index" not in st.session_state:
    st.session_state.listing_photo_index = 0


def extract_pdf_text(pdf_bytes):
    """Read selectable text from the uploaded PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for page_obj in doc:
        parts.append(page_obj.get_text("text"))
    doc.close()
    return "\n".join(parts).strip()


def guess_car_details(text):
    """Best-effort suggestions only; user should verify every field."""
    import re
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    details = {
        "car_name": "",
        "registration": "",
        "year": "",
        "fuel": "",
        "owner": "",
        "location": "",
        "price": "",
        "raw_text": text[:12000],
    }
    if lines:
        # A likely vehicle title is often one of the first short lines.
        details["car_name"] = next((line for line in lines[:8] if len(line) < 100), "")
    patterns = {
        "registration": r"\b[A-Z]{2}[- ]?\d{1,2}[- ]?[A-Z]{1,3}[- ]?\d{3,4}\b",
        "year": r"\b(?:19|20)\d{2}\b",
        "price": r"(?:Rs\.?|INR|â‚¹)\s*[\d,]+(?:\.\d+)?|[\d,]+\s*/-",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            details[key] = match.group(0).strip()
    fuel_match = re.search(r"\b(PETROL|DIESEL|CNG|ELECTRIC|HYBRID)\b", text, re.IGNORECASE)
    if fuel_match:
        details["fuel"] = fuel_match.group(1).upper()
    owner_match = re.search(r"\b(\d+\s*Owner)\b", text, re.IGNORECASE)
    if owner_match:
        details["owner"] = owner_match.group(1)
    location_match = re.search(r"(?:AT[- ]|Location[: -]+)([^\n]+)", text, re.IGNORECASE)
    if location_match:
        details["location"] = location_match.group(1).strip()[:100]
    return details


def prepare_image_for_edit(raw_bytes):
    """Normalize a source image to JPEG and limit dimensions/file size before paid editing."""
    if not raw_bytes or len(raw_bytes) > 20 * 1024 * 1024:
        raise ValueError("Photo empty hai ya 20 MB se badi hai. Chhoti JPG/PNG choose karo.")
    with Image.open(io.BytesIO(raw_bytes)) as opened:
        opened = opened.convert("RGB")
        opened.thumbnail((1536, 1536))
        buffer = io.BytesIO()
        opened.save(buffer, format="JPEG", quality=88, optimize=True)
        return buffer.getvalue()


def add_ai_disclaimer(raw_bytes):
    """Mark every output so it cannot be mistaken for an untouched evidence photo."""
    from PIL import ImageDraw, ImageFont
    with Image.open(io.BytesIO(raw_bytes)) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    band_height = max(36, image.height // 14)
    y0 = image.height - band_height
    draw.rectangle((0, y0, image.width, image.height), fill=(20, 20, 20))
    label = "AI-EDITED / ILLUSTRATIVE - NOT ORIGINAL PHOTO"
    try:
        font = ImageFont.truetype("arial.ttf", max(14, image.width // 55))
    except OSError:
        font = ImageFont.load_default()
    draw.text((max(8, image.width // 60), y0 + band_height // 4), label, fill=(255, 255, 255), font=font)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=92, optimize=True)
    return out.getvalue()



def add_brand_logo_copy(base_bytes, logo_bytes, position="Bottom-right", width_pct=12, margin_pct=2):
    """Overlay user's supplied showroom logo onto a copy; original bytes stay untouched."""
    from PIL import Image
    with Image.open(io.BytesIO(base_bytes)) as base_img:
        base = base_img.convert("RGBA")
    with Image.open(io.BytesIO(logo_bytes)) as logo_img:
        logo = logo_img.convert("RGBA")
    target_w = max(1, int(base.width * float(width_pct) / 100.0))
    target_h = max(1, int(base.height * 0.20))
    logo.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    margin_x = int(base.width * float(margin_pct) / 100.0)
    margin_y = int(base.height * float(margin_pct) / 100.0)
    if position == "Top-left":
        x, y = margin_x, margin_y
    elif position == "Top-right":
        x, y = base.width - logo.width - margin_x, margin_y
    elif position == "Bottom-left":
        x, y = margin_x, base.height - logo.height - margin_y
    else:
        x, y = base.width - logo.width - margin_x, base.height - logo.height - margin_y
    x = max(0, min(base.width - logo.width, x))
    y = max(0, min(base.height - logo.height, y))
    base.alpha_composite(logo, (x, y))
    output = io.BytesIO()
    base.convert("RGB").save(output, format="JPEG", quality=92, optimize=True)
    return output.getvalue()


def add_ai_disclaimer_header(raw_bytes):
    """Mark AI-generated edits so they cannot be mistaken for untouched original photos."""
    from PIL import ImageDraw, ImageFont
    with Image.open(io.BytesIO(raw_bytes)) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    band_height = max(28, min(48, image.height // 20))
    draw.rectangle((0, 0, image.width, band_height), fill=(24, 24, 24))
    label = "AI-EDITED COPY - VERIFY VEHICLE DETAILS"
    try:
        font = ImageFont.truetype("arial.ttf", max(12, min(22, image.width // 70)))
    except OSError:
        font = ImageFont.load_default()
    draw.text((max(6, image.width // 100), max(2, band_height // 4)), label, fill=(255, 255, 255), font=font)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92, optimize=True)
    return output.getvalue()


def extract_pdf_embedded_car_photos(pdf_bytes, pdf_name):
    """Extract sufficiently large embedded raster images from PDF pages as JPEG copies."""
    extracted = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    seen_xrefs = set()
    try:
        for page_num, page_obj in enumerate(doc, start=1):
            for image_info in page_obj.get_images(full=True):
                xref = image_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    extracted_info = doc.extract_image(xref)
                    with Image.open(io.BytesIO(extracted_info["image"])) as raw_img:
                        im = raw_img.convert("RGB")
                    if im.width < 400 or im.height < 250:
                        continue
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=93, optimize=True)
                    extracted.append({
                        "label": f"{Path(pdf_name).stem}_page{page_num}_image{len(extracted)+1}.jpg",
                        "bytes": buf.getvalue(),
                    })
                except Exception:
                    continue
        if not extracted:
            # Fallback for flattened/vector PDFs where page.get_images() returns no large photos.
            # These are whole-page renders, not isolated car photos, so the UI labels them accordingly.
            for page_num in range(min(len(doc), 30)):
                try:
                    page_obj = doc[page_num]
                    pix = page_obj.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
                    if pix.width < 400 or pix.height < 250:
                        continue
                    im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=90, optimize=True)
                    extracted.append({
                        "label": f"{Path(pdf_name).stem}_page{page_num + 1}_rendered.jpg",
                        "bytes": buf.getvalue(),
                        "rendered_page": True,
                    })
                except Exception:
                    continue
    finally:
        doc.close()
    return extracted


def get_client_secret_file():
    """Return the Google OAuth client JSON path for Render or local development."""
    if os.path.isfile(CLIENT_FILE):
        return CLIENT_FILE
    if os.path.isfile(LOCAL_CLIENT_FILE):
        return LOCAL_CLIENT_FILE
    return None


def make_google_oauth_flow(state=None):
    """Create a web-server OAuth flow for Streamlit/Render."""
    try:
        from google_auth_oauthlib.flow import Flow
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive packages missing hain. requirements.txt mein "
            "google-auth-oauthlib aur google-api-python-client install karo."
        ) from exc
    secret_path = get_client_secret_file()
    if not secret_path:
        raise FileNotFoundError(
            "Google OAuth client file nahi mili. Render > Environment > Secret Files "
            "mein /etc/secrets/client_secret.json add karo."
        )
    flow = Flow.from_client_secrets_file(secret_path, scopes=SCOPES, state=state)
    flow.redirect_uri = APP_URL
    return flow


def process_google_oauth_callback():
    """Finish the Google redirect when the browser returns to the app."""
    try:
        params = st.query_params
        code = params.get("code")
        error = params.get("error")
        if error:
            st.error(f"Google login cancel/failed: {error}")
            params.clear()
            return
        if not code:
            return
        saved_state = st.session_state.get("google_oauth_state")
        returned_state = params.get("state")
        if not saved_state or not returned_state or saved_state != returned_state:
            st.error(
                "Google login state match nahi hua. Google Drive page par jaakar "
                "Connect Google Account dobara dabao."
            )
            params.clear()
            st.session_state.pop("google_oauth_state", None)
            return
        flow = make_google_oauth_flow(state=saved_state)
        flow.fetch_token(code=code)
        st.session_state.google_credentials = flow.credentials
        st.session_state.google_oauth_state = None
        params.clear()
        try:
            service = get_drive_service()
            about = service.about().get(fields="user(displayName,emailAddress)").execute()
            st.session_state.google_email = about.get("user", {}).get(
                "emailAddress", "Google account connected"
            )
        except Exception:
            st.session_state.google_email = "Google account connected"
        st.success("Google account connect ho gaya!")
    except Exception as exc:
        st.error(f"Google OAuth callback failed: {exc}")


def get_google_credentials():
    """Get current session credentials; OAuth is started by the UI."""
    try:
        from google.auth.transport.requests import Request
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive packages missing hain. VS Code Terminal mein: "
            "python -m pip install -r requirements.txt"
        ) from exc
    credentials = st.session_state.get("google_credentials")
    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        st.session_state.google_credentials = credentials
    if not credentials or not credentials.valid:
        raise RuntimeError(
            "Google account connect nahi hai. Google Drive page par "
            "'Connect Google Account' dabao aur Google login complete karo."
        )
    return credentials


def get_drive_service():
    try:
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive package missing hai. VS Code Terminal mein: python -m pip install -r requirements.txt"
        ) from exc
    credentials = get_google_credentials()
    return build("drive", "v3", credentials=credentials)


def create_drive_folder(service):
    metadata = {
        "name": "CarPhotoStudio",
        "mimeType": "application/vnd.google-apps.folder",
    }
    folder = service.files().create(
        body=metadata,
        fields="id, name, webViewLink",
    ).execute()
    return folder


def enable_folder_link(service, folder_id):
    permission = {
        "type": "anyone",
        "role": "reader",
    }
    service.permissions().create(
        fileId=folder_id,
        body=permission,
        fields="id",
    ).execute()

    return (
        "https://drive.google.com/drive/folders/"
        + folder_id
    )


page = st.sidebar.radio(
    "Menu",
    ["Dashboard", "Upload PDF", "My Photos", "Website Analyzer", "Logo Removal", "Non-AI Logo Removal", "Add New Logo", "Car Listing", "Google Drive", "Settings"],

if page == "Dashboard":
    st.markdown(
        """<div style="padding:.25rem 0 1rem 0;">
        <div style="color:#2dd4bf;font-size:.72rem;font-weight:800;letter-spacing:.16em;">YOUR AUTOMOTIVE WORKSPACE</div>
        <div style="font-size:2rem;font-weight:850;letter-spacing:-.04em;color:#f2f8fc;margin:.2rem 0;">Welcome back.</div>
        <div style="color:#9bb0c2;font-size:.96rem;">Manage vehicle photos, prepare listings and keep your showroom brand consistent.</div>
        </div>""",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Extracted Photos", len(st.session_state.photos))
    c2.metric(
        "Original PDF",
        "Uploaded" if st.session_state.original_pdf else "None",
    )
    c3.metric(
        "Saved Listings",
        len(st.session_state.car_listings),
    )
    st.info("Upload PDF se shuru karo. Photos ke baad Car Listing menu mein details edit karke card save karo.")

elif page == "Upload PDF":
    st.header("Upload Showroom PDF")
    pdf = st.file_uploader(
        "Choose a PDF file", type=["pdf"], key="showroom_pdf"
    )

    if pdf and st.button("Extract Photos", type="primary"):
        try:
            pdf_bytes = pdf.getvalue()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            extracted = []
            seen = set()

            with st.spinner("Extracting images..."):
                for page_num in range(len(doc)):
                    page_obj = doc[page_num]

                    for item in page_obj.get_images(full=True):
                        xref = item[0]
                        if xref in seen:
                            continue
                        seen.add(xref)

                        try:
                            info = doc.extract_image(xref)
                            image = Image.open(
                                io.BytesIO(info["image"])
                            ).convert("RGB")

                            if image.width < 400 or image.height < 250:
                                continue

                            output = io.BytesIO()
                            image.save(output, format="JPEG", quality=95)

                            extracted.append({
                                "name": f"car_photo_{len(extracted) + 1}.jpg",
                                "bytes": output.getvalue(),
                                "page": page_num + 1,
                                "width": image.width,
                                "height": image.height,
                            })
                        except Exception:
                            continue

            # Some PDFs flatten each page into a single image/vector layer. If no embedded
            # images were extractable, render a limited number of pages as JPG previews.
            if not extracted:
                doc.close()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                fallback_limit = min(len(doc), 30)
                for page_idx in range(fallback_limit):
                    try:
                        page_obj = doc[page_idx]
                        pix = page_obj.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
                        if pix.width < 400 or pix.height < 250:
                            continue
                        rendered = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        output = io.BytesIO()
                        rendered.save(output, format="JPEG", quality=92, optimize=True)
                        extracted.append({
                            "name": f"pdf_page_{page_idx + 1}_rendered.jpg",
                            "bytes": output.getvalue(),
                            "page": page_idx + 1,
                            "width": rendered.width,
                            "height": rendered.height,
                            "rendered_page": True,
                        })
                    except Exception:
                        continue
                doc.close()
                if extracted:
                    st.info("PDF mein alag embedded photos nahi mili; fallback mein PDF pages ko JPG copies ke roop mein render kiya gaya hai. In pages mein text/layout bhi ho sakta hai.")
            else:
                doc.close()

            st.session_state.photos = extracted
            st.session_state.original_pdf = {
                "name": pdf.name,
                "bytes": pdf_bytes,
            }
            st.session_state.pdf_text = extract_pdf_text(pdf_bytes)
            st.session_state.combined_pdf = None

            if extracted:
                st.success(f"{len(extracted)} images extracted!")
                st.info("Open My Photos to view and download.")
            else:
                st.warning(
                    "Embedded images nahi mile. Kuch PDFs ke liye "
                    "page-rendering extraction ki zaroorat hoti hai."
                )

        except Exception as e:
            st.error(f"PDF processing failed: {e}")

elif page == "My Photos":
    st.header("My Car Photos")
    photos = st.session_state.photos

    if not photos:
        st.info("Pehle Upload PDF mein PDF upload karke Extract Photos dabao.")
    else:
        st.write(f"Total photos: {len(photos)}")

        for start in range(0, len(photos), 3):
            cols = st.columns(3)

            for col, photo in zip(cols, photos[start:start + 3]):
                with col:
                    st.image(photo["bytes"], width="stretch")
                    st.caption(
                        f'{photo["name"]} | Page {photo["page"]} | '
                        f'{photo["width"]} Ã— {photo["height"]}'
                    )
                    st.download_button(
                        "Download JPG",
                        data=photo["bytes"],
                        file_name=photo["name"],
                        mime="image/jpeg",
                        key=f'jpg_{photo["name"]}',
                    )

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(
            zip_buffer, "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for photo in photos:
                archive.writestr(photo["name"], photo["bytes"])

        st.download_button(
            "Download All Photos (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="CarPhotoStudio_Photos.zip",
            mime="application/zip",
        )

        if st.button("Prepare Combined PDF"):
            try:
                images = [
                    Image.open(io.BytesIO(p["bytes"])).convert("RGB")
                    for p in photos
                ]

                if images:
                    output = io.BytesIO()
                    images[0].save(
                        output,
                        format="PDF",
                        save_all=True,
                        append_images=images[1:],
                        resolution=150,
                    )
                    st.session_state.combined_pdf = output.getvalue()
                    st.success("Combined PDF ready!")

            except Exception as e:
                st.error(f"PDF creation failed: {e}")

        if st.session_state.combined_pdf:
            st.download_button(
                "Download Combined Photos PDF",
                data=st.session_state.combined_pdf,
                file_name="CarPhotoStudio_All_Photos.pdf",
                mime="application/pdf",
            )

        if st.session_state.original_pdf:
            original = st.session_state.original_pdf
            st.subheader("Original Showroom PDF")
            st.download_button(
                "Download Original PDF",
                data=original["bytes"],
                file_name=original["name"],
                mime="application/pdf",
            )

elif page == "Car Listing":
    st.header("ðŸš˜ Car Listing Maker")
    st.caption("PDF se milne wali details suggestions hainâ€”save karne se pehle verify/edit karein.")

    photos = st.session_state.photos
    if not photos:
        st.info("Pehle Upload PDF mein showroom PDF upload karke Extract Photos dabao. Phir yahan listing bana sakte ho.")
    else:
        photo_labels = [f'{i + 1}. {p["name"]} (Page {p["page"]})' for i, p in enumerate(photos)]
        selected_label = st.selectbox("Listing ke liye car photo chuno", photo_labels)
        selected_index = photo_labels.index(selected_label)
        selected_photo = photos[selected_index]

        pdf_text = st.session_state.get("pdf_text", "")
        if st.button("PDF text se details suggest karo"):
            st.session_state.suggested_details = guess_car_details(pdf_text)
        suggested = st.session_state.get("suggested_details", {})

        with st.form("car_listing_form"):
            st.subheader("Car details â€” sab fields edit kar sakte ho")
            car_name = st.text_input("Car name / model", value=suggested.get("car_name", ""))
            registration = st.text_input("Registration number", value=suggested.get("registration", ""))
            year = st.text_input("Year", value=suggested.get("year", ""))
            fuel = st.selectbox(
                "Fuel",
                ["Not specified", "PETROL", "DIESEL", "CNG", "ELECTRIC", "HYBRID"],
                index=(["Not specified", "PETROL", "DIESEL", "CNG", "ELECTRIC", "HYBRID"].index(suggested.get("fuel", ""))
                       if suggested.get("fuel", "") in ["PETROL", "DIESEL", "CNG", "ELECTRIC", "HYBRID"] else 0),
            )
            owner = st.text_input("Owner", value=suggested.get("owner", ""))
            location = st.text_input("Location", value=suggested.get("location", ""))
            price = st.text_input("Starting price", value=suggested.get("price", ""))
            extra = st.text_area("Extra details / notes", placeholder="Variant, transmission, kilometres, contact number...")
            save_listing = st.form_submit_button("Save listing card", type="primary")

        if save_listing:
            listing = {
                "photo": selected_photo["bytes"],
                "car_name": car_name.strip(),
                "registration": registration.strip(),
                "year": year.strip(),
                "fuel": fuel,
                "owner": owner.strip(),
                "location": location.strip(),
                "price": price.strip(),
                "extra": extra.strip(),
            }
            st.session_state.car_listings.append(listing)
            st.success("Listing card save ho gaya!")

    if st.session_state.car_listings:
        st.divider()
        st.subheader("Saved listing cards")
        for idx, listing in enumerate(st.session_state.car_listings):
            left, right = st.columns([1, 2])
            with left:
                st.image(listing["photo"], width="stretch")
            with right:
                st.markdown(f'### {listing["car_name"] or "Car listing"}')
                if listing["registration"]:
                    st.write(f'**Registration:** {listing["registration"]}')
                if listing["year"]:
                    st.write(f'**Year:** {listing["year"]}')
                if listing["fuel"] != "Not specified":
                    st.write(f'**Fuel:** {listing["fuel"]}')
                if listing["owner"]:
                    st.write(f'**Owner:** {listing["owner"]}')
                if listing["location"]:
                    st.write(f'**Location:** {listing["location"]}')
                if listing["price"]:
                    st.write(f'**Start Price:** {listing["price"]}')
                if listing["extra"]:
                    st.write(listing["extra"])

                card_text = "\n".join([
                    listing["car_name"] or "Car listing",
                    f'Registration: {listing["registration"]}' if listing["registration"] else "",
                    f'Year: {listing["year"]}' if listing["year"] else "",
                    f'Fuel: {listing["fuel"]}' if listing["fuel"] != "Not specified" else "",
                    f'Owner: {listing["owner"]}' if listing["owner"] else "",
                    f'Location: {listing["location"]}' if listing["location"] else "",
                    f'Start Price: {listing["price"]}' if listing["price"] else "",
                    listing["extra"],
                ])
                st.download_button(
                    "Download details (.txt)",
                    data=card_text,
                    file_name=f"car_listing_{idx + 1}.txt",
                    mime="text/plain",
                    key=f"listing_txt_{idx}",
                )

elif page == "Google Drive":
    st.header("Google Drive Integration")
    st.write("Google account connect karke apna CarPhotoStudio folder banao.")

    process_google_oauth_callback()
    secret_path = get_client_secret_file()
    if not secret_path:
        st.error("Google OAuth client file nahi mili.")
        st.markdown(
            "Render mein **Environment â†’ Secret Files â†’ Add file** par jaakar "
            "`client_secret.json` upload karo aur file path "
            "`/etc/secrets/client_secret.json` rakho. Local VS Code ke liye "
            "`client_secret.json` ko `app.py` ke saath rakho."
        )
    else:
        connected = bool(
            st.session_state.get("google_credentials")
            and st.session_state.google_credentials.valid
        )
        if connected:
            st.success(
                f'Connected: {st.session_state.get("google_email", "Google account")}'
            )
        else:
            st.info("Google account connect karne ke liye neeche button dabao.")
            if st.button("1. Connect Google Account", type="primary"):
                try:
                    flow = make_google_oauth_flow()
                    auth_url, state = flow.authorization_url(
                        access_type="offline",
                        include_granted_scopes="true",
                        prompt="consent",
                    )
                    st.session_state.google_oauth_state = state
                    st.link_button("Google par login karo", auth_url)
                    st.caption(
                        "Google login button par click karo. Login ke baad website par wapas aa jaoge."
                    )
                except Exception as exc:
                    st.error(f"Google login start nahi ho paya: {exc}")

        if connected and st.button("2. Create Drive Folder"):
            try:
                service = get_drive_service()
                folder = create_drive_folder(service)
                st.session_state.drive_folder = folder
                st.success("CarPhotoStudio folder created!")
            except Exception as exc:
                st.error(f"Folder creation failed: {exc}")

        folder = st.session_state.get("drive_folder")
        if folder:
            st.write(f'Folder: {folder.get("name", "CarPhotoStudio")}')
            st.warning(
                "Link sharing enable karne par link wala koi bhi vyakti folder dekh sakega."
            )
            if st.button("3. Enable Link Sharing"):
                try:
                    service = get_drive_service()
                    link = enable_folder_link(service, folder["id"])
                    st.success("Link sharing enabled.")
                    st.code(link)
                    st.link_button("Open Google Drive Folder", link)
                except Exception as exc:
                    st.error(f"Sharing failed: {exc}")

elif page == "Website Analyzer":
    import re
    from urllib.parse import urljoin, urlparse

    st.header("ðŸŒ Website Analyzer")
    st.write("Public car listing link paste karo. Website ka text aur accessible photos collect karne ki koshish hogi.")
    st.warning("Sirf wahi content import karo jiske use/download ki permission tumhare paas ho. Login/CAPTCHA/protection ko bypass nahi kiya jayega. Original source photos untouched rahengi.")
    website_url = st.text_input("Website URL", placeholder="https://example.com/car-listing")
    if st.button("Analyze website", type="primary"):
        if not website_url.strip():
            st.error("Pehle website link paste karo.")
        else:
            try:
                from urllib.parse import urlparse
                import ipaddress
                import socket
                import requests
                from bs4 import BeautifulSoup
                parsed = urlparse(website_url.strip())
                if parsed.scheme not in ("http", "https") or not parsed.hostname:
                    raise ValueError("Sahi http:// ya https:// website URL daalo.")
                # Basic SSRF protection: reject localhost and private/reserved IPs.
                host = parsed.hostname.lower()
                if host in ("localhost",) or host.endswith((".local", ".internal")):
                    raise ValueError("Local/private network URLs allowed nahi hain.")
                try:
                    ip = ipaddress.ip_address(host)
                    if not ip.is_global:
                        raise ValueError("Private/local IP URLs allowed nahi hain.")
                except ValueError as ip_error:
                    if "allowed nahi" in str(ip_error):
                        raise
                response = requests.get(
                    website_url.strip(),
                    timeout=15,
                    headers={"User-Agent": "CarPhotoStudio/1.0 (public listing importer)"},
                    allow_redirects=True,
                    stream=True,
                )
                response.raise_for_status()
                if int(response.headers.get("content-length", "0") or 0) > 5_000_000:
                    raise ValueError("Website page 5 MB se badi hai; import rok diya.")
                if "text/html" not in response.headers.get("content-type", "").lower():
                    raise ValueError("Yeh URL HTML webpage nahi lagta.")
                body = response.raw.read(5_000_001)
                if len(body) > 5_000_000:
                    raise ValueError("Page 5 MB limit se badi hai; import rok diya.")
                soup = BeautifulSoup(body, "html.parser")
                for tag in soup(["script", "style", "noscript", "svg"]):
                    tag.decompose()
                title = soup.title.get_text(" ", strip=True) if soup.title else ""
                text_content = soup.get_text(" ", strip=True)[:30000]
                image_urls = []
                for img_tag in soup.find_all("img"):
                    src = img_tag.get("src") or img_tag.get("data-src") or img_tag.get("data-lazy-src")
                    if not src:
                        srcset = img_tag.get("srcset", "")
                        if srcset:
                            src = srcset.split(",")[-1].strip().split(" ")[0]
                    if src:
                        absolute = urljoin(response.url, src)
                        if urlparse(absolute).scheme in ("http", "https") and absolute not in image_urls:
                            image_urls.append(absolute)
                    if len(image_urls) >= 15:
                        break
                # Basic field suggestions, not authoritative facts.
                def find(pattern):
                    m = re.search(pattern, text_content, re.I)
                    return m.group(0) if m else ""
                suggestions = {
                    "Page title": title,
                    "Car / model clue": title,
                    "Year clue": find(r"\b(?:19|20)\d{2}\b"),
                    "Fuel clue": find(r"\b(?:petrol|gasoline|diesel|CNG|electric|hybrid)\b"),
                    "Transmission clue": find(r"\b(?:automatic|manual|CVT|DCT)\b"),
                    "Price clue": find(r"(?:â‚¹|INR|Rs\.?\s*)[\s\d,]+|\b\d[\d,]{3,}\s*/-"),
                    "Registration clue": find(r"\b[A-Z]{2}[- ]?\d{1,2}[- ]?[A-Z]{1,3}[- ]?\d{3,4}\b"),
                }
                st.session_state.website_analyzer_data = {
                    "url": response.url, "title": title, "text": text_content,
                    "images": [], "suggestions": suggestions,
                }
                st.success(f"Page read ho gaya. {len(image_urls)} tak public image links mile.")
                with st.spinner("Allowed public image links check kar rahe hain..."):
                    for img_url in image_urls:
                        try:
                            iparsed = urlparse(img_url)
                            if iparsed.hostname in ("localhost",) or (iparsed.hostname or "").endswith((".local", ".internal")):
                                continue
                            ir = requests.get(img_url, timeout=8, headers={"User-Agent": "CarPhotoStudio/1.0"}, stream=True, allow_redirects=True)
                            if not ir.ok or not ir.headers.get("content-type", "").lower().startswith("image/"):
                                ir.close(); continue
                            if int(ir.headers.get("content-length", "0") or 0) > 5_000_000:
                                ir.close(); continue
                            data = ir.raw.read(5_000_001); ir.close()
                            if len(data) > 5_000_000:
                                continue
                            try:
                                im = Image.open(io.BytesIO(data)); im.verify()
                            except Exception:
                                continue
                            st.session_state.website_analyzer_data["images"].append({"url": img_url, "bytes": data})
                        except Exception:
                            continue
            except Exception as e:
                st.error(f"Website analyze nahi ho saki: {e}")

    data = st.session_state.get("website_analyzer_data")
    if data:
        st.subheader("Website se mili details â€” verify karke hi save karo")
        st.caption(data.get("url", ""))
        suggestions = data.get("suggestions", {})
        with st.form("website_car_details"):
            car_name = st.text_input("Car name / model", value=suggestions.get("Car / model clue", ""))
            year = st.text_input("Year", value=suggestions.get("Year clue", ""))
            fuel = st.text_input("Fuel", value=suggestions.get("Fuel clue", ""))
            transmission = st.text_input("Transmission", value=suggestions.get("Transmission clue", ""))
            price = st.text_input("Price", value=suggestions.get("Price clue", ""))
            registration = st.text_input("Registration", value=suggestions.get("Registration clue", ""))
            save_details = st.form_submit_button("Save checked details to session")
        if save_details:
            st.session_state.website_car_details = {
                "source_url": data.get("url", ""), "car_name": car_name,
                "year": year, "fuel": fuel, "transmission": transmission,
                "price": price, "registration": registration,
            }
            st.success("Details is session mein save ho gayi hain. Abhi publish nahi hui hain.")
        st.subheader(f"Available photos ({len(data.get('images', []))})")
        for i, item in enumerate(data.get("images", [])):
            try:
                st.image(item["bytes"], caption=f"Imported copy {i+1} â€” source website", width=320)
                st.download_button(
                    f"Download photo {i+1}", data=item["bytes"],
                    file_name=f"website_photo_{i+1}.jpg", mime="application/octet-stream",
                    key=f"website_photo_dl_{i}",
                )
            except Exception:
                pass
        st.caption("Note: har website ki saari photos milna guaranteed nahi. Website restrictions, lazy loading aur JavaScript galleries ki wajah se kuch photos miss ho sakti hain.")

elif page == "Logo Removal":
    st.header("ðŸ§½ Local LaMa AI Logo Cleanup â€” Batch Photos")
    st.write(
        "Yeh local AI processing tool sirf aapke apne ya authorized logo/overlay ke liye hai. "
        "LaMa AI selected area ke aas-paas ke pixels se background reconstruct karne ki koshish karega; perfect result guaranteed nahi."
    )
    st.warning(
        "Sirf apne ya authorized showroom logo/overlay ke liye use karein. Source website/auction watermark, "
        "date-time/location stamp, inspection/insurance label, registration details ya provenance marks ko na hatayein. "
        "AI reconstruction hamesha perfect nahi hoti. Original photos unchanged rahengi aur edited copies alag milengi."
    )

    non_ai_source_choice = st.radio(
        "Photos kahan se leni hain?",
        ["PDF / My Photos", "Upload multiple photos"],
        key="non_ai_logo_source_choice",
    )
    non_ai_items = []
    if non_ai_source_choice == "PDF / My Photos":
        non_ai_items = [
            {"label": f'{i+1}. {photo.get("name", "photo")}', "bytes": photo["bytes"]}
            for i, photo in enumerate(st.session_state.get("photos", []))
            if photo.get("bytes")
        ]
        if not non_ai_items:
            st.info("Pehle Upload PDF â†’ PDF select â†’ Extract Photos dabao. Phir yahan sab extracted JPG ek saath milenge.")
    else:
        non_ai_uploads = st.file_uploader(
            "Apni/authorized JPG, PNG ya WebP photos chuno (multiple allowed)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="non_ai_logo_uploads",
        )
        non_ai_items = [
            {"label": u.name, "bytes": u.getvalue()}
            for u in (non_ai_uploads or [])
        ]

    if non_ai_items:
        st.write(f"Available photos: {len(non_ai_items)}")
        preview_labels = [item["label"] for item in non_ai_items]
        preview_label = st.selectbox(
            "Ek photo preview ke liye chuno (settings sab selected photos par lagu hongi)",
            preview_labels,
            key="non_ai_logo_preview_choice",
        )
        preview_item = next(item for item in non_ai_items if item["label"] == preview_label)
        _last_preview_label = st.session_state.get("non_ai_last_preview_label")
        if _last_preview_label != preview_label:
            st.session_state.pop("non_ai_manual_rect_pct", None)
            st.session_state["non_ai_last_preview_label"] = preview_label

        st.subheader("Logo area select karo")
        st.write("Yahan sliders se authorized logo/overlay ka area select karo. Red outline preview mein dikhai dega; white rectangle nahi lagaya jayega.")
        try:
            from PIL import ImageDraw
            with Image.open(io.BytesIO(preview_item["bytes"])) as im0:
                preview_image = im0.convert("RGB")

            left_col, right_col = st.columns(2)
            with left_col:
                logo_left = st.slider("Area ka left edge (% width)", 0, 95, 75, key="non_ai_logo_left")
                logo_top = st.slider("Area ka top edge (% height)", 0, 95, 80, key="non_ai_logo_top")
            with right_col:
                logo_width = st.slider("Area ki width (%)", 1, 50, 20, key="non_ai_logo_width")
                logo_height = st.slider("Area ki height (%)", 1, 50, 12, key="non_ai_logo_height")

            preview_copy = preview_image.copy()
            x1 = min(preview_copy.width - 1, round(preview_copy.width * logo_left / 100))
            y1 = min(preview_copy.height - 1, round(preview_copy.height * logo_top / 100))
            x2 = min(preview_copy.width - 1, max(x1 + 1, round(preview_copy.width * (logo_left + logo_width) / 100)))
            y2 = min(preview_copy.height - 1, max(y1 + 1, round(preview_copy.height * (logo_top + logo_height) / 100)))
            ImageDraw.Draw(preview_copy).rectangle((x1, y1, x2, y2), outline=(255, 35, 35), width=max(3, preview_copy.width // 300))
            st.image(preview_copy, caption="Preview: red outline = area to inpaint", use_container_width=True)
            manual_rect_selected = (logo_width > 0 and logo_height > 0)
        except Exception as exc:
            logo_left, logo_top, logo_width, logo_height = 75, 80, 20, 12
            manual_rect_selected = False
            st.error(f"Preview/area selection load nahi hui: {exc}")

        st.caption(
            "Manual rectangle ka relative area sab selected photos par apply hoga. Agar logo ki position ya size photos mein alag hai, "
            "unhe alag batch mein process karo. Ye non-AI tool sirf tumhare apne ya authorized logo/overlay ke liye hai; "
            "website/auction source watermark, location/date stamp ya inspection provenance mark ko na hatayein."
        )

        try:
            import cv2
            import numpy as np
            cv2_available = True
        except ImportError:
            cv2_available = False
            cv2 = None
            np = None

        non_ai_method = "LaMa AI inpainting"
        if cv2_available:
            st.success("LaMa AI mode selected: model logo area ke aas-paas ke background se reconstruction karega.")
        else:
            st.error("Mask banane ke liye OpenCV install hona zaroori hai. Koi white rectangle fallback nahi hoga.")
        inpaint_radius = st.slider(
            "Mask expansion (logo ke edge ke bahar kitna area include karein)",
            1, 9, 1,
            key="non_ai_logo_inpaint_radius",
            disabled=(not cv2_available),
        )
        consent_non_ai = st.checkbox(
            "Main confirm karta/karti hoon ki selected area mera apna logo ya authorized overlay hai.",
            key="non_ai_logo_consent",
        )
        st.caption("LaMa model local computer par chalega; OpenAI API charge nahi lagega. Pehli baar model load hone mein waqt lag sakta hai. Rectangle ke andar ke details badal sakte hain.")

        if st.button(
            "LaMa AI se sab selected photos clean karo",
            type="primary",
            disabled=(not consent_non_ai or not manual_rect_selected or not cv2_available or not non_ai_items),
            key="non_ai_logo_run",
        ):
            non_ai_outputs = []
            non_ai_errors = []
            progress = st.progress(0, text="Batch processing shuru ho rahi haiâ€¦")
            total = len(non_ai_items)
            for idx, item in enumerate(non_ai_items, start=1):
                try:
                    with Image.open(io.BytesIO(item["bytes"])) as im0:
                        im = im0.convert("RGB")
                    xa = min(im.width - 1, round(im.width * logo_left / 100))
                    ya = min(im.height - 1, round(im.height * logo_top / 100))
                    xb = min(im.width, max(xa + 1, round(im.width * (logo_left + logo_width) / 100)))
                    yb = min(im.height, max(ya + 1, round(im.height * (logo_top + logo_height) / 100)))
                    if not cv2_available:
                        raise RuntimeError("OpenCV missing hai; white rectangle se replace karna band hai. opencv-python-headless install karo.")
                    arr = np.array(im)
                    mask = np.zeros((im.height, im.width), dtype=np.uint8)
                    # Small dilation gives inpainting a little edge context without expanding the selected area too much.
                    cv2.rectangle(mask, (xa, ya), (max(xa, xb - 1), max(ya, yb - 1)), 255, thickness=-1)
                    kernel_size = max(3, int(inpaint_radius) * 2 + 1)
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=1)
                    # LaMa expects the source image and a grayscale mask where white pixels are replaced.
                    mask_image = Image.fromarray(mask, mode="L")
                    lama_model = get_lama_model()
                    result_im = lama_model(im, mask_image).convert("RGB")
                    buffer = io.BytesIO()
                    result_im.save(buffer, format="JPEG", quality=94, optimize=True)
                    stem = Path(item["label"]).stem.replace(" ", "_")[:60]
                    non_ai_outputs.append({
                        "name": f"NON_AI_LOGO_REMOVED_{stem}_{idx}.jpg",
                        "bytes": buffer.getvalue(),
                        "source": item["label"],
                    })
                except Exception as exc:
                    non_ai_errors.append(f'{item["label"]}: {str(exc)[:200]}')
                progress.progress(idx / total, text=f"Processed {idx}/{total}")
            st.session_state["non_ai_logo_outputs"] = non_ai_outputs
            st.session_state["non_ai_logo_errors"] = non_ai_errors
            st.rerun()

    non_ai_errors = st.session_state.get("non_ai_logo_errors", [])
    if non_ai_errors:
        st.warning("Kuch photos process nahi ho paayin:")
        for error in non_ai_errors:
            st.write(f"- {error}")
    non_ai_outputs = st.session_state.get("non_ai_logo_outputs", [])
    if non_ai_outputs:
        st.divider()
        st.subheader(f"Non-AI edited photos ready: {len(non_ai_outputs)} (originals unchanged)")
        zip_stream = io.BytesIO()
        with zipfile.ZipFile(zip_stream, "w", zipfile.ZIP_DEFLATED) as zf:
            for output in non_ai_outputs:
                zf.writestr(output["name"], output["bytes"])
        st.download_button(
            "Download all LaMa AI edited photos (ZIP)",
            data=zip_stream.getvalue(),
            file_name="CarPhotoStudio_LaMa_AI_Logo_Removed.zip",
            mime="application/zip",
            key="non_ai_logo_zip_download",
        )
        for j, output in enumerate(reversed(non_ai_outputs)):
            with st.expander(f'{output["name"]} â€¢ Source: {output["source"]}', expanded=(j == 0)):
                st.image(output["bytes"], width="stretch")
                st.download_button(
                    "Download JPG",
                    data=output["bytes"],
                    file_name=output["name"],
                    mime="image/jpeg",
                    key=f'non_ai_logo_download_{len(non_ai_outputs)-j-1}',
                )

elif page == "Add New Logo":
    st.header("ðŸ·ï¸ Add New Logo â€” Separate Batch Tool")
    st.write("Yeh page sirf naya showroom logo add karta hai. Purana logo yahan remove/cover nahi hota; uske liye Logo Removal page use karo.")
    st.info("Yeh local processing hai aur OpenAI API request nahi karta. Original photos/PDF unchanged rahenge; edited copies alag download hongi.")

    source_uploads = st.file_uploader(
        "PDF aur JPG/PNG photos upload karo (multiple files allowed)",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="add_logo_sources_separate",
    )
    include_my_photos = st.checkbox(
        f'My Photos ke extracted photos bhi add karo ({len(st.session_state.get("photos", []))} available)',
        value=bool(st.session_state.get("photos", [])),
        key="add_logo_include_my_photos_separate",
    )
    logo_upload = st.file_uploader(
        "Apna naya logo upload karo (transparent PNG recommended)",
        type=["png", "jpg", "jpeg", "webp"],
        key="add_logo_new_upload_separate",
    )
    sources = []
    for upload in (source_uploads or []):
        sources.append({"name": upload.name, "bytes": upload.getvalue()})
    if include_my_photos:
        for idx, photo in enumerate(st.session_state.get("photos", [])):
            if photo.get("bytes"):
                sources.append({"name": photo.get("name", f"my_photo_{idx+1}.jpg"), "bytes": photo["bytes"]})
    if sources:
        st.write(f"Selected source files: {len(sources)}. Har PDF ka har page aur har selected photo process hoga.")
    else:
        st.info("PDF/photos upload karo ya My Photos checkbox use karo.")

    logo_position = st.selectbox(
        "Logo position", ["Top-right", "Bottom-right", "Top-left", "Bottom-left", "Custom position"],
        index=1, key="separate_add_logo_position",
    )
    c1, c2 = st.columns(2)
    with c1:
        logo_width_pct = st.slider("Logo width (%)", 3, 30, 12, key="separate_add_logo_width")
    with c2:
        logo_margin_pct = st.slider("Margin (%)", 0.0, 8.0, 2.0, 0.5, key="separate_add_logo_margin")
    custom_x, custom_y = 75.0, 85.0
    if logo_position == "Custom position":
        cx, cy = st.columns(2)
        with cx:
            custom_x = float(st.slider("Logo left (%)", 0, 100, 75, key="separate_add_logo_x"))
        with cy:
            custom_y = float(st.slider("Logo top (%)", 0, 100, 85, key="separate_add_logo_y"))
    logo_permission = st.checkbox(
        "Main in files ko edit karne ka owner hoon ya mere paas permission hai.",
        key="separate_add_logo_permission",
    )
    if logo_upload:
        try:
            st.image(logo_upload.getvalue(), caption="New logo preview", width=200)
            with Image.open(io.BytesIO(logo_upload.getvalue())) as logo_check:
                logo_ratio = logo_check.width / max(1, logo_check.height)
        except Exception as exc:
            logo_ratio = None
            st.error(f"Logo file open nahi hui: {exc}")
    else:
        logo_ratio = None
        st.info("Pehle apna naya logo upload karo.")

    if st.button(
        "Sabhi PDFs aur photos par sirf naya logo add karo",
        type="primary", disabled=not bool(sources and logo_upload and logo_ratio and logo_permission),
        key="separate_add_logo_process",
    ):
        try:
            from PIL import ImageDraw
            logo_bytes = logo_upload.getvalue()
            with Image.open(io.BytesIO(logo_bytes)) as temp_logo:
                logo_image = temp_logo.convert("RGBA")
            outputs, problems = [], []
            progress = st.progress(0, text="Logo sabhi files par add ho raha haiâ€¦")

            def _box(canvas_w, canvas_h, box_w, box_h, corner, margin_pct, custom_x_pct, custom_y_pct):
                mx, my = canvas_w * margin_pct / 100.0, canvas_h * margin_pct / 100.0
                if corner == "Custom position":
                    xx = (canvas_w - box_w) * custom_x_pct / 100.0
                    yy = (canvas_h - box_h) * custom_y_pct / 100.0
                else:
                    xx = mx if "left" in corner.lower() else canvas_w - box_w - mx
                    yy = my if "top" in corner.lower() else canvas_h - box_h - my
                xx = max(0, min(max(0, canvas_w - box_w), xx))
                yy = max(0, min(max(0, canvas_h - box_h), yy))
                return xx, yy, xx + box_w, yy + box_h

            for source_idx, source in enumerate(sources):
                source_name = Path(source["name"]).name
                stem = Path(source_name).stem or f"source_{source_idx+1}"
                try:
                    if source_name.lower().endswith(".pdf"):
                        doc = fitz.open(stream=source["bytes"], filetype="pdf")
                        for page_obj in doc:
                            pw, ph = float(page_obj.rect.width), float(page_obj.rect.height)
                            nw = pw * logo_width_pct / 100.0
                            nh = min(nw / logo_ratio, ph * 0.22)
                            nw = nh * logo_ratio
                            if logo_position == "Custom position":
                                nx0, ny0, nx1, ny1 = _box(pw, ph, nw, nh, logo_position, logo_margin_pct, custom_x, custom_y)
                            else:
                                nx0, ny0, nx1, ny1 = _box(pw, ph, nw, nh, logo_position, logo_margin_pct, custom_x, custom_y)
                            page_obj.insert_image(fitz.Rect(nx0, ny0, nx1, ny1), stream=logo_bytes, keep_proportion=True, overlay=True)
                        outputs.append((f"{stem}_with_new_logo.pdf", doc.tobytes(garbage=4, deflate=True), "application/pdf"))
                        for page_idx, page_obj in enumerate(doc):
                            pix = page_obj.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False)
                            rendered = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            buf = io.BytesIO(); rendered.save(buf, format="JPEG", quality=92, optimize=True)
                            outputs.append((f"{stem}_page_{page_idx+1:03d}_with_logo.jpg", buf.getvalue(), "image/jpeg"))
                        doc.close()
                    else:
                        with Image.open(io.BytesIO(source["bytes"])) as original:
                            canvas = original.convert("RGBA")
                        nw = max(1, int(canvas.width * logo_width_pct / 100))
                        nh = max(1, int(nw / logo_ratio))
                        max_h = max(1, int(canvas.height * 0.22))
                        if nh > max_h:
                            nh = max_h
                            nw = max(1, int(nh * logo_ratio))
                        resized = logo_image.resize((nw, nh), Image.Resampling.LANCZOS)
                        x0, y0, _, _ = _box(canvas.width, canvas.height, nw, nh, logo_position, logo_margin_pct, custom_x, custom_y)
                        canvas.alpha_composite(resized, (int(x0), int(y0)))
                        buf = io.BytesIO(); canvas.convert("RGB").save(buf, format="JPEG", quality=92, optimize=True)
                        outputs.append((f"{stem}_with_new_logo.jpg", buf.getvalue(), "image/jpeg"))
                except Exception as exc:
                    problems.append(f"{source_name}: {str(exc)[:200]}")
                progress.progress((source_idx + 1) / len(sources), text=f"Processed {source_idx+1}/{len(sources)} files")
            if outputs:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                    names = set()
                    for name, data, _mime in outputs:
                        safe_name = name; counter = 2
                        while safe_name in names:
                            safe_name = f"{Path(name).stem}_{counter}{Path(name).suffix}"; counter += 1
                        names.add(safe_name)
                        archive.writestr(safe_name, data)
                st.session_state["separate_add_logo_results"] = {"outputs": outputs, "zip": zip_buffer.getvalue()}
                st.success(f"Ho gaya! {len(outputs)} edited output file(s) taiyar hain. Original files unchanged.")
            if problems:
                st.warning("Kuch files process nahi ho paayin:")
                for issue in problems:
                    st.write(f"- {issue}")
        except Exception as exc:
            st.error("Logo add karte waqt error aaya.")
            st.caption(f"Technical detail: {str(exc)[:400]}")

    add_logo_results = st.session_state.get("separate_add_logo_results")
    if add_logo_results:
        st.divider()
        st.subheader("Download branded copies")
        st.download_button(
            "Download all edited PDFs + JPGs (ZIP)", data=add_logo_results["zip"],
            file_name="CarPhotoStudio_New_Logo_Batch.zip", mime="application/zip",
            key="separate_add_logo_zip_download",
        )
        with st.expander("Individual output files"):
            for i, (name, data, mime) in enumerate(add_logo_results["outputs"]):
                st.download_button(
                    f"Download {name}", data=data, file_name=name, mime=mime,
                    key=f"separate_add_logo_individual_{i}",
                )

elif page == "AI Image Studio":
    st.header("âœ¨ Local AI Photo Tools")
    st.info(
        "OpenAI-based image generation/editing disable kar diya gaya hai. Local LaMa AI logo-area cleanup ke liye "
        "left menu mein **Non-AI Logo Removal** kholo. Naya showroom logo lagane ke liye **Add New Logo** page use karo."
    )
    st.caption("LaMa background inpainting karta hai; yeh text-to-image generator nahi hai. Original files alag safe rahengi.")

elif page == "Settings":
    st.header("Settings")
    st.info(
        "Local LaMa logo cleanup ke liye OpenAI API key ki zaroorat nahi hai. "
        "Original files unchanged rehti hain aur edited copies alag download hoti hain."
    )


st.markdown(
    """<div style="margin-top:2.5rem;padding-top:1rem;border-top:1px solid rgba(153,180,204,.18);color:#71889b;font-size:.74rem;display:flex;justify-content:space-between;gap:.7rem;flex-wrap:wrap;">
    <span>CarPhotoStudio Â· Showroom workspace</span><span>Original files stay separate from edited copies</span></div>""",
    unsafe_allow_html=True,
)

