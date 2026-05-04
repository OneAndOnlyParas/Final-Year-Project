import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models
import time
from datetime import datetime


st.set_page_config(
    page_title="Paras Bca Project",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


if "page" not in st.session_state:
    st.session_state.page = "Home"
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "total_predictions" not in st.session_state:
    st.session_state.total_predictions = 0
if "animal_counts" not in st.session_state:
    st.session_state.animal_counts = {}
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "analyzing" not in st.session_state:
    st.session_state.analyzing = False

def navigate(page_name):
    st.session_state.page = page_name
    st.rerun()

def reset_prediction():
    st.session_state.last_result = None
    st.rerun()

# ========================
# CSS
# ========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

/* ---- Keyframes ---- */
@keyframes fadeUp    { from { opacity:0; transform:translateY(18px); } to { opacity:1; transform:translateY(0); } }
@keyframes fadeIn    { from { opacity:0; } to { opacity:1; } }
@keyframes scaleIn   { from { opacity:0; transform:scale(0.93); } to { opacity:1; transform:scale(1); } }
@keyframes slideL    { from { opacity:0; transform:translateX(-20px); } to { opacity:1; transform:translateX(0); } }
@keyframes slideR    { from { opacity:0; transform:translateX(20px); } to { opacity:1; transform:translateX(0); } }
@keyframes glow      { 0%,100% { box-shadow:0 0 0 0 rgba(99,102,241,0); } 50% { box-shadow:0 0 20px 6px rgba(99,102,241,0.22); } }
@keyframes pulse     { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
@keyframes gradFlow  { 0%,100% { background-position:0% 50%; } 50% { background-position:100% 50%; } }
@keyframes barGrow   { from { transform:scaleX(0); } to { transform:scaleX(1); } }
@keyframes stepFade  { from { opacity:0; transform:translateX(8px); } to { opacity:1; transform:translateX(0); } }
@keyframes countPop  { from { opacity:0; transform:scale(0.75); } to { opacity:1; transform:scale(1); } }

/* ---- Base ---- */
*, *::before, *::after { box-sizing:border-box; }

.stApp {
    background: #07090f;
    color: #cdd1ec;
    font-family: 'Inter', system-ui, sans-serif !important;
}
.stApp p, .stApp span, .stApp div, .stApp label,
.stApp li, .stApp td, .stApp th {
    font-family: 'Inter', sans-serif !important;
}
h1,h2,h3,h4,h5,h6 {
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    color: #ffffff !important;
    letter-spacing:-0.5px;
}
#MainMenu, footer, header { visibility:hidden; }
.stDeployButton { display:none; }
.block-container {
    padding-top: 1.6rem !important;
    padding-bottom: 3rem !important;
    max-width: 1080px;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0d1a 0%, #070910 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.1);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
[data-testid="stSidebar"] .stRadio > div { gap: 0.1rem; }
[data-testid="stSidebar"] .stRadio label {
    font-family:'Inter',sans-serif !important;
    font-size:0.88rem !important; font-weight:500 !important;
    color:#5a6090 !important;
    padding:0.62rem 1rem !important; border-radius:10px !important;
    cursor:pointer; transition:all 0.18s ease !important;
    border:1px solid transparent !important;
    display:block !important; width:100% !important; margin:0 !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color:#c0c8f0 !important; background:rgba(99,102,241,0.07) !important;
    border-color:rgba(99,102,241,0.12) !important;
}
[data-testid="stSidebar"] .stRadio [data-checked="true"] label,
[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    color:#a5abff !important; background:rgba(99,102,241,0.12) !important;
    border-color:rgba(99,102,241,0.22) !important;
}

/* ---- Buttons ---- */
.stButton button {
    font-family:'Inter',sans-serif !important; font-weight:600 !important;
    border-radius:10px !important; transition:all 0.2s ease !important;
    letter-spacing:0.01em; font-size:0.88rem !important;
}
.stButton button[kind="primary"],
.stButton button[data-testid="baseButton-primary"] {
    background:linear-gradient(135deg,#5d60ef 0%,#7c3aed 100%) !important;
    border:none !important; color:#fff !important;
    box-shadow:0 4px 18px rgba(93,96,239,0.38) !important;
}
.stButton button[kind="primary"]:hover {
    background:linear-gradient(135deg,#6e71f5 0%,#8d4af5 100%) !important;
    box-shadow:0 6px 26px rgba(93,96,239,0.55) !important;
    transform:translateY(-1px) !important;
}
.stButton button[kind="secondary"],
.stButton button[data-testid="baseButton-secondary"] {
    background:rgba(99,102,241,0.07) !important;
    border:1px solid rgba(99,102,241,0.18) !important;
    color:#8890d8 !important;
}
.stButton button[kind="secondary"]:hover {
    background:rgba(99,102,241,0.12) !important;
    border-color:rgba(99,102,241,0.32) !important;
    transform:translateY(-1px) !important;
}

/* ---- Streamlit widgets ---- */
.stProgress > div > div > div {
    background:linear-gradient(90deg,#5d60ef,#7c3aed) !important;
    border-radius:999px !important; transition:width 0.35s ease !important;
}
.stProgress > div > div {
    background:rgba(255,255,255,0.04) !important; border-radius:999px !important;
}
[data-testid="stFileUploader"] {
    background:rgba(99,102,241,0.035) !important;
    border:2px dashed rgba(99,102,241,0.2) !important;
    border-radius:14px !important; transition:border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color:rgba(99,102,241,0.42) !important; }
[data-testid="stExpander"] {
    background:#0c0f1f !important;
    border:1px solid rgba(99,102,241,0.13) !important; border-radius:16px !important;
}
[data-testid="stExpander"] summary {
    font-family:'Space Grotesk',sans-serif !important;
    font-weight:600 !important; color:#b8bde0 !important;
}
[data-testid="stCameraInput"] { border-radius:14px !important; }
.stRadio label { font-family:'Inter',sans-serif !important; color:#7880a8 !important; }

/* =====================
   PAGE WRAPPER
   ===================== */
.pw { animation:fadeUp 0.42s ease both; }

/* =====================
   SECTION LABEL
   ===================== */
.slbl {
    font-size:0.68rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.13em; color:#30344e; margin-bottom:0.75rem;
    display:flex; align-items:center; gap:0.5rem;
}
.slbl::after {
    content:''; flex:1; height:1px;
    background:linear-gradient(90deg,rgba(99,102,241,0.12),transparent);
}

/* =====================
   HOME
   ===================== */
.hero {
    position:relative; overflow:hidden;
    padding:3.8rem 3rem 3rem;
    border-radius:24px;
    background:linear-gradient(135deg,#0c1030 0%,#10152e 55%,#0d1226 100%);
    border:1px solid rgba(99,102,241,0.14);
    margin-bottom:1.6rem;
    animation:fadeUp 0.48s ease;
}
.hero::before {
    content:''; position:absolute; top:-90px; right:-80px;
    width:340px; height:340px; border-radius:50%;
    background:radial-gradient(circle,rgba(93,96,239,0.2) 0%,transparent 68%);
    pointer-events:none;
}
.hero::after {
    content:''; position:absolute; bottom:-70px; left:28%;
    width:280px; height:280px; border-radius:50%;
    background:radial-gradient(circle,rgba(124,58,237,0.12) 0%,transparent 68%);
    pointer-events:none;
}
.hero-ey {
    font-size:0.7rem; font-weight:700; letter-spacing:0.15em;
    text-transform:uppercase; color:#5d60ef;
    display:flex; align-items:center; gap:0.5rem; margin-bottom:0.9rem;
}
.hero-ey::before {
    content:''; width:18px; height:2px;
    background:#5d60ef; border-radius:2px;
}
.hero-t {
    font-family:'Space Grotesk',sans-serif !important;
    font-size:3.3rem !important; font-weight:800 !important;
    line-height:1.08 !important; letter-spacing:-1.8px !important;
    margin:0 0 1rem 0 !important;
    background:linear-gradient(130deg,#eaecff 0%,#9fa8ff 45%,#c084fc 100%);
    background-size:200% 200%;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    animation:gradFlow 4.5s ease infinite;
}
.hero-sub {
    font-size:1rem; color:#5a628a; line-height:1.7;
    max-width:520px; margin-bottom:1.8rem;
}
.hero-pills { display:flex; flex-wrap:wrap; gap:0.45rem; margin-top:1.8rem; }
.hero-pill {
    display:inline-flex; align-items:center; gap:0.38rem;
    padding:0.28rem 0.7rem; border-radius:999px; font-size:0.72rem;
    font-weight:600; background:rgba(93,96,239,0.09);
    border:1px solid rgba(93,96,239,0.18); color:#8890d8;
}
.hero-pill .dot { width:5px; height:5px; background:#5d60ef; border-radius:50%; }

.feat-grid {
    display:grid; grid-template-columns:repeat(3,1fr); gap:0.9rem;
    margin-bottom:1.4rem; animation:fadeUp 0.5s 0.08s ease both;
}
.fc {
    position:relative; background:rgba(255,255,255,0.02);
    border:1px solid rgba(255,255,255,0.05); border-radius:16px;
    padding:1.3rem 1.1rem; transition:all 0.22s ease; overflow:hidden;
}
.fc::before {
    content:''; position:absolute; inset:0; border-radius:16px; opacity:0;
    background:linear-gradient(135deg,rgba(93,96,239,0.07),rgba(124,58,237,0.04));
    transition:opacity 0.22s ease;
}
.fc:hover { border-color:rgba(99,102,241,0.26); transform:translateY(-3px); box-shadow:0 10px 36px rgba(0,0,0,0.35); }
.fc:hover::before { opacity:1; }
.fc-icon {
    width:40px; height:40px; border-radius:11px; font-size:1.15rem;
    background:rgba(93,96,239,0.09); border:1px solid rgba(93,96,239,0.15);
    display:flex; align-items:center; justify-content:center; margin-bottom:0.85rem;
}
.fc-t { font-family:'Space Grotesk',sans-serif; font-size:0.88rem; font-weight:600; color:#c0c8ee; margin-bottom:0.28rem; }
.fc-d { font-size:0.75rem; color:#3e4460; line-height:1.5; }

.atag {
    display:inline-flex; align-items:center; gap:0.3rem;
    padding:0.28rem 0.7rem; border-radius:999px; font-size:0.76rem; font-weight:500;
    background:rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.055);
    color:#5a628a; transition:all 0.17s ease;
}
.atag:hover { background:rgba(93,96,239,0.08); border-color:rgba(99,102,241,0.2); color:#9fa8ff; }

/* =====================
   DETECT
   ===================== */
.det-hdr { animation:fadeUp 0.38s ease; margin-bottom:1.3rem; }
.det-t { font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:700; color:#fff; margin-bottom:0.18rem; }
.det-s { font-size:0.85rem; color:#3e4460; }

/* Mobile camera notice */
.mob-notice {
    display:flex; align-items:flex-start; gap:0.7rem;
    background:rgba(14,165,233,0.08); border:1px solid rgba(14,165,233,0.18);
    border-radius:12px; padding:0.9rem 1.1rem; margin-bottom:1rem;
    animation:fadeUp 0.35s ease;
}
.mob-notice-icon { font-size:1.1rem; flex-shrink:0; }
.mob-notice-text { font-size:0.82rem; color:#7ec8e8; line-height:1.55; }
.mob-notice-text strong { color:#bae6fd; }

/* Loading steps */
.load-step {
    display:flex; align-items:center; gap:0.7rem;
    font-size:0.85rem; font-weight:500; color:#8890b8;
    animation:stepFade 0.3s ease;
    padding:0.2rem 0;
}
.load-step .ls-icon { font-size:1rem; width:1.4rem; text-align:center; }
.load-step.active { color:#a5abff; }
.load-step.done   { color:#34d399; }

/* Big result card */
.rc {
    background:linear-gradient(145deg,rgba(11,15,40,0.95) 0%,rgba(14,18,48,0.95) 100%);
    border:1px solid rgba(93,96,239,0.28); border-radius:20px;
    padding:1.8rem 1.8rem 1.5rem;
    box-shadow:0 12px 48px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
    animation:scaleIn 0.38s ease;
    position:relative; overflow:hidden;
}
.rc::before {
    content:''; position:absolute; top:-40px; right:-40px;
    width:180px; height:180px; border-radius:50%;
    background:radial-gradient(circle,rgba(93,96,239,0.14) 0%,transparent 70%);
    pointer-events:none;
}
.rc-lbl {
    font-size:0.65rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.15em; color:#2c3050; margin-bottom:0.6rem;
}
.rc-emoji { font-size:2.4rem; line-height:1; margin-bottom:0.4rem; }
.rc-name {
    font-family:'Space Grotesk',sans-serif; font-size:2.4rem; font-weight:800;
    color:#c5c9ff; letter-spacing:-1px; line-height:1.05; margin-bottom:0.5rem;
}
.rc-conf-high   { display:flex; align-items:center; gap:0.45rem; font-size:1rem; font-weight:700; color:#34d399; }
.rc-conf-medium { display:flex; align-items:center; gap:0.45rem; font-size:1rem; font-weight:700; color:#fbbf24; }
.rc-conf-low    { display:flex; align-items:center; gap:0.45rem; font-size:1rem; font-weight:700; color:#f87171; }
.rc-dot {
    width:9px; height:9px; border-radius:50%; flex-shrink:0;
    animation:pulse 2s ease infinite;
}
.rc-dot-high   { background:#34d399; box-shadow:0 0 8px rgba(52,211,153,0.65); }
.rc-dot-medium { background:#fbbf24; box-shadow:0 0 8px rgba(251,191,36,0.65); }
.rc-dot-low    { background:#f87171; box-shadow:0 0 8px rgba(248,113,113,0.65); }

/* Warn */
.wbx {
    display:flex; align-items:flex-start; gap:0.65rem;
    background:rgba(100,44,7,0.22); border:1px solid rgba(251,191,36,0.18);
    border-radius:12px; padding:0.85rem 1rem; margin-top:0.75rem; animation:fadeIn 0.35s ease;
}
.wbx-t { font-size:0.8rem; color:#c9a840; line-height:1.55; }

/* Pred bars */
.pst { font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.13em; color:#2c3050; margin:1.2rem 0 0.6rem 0; }
.pi { margin-bottom:0.55rem; }
.pi-row { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.28rem; }
.pi-n  { font-size:0.83rem; font-weight:500; color:#7880a8; }
.pi-n1 { font-size:0.88rem; font-weight:600; color:#c0c8ee; }
.pi-p  { font-size:0.8rem; font-weight:600; color:#5a628a; font-variant-numeric:tabular-nums; }
.pi-p1 { color:#a5abff; }

/* =====================
   SPECIES PANEL — UPGRADED
   ===================== */
.sp-wrap {
    background:#09101f;
    border:1px solid rgba(99,102,241,0.14);
    border-radius:20px; overflow:hidden;
    animation:fadeUp 0.42s ease;
    margin-top:1.5rem;
}

/* Banner strip */
.sp-banner {
    background:linear-gradient(135deg,#0e1535 0%,#131c40 100%);
    padding:1.8rem 2rem 1.6rem;
    border-bottom:1px solid rgba(99,102,241,0.1);
    position:relative; overflow:hidden;
}
.sp-banner::after {
    content:''; position:absolute; top:-60px; right:-50px;
    width:200px; height:200px; border-radius:50%;
    background:radial-gradient(circle,rgba(93,96,239,0.15) 0%,transparent 70%);
    pointer-events:none;
}
.sp-bn-top { display:flex; align-items:flex-start; gap:1.2rem; }
.sp-big-emoji {
    width:64px; height:64px; border-radius:18px; font-size:2rem;
    background:rgba(93,96,239,0.1); border:1px solid rgba(93,96,239,0.2);
    display:flex; align-items:center; justify-content:center; flex-shrink:0;
}
.sp-bn-name {
    font-family:'Space Grotesk',sans-serif; font-size:1.55rem;
    font-weight:800; color:#e8eaff; letter-spacing:-0.5px;
}
.sp-bn-sub { font-size:0.8rem; color:#3e4460; margin-top:0.22rem; }
.sp-overview {
    font-size:0.88rem; color:#5a628a; line-height:1.7;
    margin-top:1rem; max-width:560px;
}

/* Stats row */
.sp-stats {
    display:grid; grid-template-columns:repeat(4,1fr);
    border-bottom:1px solid rgba(255,255,255,0.04);
}
.sp-stat {
    padding:1.1rem 1.2rem;
    border-right:1px solid rgba(255,255,255,0.04);
}
.sp-stat:last-child { border-right:none; }
.sp-stat-lbl { font-size:0.62rem; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; color:#2c3050; margin-bottom:0.35rem; }
.sp-stat-val { font-size:0.88rem; font-weight:600; color:#b0b8e0; line-height:1.3; }

/* Body sections */
.sp-body { padding:1.4rem 2rem; display:flex; flex-direction:column; gap:1.2rem; }

.sp-sec { }
.sp-sec-title {
    font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.13em;
    color:#2c3050; margin-bottom:0.65rem;
    display:flex; align-items:center; gap:0.4rem;
}
.sp-sec-title::before { content:''; width:12px; height:2px; background:#5d60ef; border-radius:2px; }
.sp-sec-text { font-size:0.86rem; color:#6870a0; line-height:1.72; }

/* Fact cards */
.sp-facts-row { display:flex; flex-direction:column; gap:0.5rem; }
.sp-fact {
    display:flex; align-items:flex-start; gap:0.85rem;
    background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.04);
    border-radius:11px; padding:0.75rem 1rem;
    animation:slideL 0.3s ease both;
}
.sp-fact:nth-child(2) { animation-delay:0.07s; }
.sp-fact:nth-child(3) { animation-delay:0.14s; }
.sp-fact-num {
    width:22px; height:22px; border-radius:7px; flex-shrink:0;
    background:rgba(93,96,239,0.1); border:1px solid rgba(93,96,239,0.16);
    display:flex; align-items:center; justify-content:center;
    font-size:0.65rem; font-weight:700; color:#6870c0; margin-top:0.05rem;
}
.sp-fact-text { font-size:0.83rem; color:#5a628a; line-height:1.55; }

/* Conservation badge */
.cbadge {
    display:inline-flex; align-items:center; gap:0.35rem;
    padding:0.25rem 0.65rem; border-radius:999px; font-size:0.72rem; font-weight:700; letter-spacing:0.04em;
}
.cbadge-lc  { background:rgba(20,83,45,0.45); color:#4ade80; border:1px solid rgba(74,222,128,0.22); }
.cbadge-nt  { background:rgba(30,58,95,0.45); color:#60a5fa; border:1px solid rgba(96,165,250,0.22); }
.cbadge-vu  { background:rgba(66,32,6,0.45); color:#fb923c; border:1px solid rgba(251,146,60,0.22); }
.cbadge-en  { background:rgba(69,10,10,0.45); color:#f87171; border:1px solid rgba(248,113,113,0.22); }
.cbadge-dom { background:rgba(45,26,95,0.45); color:#c084fc; border:1px solid rgba(192,132,252,0.22); }

/* =====================
   DASHBOARD
   ===================== */
.dsg { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1.6rem; animation:fadeUp 0.42s ease; }
.ds {
    background:rgba(255,255,255,0.022); border:1px solid rgba(255,255,255,0.055);
    border-radius:16px; padding:1.35rem 1.2rem; position:relative;
    overflow:hidden; transition:border-color 0.2s ease;
}
.ds:hover { border-color:rgba(99,102,241,0.2); }
.ds::after {
    content:''; position:absolute; bottom:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,#5d60ef,#7c3aed); opacity:0; transition:opacity 0.2s;
}
.ds:hover::after { opacity:1; }
.ds-num {
    font-family:'Space Grotesk',sans-serif; font-size:2.1rem; font-weight:700;
    color:#9fa8ff; letter-spacing:-0.5px; animation:countPop 0.45s ease;
}
.ds-lbl { font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#2c3050; margin-top:0.25rem; }
.ds-ico { position:absolute; top:1rem; right:1.1rem; font-size:1.3rem; opacity:0.18; }
.dbd { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.04); border-radius:16px; padding:1.3rem; margin-bottom:1.3rem; animation:fadeUp 0.45s 0.05s ease both; }
.dbd-row { display:flex; align-items:center; gap:0.8rem; margin-bottom:0.65rem; }
.dbd-animal { min-width:115px; font-size:0.84rem; font-weight:500; color:#7880a8; }
.dbd-cnt { font-size:0.74rem; font-weight:700; color:#3e4460; min-width:26px; text-align:right; }
.dh { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.04); border-radius:16px; padding:1.3rem; animation:fadeUp 0.45s 0.1s ease both; }
.hr-row {
    display:flex; align-items:center; gap:0.9rem;
    padding:0.65rem 0.75rem; border-radius:10px; margin-bottom:0.28rem;
    background:rgba(255,255,255,0.018); border:1px solid rgba(255,255,255,0.035);
    transition:border-color 0.15s; animation:slideL 0.28s ease both;
}
.hr-row:hover { border-color:rgba(99,102,241,0.14); }
.hr-n { font-size:0.7rem; font-weight:700; color:#1e2236; min-width:22px; font-variant-numeric:tabular-nums; }
.hr-a { font-size:0.84rem; font-weight:600; color:#7880a8; min-width:105px; }
.hr-c-high   { font-size:0.78rem; font-weight:700; color:#34d399; }
.hr-c-medium { font-size:0.78rem; font-weight:700; color:#fbbf24; }
.hr-c-low    { font-size:0.78rem; font-weight:700; color:#f87171; }
.hr-t { margin-left:auto; font-size:0.7rem; color:#1e2236; font-variant-numeric:tabular-nums; }

/* =====================
   ABOUT
   ===================== */
.ab-hero {
    background:linear-gradient(135deg,#0c1030 0%,#101530 100%);
    border:1px solid rgba(99,102,241,0.12); border-radius:20px;
    padding:2rem 2.3rem; margin-bottom:1.4rem; animation:fadeUp 0.42s ease;
}
.ab-t { font-family:'Space Grotesk',sans-serif; font-size:1.45rem; font-weight:700; color:#e8eaff; margin-bottom:0.65rem; }
.ab-b { font-size:0.87rem; color:#4a5278; line-height:1.75; }

.gc {
    background:rgba(255,255,255,0.025); backdrop-filter:blur(10px);
    border:1px solid rgba(255,255,255,0.06); border-radius:18px; padding:1.3rem 1.4rem;
}
.tech-i {
    display:flex; align-items:center; gap:0.9rem; padding:0.8rem 0;
    border-bottom:1px solid rgba(255,255,255,0.04); animation:slideL 0.28s ease both;
}
.tech-i:last-child { border-bottom:none; }
.tech-ico {
    width:36px; height:36px; border-radius:10px; font-size:1.05rem;
    background:rgba(93,96,239,0.08); border:1px solid rgba(93,96,239,0.13);
    display:flex; align-items:center; justify-content:center; flex-shrink:0;
}
.tech-n { font-size:0.87rem; font-weight:600; color:#b8c0e8; }
.tech-d { font-size:0.73rem; color:#3e4460; margin-top:0.08rem; }
.arch-i { display:flex; justify-content:space-between; align-items:flex-start; padding:0.72rem 0; border-bottom:1px solid rgba(255,255,255,0.04); }
.arch-i:last-child { border-bottom:none; }
.arch-k { font-size:0.76rem; font-weight:700; color:#3e4460; text-transform:uppercase; letter-spacing:0.08em; min-width:130px; }
.arch-v { font-size:0.83rem; color:#7880a8; text-align:right; line-height:1.4; }

/* Model insights */
.ins-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.85rem; margin-top:0.5rem; }
.ins-card {
    background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05);
    border-radius:14px; padding:1.1rem 1.2rem; transition:border-color 0.2s;
}
.ins-card:hover { border-color:rgba(99,102,241,0.2); }
.ins-hd {
    font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.11em;
    color:#2c3050; margin-bottom:0.55rem;
    display:flex; align-items:center; gap:0.4rem;
}
.ins-hd::before { content:''; width:10px; height:2px; background:#5d60ef; border-radius:2px; }
.ins-body { font-size:0.82rem; color:#4a5278; line-height:1.65; }
.ins-hl { color:#8890d8; font-weight:600; }

/* Accuracy badge */
.acc-badge {
    display:inline-flex; align-items:center; gap:0.4rem;
    background:rgba(52,211,153,0.1); border:1px solid rgba(52,211,153,0.22);
    padding:0.2rem 0.65rem; border-radius:999px; font-size:0.75rem; font-weight:700; color:#34d399;
    margin-left:0.5rem;
}

/* Empty state */
.empty { text-align:center; padding:3rem 2rem; animation:fadeIn 0.38s ease; }
.empty-ico { font-size:2.5rem; margin-bottom:0.7rem; opacity:0.25; }
.empty-t { font-family:'Space Grotesk',sans-serif; font-size:1rem; font-weight:600; color:#2c3050; margin-bottom:0.35rem; }
.empty-s { font-size:0.81rem; color:#1e2236; }
</style>
""", unsafe_allow_html=True)


# DATA
# ========================
CLASS_NAMES = ['Cat', 'Cow', 'Deer', 'Dog', 'Goat', 'Hen', 'NightVision', 'Rabbit', 'Sheep']

ANIMAL_INFO = {
    "Cat": {
        "emoji": "🐱", "badge_class": "cbadge-dom", "conservation": "Domesticated",
        "overview": "The domestic cat (Felis catus) is one of the most widespread and beloved companion animals on Earth. Highly independent yet affectionate, cats have coexisted with humans for over 10,000 years.",
        "habitat": "Worldwide in domestic settings; feral populations on every inhabited continent.",
        "diet": "Obligate carnivore — requires meat-based protein; cannot synthesise taurine independently.",
        "lifespan": "12–18 years",
        "behavior": "Solitary and territorial by nature. Crepuscular — most active at dawn and dusk. Communicates through vocalizations (purring, chirping, meowing), body posture, and scent glands.",
        "fun_facts": ["Cats spend up to 70% of their lives sleeping — roughly 13–16 hours daily.", "A group of cats is called a clowder, and a group of kittens is a kindle.", "Cats can rotate each ear independently up to 180 degrees."]
    },
    "Cow": {
        "emoji": "🐄", "badge_class": "cbadge-dom", "conservation": "Domesticated",
        "overview": "Cattle (Bos taurus) are large domesticated ungulates raised for dairy, meat, and draft work. Among the first animals domesticated for agriculture roughly 10,000 years ago in the Fertile Crescent.",
        "habitat": "Domestic — global farmland, pastures, and grasslands.",
        "diet": "Herbivore. Primary diet of grasses, hay, silage, and grain supplements.",
        "lifespan": "15–25 years",
        "behavior": "Highly social herd animals with defined social hierarchies. Excellent long-term memory — can recognize up to 100 individual cows. Display visible emotional responses to stress and companionship.",
        "fun_facts": ["Cows have nearly 360-degree panoramic vision thanks to their wide-set eyes.", "A single dairy cow can produce approximately 200,000 glasses of milk in its lifetime.", "Cattle have four stomach compartments to efficiently ferment and digest tough plant fiber."]
    },
    "Deer": {
        "emoji": "🦌", "badge_class": "cbadge-lc", "conservation": "Least Concern",
        "overview": "Deer (family Cervidae) are graceful, hoofed mammals found across forests and grasslands on every continent except Antarctica and Australia. Over 90 species exist worldwide.",
        "habitat": "Temperate forests, grasslands, wetlands, and mountain meadows across North America, Europe, Asia, and South America.",
        "diet": "Herbivore — leaves, twigs, bark, berries, and grasses depending on season.",
        "lifespan": "10–20 years in the wild",
        "behavior": "Generally solitary or in small groups. Bucks become territorial during the rut (mating season). Highly alert with acute hearing and smell; will freeze or flee at perceived danger.",
        "fun_facts": ["Male deer grow a completely new set of antlers every single year.", "Deer can sprint at up to 30 mph and leap over 10 feet high.", "Newborn fawns are virtually odorless — a natural camouflage against predators."]
    },
    "Dog": {
        "emoji": "🐶", "badge_class": "cbadge-dom", "conservation": "Domesticated",
        "overview": "The domestic dog (Canis lupus familiaris) is the first animal ever domesticated by humans — over 15,000 years ago. Today over 340 recognized breeds exist, bred for companionship, herding, hunting, and assistance.",
        "habitat": "Worldwide alongside human settlements in all climates and environments.",
        "diet": "Omnivore — commercial kibble, raw protein, vegetables; dietary needs vary by size and breed.",
        "lifespan": "10–15 years (breed-dependent)",
        "behavior": "Pack animals with strong social bonds to humans and other animals. Highly trainable. Communicate through barking, whining, tail position, ear angle, and scent marking.",
        "fun_facts": ["A dog's sense of smell is 10,000–100,000 times more sensitive than a human's.", "A dog's nose print is as unique as a human fingerprint — used for identification.", "Dogs enter REM sleep and exhibit dreaming behavior, including twitching and eye movement."]
    },
    "Goat": {
        "emoji": "🐐", "badge_class": "cbadge-dom", "conservation": "Domesticated",
        "overview": "Domestic goats (Capra aegagrus hircus) were among the earliest livestock animals, first domesticated ~10,000 years ago in the Zagros Mountains. Valued globally for milk, meat, fiber (cashmere/mohair), and hides.",
        "habitat": "Farms, smallholdings, and mountainous terrain across the globe.",
        "diet": "Browser — prefers leaves, twigs, and shrubs over grass; remarkably adaptable to sparse vegetation.",
        "lifespan": "15–18 years",
        "behavior": "Curious, playful, and highly intelligent. Exceptional balance and climbing ability allows navigation of steep rocky terrain. Live in herds with well-defined dominance hierarchies maintained through head-butting.",
        "fun_facts": ["Goats have horizontal, rectangular pupils that give them an almost 340-degree field of vision.", "They are one of the very first animals domesticated for agricultural use — roughly 10,000 years ago.", "Goats can learn and recognize their own name, and respond to it when called."]
    },
    "Hen": {
        "emoji": "🐔", "badge_class": "cbadge-dom", "conservation": "Domesticated",
        "overview": "The domestic hen (Gallus gallus domesticus) is the world's most numerous bird — over 33 billion exist globally. Descended from the Red Junglefowl of Southeast Asia, domesticated roughly 8,000 years ago.",
        "habitat": "Domestic — poultry farms, backyard flocks, and free-range environments worldwide.",
        "diet": "Omnivore — seeds, grains, insects, worms, and kitchen scraps.",
        "lifespan": "5–10 years",
        "behavior": "Social flock animals governed by a strict pecking order. Communicate with over 30 distinct vocalizations for different situations — alarm calls, food calls, and conversational clucking. Hens cluck to their unhatched eggs.",
        "fun_facts": ["Chickens experience REM sleep cycles and exhibit dreaming behavior.", "They can run at speeds up to 9 mph when motivated.", "Chickens outnumber humans approximately 3-to-1 worldwide."]
    },
    "NightVision": {
        "emoji": "🌙", "badge_class": "cbadge-nt", "conservation": "Varies",
        "overview": "Images classified as NightVision are captured using infrared or low-light cameras — typically trail cameras or security systems — in dark conditions. The subject animal varies but exhibits nocturnal or crepuscular behavior.",
        "habitat": "Variable — forests, farmland, and suburban areas at night.",
        "diet": "Varies by the specific animal captured.",
        "lifespan": "Varies by species",
        "behavior": "Nocturnal animals active after dark have evolved specialized sensory systems including enlarged eyes with more rod photoreceptors, heightened hearing, and acute olfaction. Many produce minimal vocalizations to avoid detection.",
        "fun_facts": ["Many nocturnal animals possess a tapetum lucidum — a reflective eye layer that amplifies available light and causes eye-shine.", "Infrared cameras detect thermal radiation (heat), not visible light — rendering darkness irrelevant.", "Nocturnal lifestyle in many species evolved as a strategy to avoid daytime predators or competition."]
    },
    "Rabbit": {
        "emoji": "🐇", "badge_class": "cbadge-lc", "conservation": "Least Concern",
        "overview": "Rabbits (family Leporidae) are small herbivorous mammals found wild on every inhabited continent. Domestic rabbits (Oryctolagus cuniculus) are widely kept as pets and for meat and fur production.",
        "habitat": "Meadows, grasslands, forests, wetlands; domestic rabbits in homes worldwide.",
        "diet": "Herbivore — hay (70–80% of diet), leafy greens, vegetables, and limited pellets.",
        "lifespan": "8–12 years (domestic); 1–2 years (wild)",
        "behavior": "Social colony animals living in underground tunnel networks called warrens. Communicate through thumping, body posture, and ear positioning. Possess nearly 360-degree vision to detect aerial and ground predators simultaneously.",
        "fun_facts": ["Rabbits can leap up to 3 feet vertically and 9 feet horizontally in a single bound.", "A baby rabbit is called a kitten (or kit) — not a bunny.", "Rabbit incisors are open-rooted — they never stop growing and must be worn down through constant chewing."]
    },
    "Sheep": {
        "emoji": "🐑", "badge_class": "cbadge-dom", "conservation": "Domesticated",
        "overview": "Domestic sheep (Ovis aries) are ruminant mammals domesticated ~10,000 years ago in Mesopotamia. With over 1 billion individuals globally, they remain one of the most economically important livestock species — valued for wool, milk, and meat.",
        "habitat": "Domestic — grasslands, hillsides, and farms across all temperate regions.",
        "diet": "Herbivore — grasses, clovers, hay, and legumes.",
        "lifespan": "10–12 years",
        "behavior": "Highly gregarious flock animals that experience stress when isolated. Naturally follow a flock leader. Possess excellent long-term memory — capable of recognizing up to 50 individual sheep and several human faces for years.",
        "fun_facts": ["Sheep have horizontal, rectangular pupils providing a 320-degree field of vision without moving their head.", "They can recognize and remember individual human faces for up to two years.", "A sheep's wool never stops growing — an unshorn Merino can accumulate enormous fleece within a few years."]
    }
}

# ========================
# MODEL
# ========================
@st.cache_resource
def load_model():
    base = tf.keras.applications.MobileNetV2(input_shape=(180,180,3), include_top=False, weights=None)
    mdl = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(9, activation='softmax')
    ])
    mdl.load_weights("animal_weights.weights.h5")
    return mdl

def preprocess_image(image):
    img = image.resize((180,180)).convert("RGB")
    arr = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(arr, axis=0) / 255.0

def cc(c):
    return "high" if c >= 0.75 else ("medium" if c >= 0.60 else "low")

# ========================
# SIDEBAR
# ========================
with st.sidebar:
    st.markdown("""
    <div style="padding:0.3rem 0.8rem 1.6rem 0.8rem;">
        <div style="display:flex;align-items:center;gap:0.7rem;">
            <div style="width:34px;height:34px;border-radius:10px;background:rgba(93,96,239,0.14);
                border:1px solid rgba(93,96,239,0.24);display:flex;align-items:center;
                justify-content:center;font-size:1.1rem;">🐾</div>
            <div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:0.94rem;
                    font-weight:700;color:#e0e4ff;">My Project</div>
                <div style="font-size:0.67rem;color:#1e2236;margin-top:0.04rem;">AI Animal Classifier</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pages    = ["🏠  Home", "🔍  Detect", "📊  Dashboard", "📘  About"]
    page_map = {"🏠  Home":"Home","🔍  Detect":"Detect","📊  Dashboard":"Dashboard","📘  About":"About"}
    rev_map  = {v:k for k,v in page_map.items()}

    sel = st.radio("nav", pages, index=pages.index(rev_map[st.session_state.page]), label_visibility="collapsed")
    if page_map[sel] != st.session_state.page:
        st.session_state.page = page_map[sel]
        st.rerun()

    st.markdown("<div style='border-top:1px solid rgba(255,255,255,0.035);margin:0.4rem 0 1rem;'></div>", unsafe_allow_html=True)

    total = st.session_state.total_predictions
    last  = st.session_state.prediction_history[-1]["animal"] if st.session_state.prediction_history else "—"
    le    = ANIMAL_INFO.get(last,{}).get("emoji","") if last != "—" else ""

    st.markdown(f"""
    <div style="padding:0 0.8rem;">
        <div style="font-size:0.62rem;font-weight:700;text-transform:uppercase;
            letter-spacing:0.13em;color:#1e2236;margin-bottom:0.85rem;">Session</div>
        <div style="display:flex;justify-content:space-between;align-items:center;
            padding:0.45rem 0;border-bottom:1px solid rgba(255,255,255,0.035);">
            <span style="font-size:0.79rem;color:#3e4460;">Predictions</span>
            <span style="font-family:'Space Grotesk',sans-serif;font-size:0.9rem;font-weight:700;color:#7c83f0;">{total}</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;
            padding:0.45rem 0;border-bottom:1px solid rgba(255,255,255,0.035);">
            <span style="font-size:0.79rem;color:#3e4460;">Unique species</span>
            <span style="font-family:'Space Grotesk',sans-serif;font-size:0.9rem;font-weight:700;color:#7c83f0;">{len(st.session_state.animal_counts)}</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;padding:0.45rem 0;">
            <span style="font-size:0.79rem;color:#3e4460;">Last detected</span>
            <span style="font-size:0.8rem;font-weight:600;color:#7880a8;">{le} {last}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="position:fixed;bottom:1rem;left:0;width:17rem;text-align:center;">
        <span style="font-size:0.63rem;color:#151825;">Final Year BCA AI Project · 2024</span>
    </div>
    """, unsafe_allow_html=True)

# ========================
# HOME
# ========================
if st.session_state.page == "Home":
    st.markdown('<div class="pw">', unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <div class="hero-ey">AI-Powered Recognition</div>
        <h1 class="hero-t">Identify any animal<br>in seconds.</h1>
        <p class="hero-sub">
            My Project uses a fine-tuned MobileNetV2 model to classify 9 animal
            species from any photo or live webcam shot — delivering rich species profiles
            and session-based analytics alongside every prediction.
        </p>
        <div class="hero-pills">
            <span class="hero-pill"><span class="dot"></span>MobileNetV2</span>
            <span class="hero-pill"><span class="dot"></span>9 Species</span>
            <span class="hero-pill"><span class="dot"></span>~93% Accuracy</span>
            <span class="hero-pill"><span class="dot"></span>Species Profiles</span>
            <span class="hero-pill"><span class="dot"></span>Session Analytics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Start Detecting →", use_container_width=True, type="primary"):
            navigate("Detect")
    with c2:
        if st.button("View Dashboard", use_container_width=True):
            navigate("Dashboard")

    st.markdown('<div style="margin-top:1.8rem;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="feat-grid">
        <div class="fc"><div class="fc-icon">📸</div><div class="fc-t">Upload or Capture</div><div class="fc-d">JPG/PNG upload or live webcam — instant results from either.</div></div>
        <div class="fc"><div class="fc-icon">🧠</div><div class="fc-t">Deep Learning</div><div class="fc-d">MobileNetV2 backbone, fine-tuned for 9 species with softmax output.</div></div>
        <div class="fc"><div class="fc-icon">📊</div><div class="fc-t">Top-3 Predictions</div><div class="fc-d">Ranked confidence scores with colour-coded animated progress bars.</div></div>
        <div class="fc"><div class="fc-icon">🦎</div><div class="fc-t">Rich Species Profiles</div><div class="fc-d">Habitat, diet, lifespan, behavior, conservation status and fun facts.</div></div>
        <div class="fc"><div class="fc-icon">📈</div><div class="fc-t">Session Analytics</div><div class="fc-d">Detection counts, breakdowns, and a full timestamped history log.</div></div>
        <div class="fc"><div class="fc-icon">⚡</div><div class="fc-t">Fast & Lightweight</div><div class="fc-d">MobileNetV2 is CPU-optimised — predictions complete in under a second.</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin:0.2rem 0 0.8rem;border-top:1px solid rgba(255,255,255,0.04);"></div>', unsafe_allow_html=True)
    st.markdown('<div class="slbl">Supported species</div>', unsafe_allow_html=True)
    tags = "".join([f'<span class="atag">{ANIMAL_INFO[a]["emoji"]} {a}</span>' for a in CLASS_NAMES if a in ANIMAL_INFO])
    st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:0.42rem;">{tags}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# DETECT
# ========================
elif st.session_state.page == "Detect":
    st.markdown('<div class="pw">', unsafe_allow_html=True)

    st.markdown("""
    <div class="det-hdr">
        <div class="det-t">Animal Detection</div>
        <div class="det-s">Upload a photo or capture from your webcam — the model will identify the animal instantly.</div>
    </div>
    """, unsafe_allow_html=True)

    # If a result already exists, show "Analyze Another" flow
    if st.session_state.last_result:
        res = st.session_state.last_result
        best  = res["animal"]
        conf  = res["conf"]
        top3  = res["top3"]
        preds = res["preds"]
        level = cc(conf)
        emoji = ANIMAL_INFO.get(best, {}).get("emoji", "🐾")

        # Big result card
        st.markdown(f"""
        <div class="rc">
            <div class="rc-lbl">Detection Result</div>
            <div class="rc-emoji">{emoji}</div>
            <div class="rc-name">{best}</div>
            <div class="rc-conf-{level}">
                <span class="rc-dot rc-dot-{level}"></span>{conf*100:.1f}% confidence
            </div>
        </div>
        """, unsafe_allow_html=True)

        if conf < 0.60:
            st.markdown("""
            <div class="wbx">
                <span style="font-size:1rem;">⚠️</span>
                <span class="wbx-t">Low confidence — try a clearer, well-lit photo with the animal centred in frame.</span>
            </div>
            """, unsafe_allow_html=True)

        # Top-3
        st.markdown('<div class="pst">Top predictions</div>', unsafe_allow_html=True)
        for rank, idx in enumerate(top3):
            anim = CLASS_NAMES[idx]
            prob = float(preds[idx])
            st.markdown(f"""
            <div class="pi">
                <div class="pi-row">
                    <span class="{'pi-n1' if rank==0 else 'pi-n'}">{ANIMAL_INFO.get(anim,{}).get('emoji','🐾')} {anim}</span>
                    <span class="{'pi-p pi-p1' if rank==0 else 'pi-p'}">{prob*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(prob))

        st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
        if st.button("🔄  Analyze Another Image", use_container_width=True):
            reset_prediction()

        # — Species Profile Panel —
        if best in ANIMAL_INFO:
            info = ANIMAL_INFO[best]
            badge = f'<span class="cbadge {info["badge_class"]}">{info["conservation"]}</span>'
            facts_html = "".join([
                f'<div class="sp-fact"><div class="sp-fact-num">{i+1}</div><div class="sp-fact-text">{f}</div></div>'
                for i, f in enumerate(info["fun_facts"])
            ])
            st.markdown(f"""
            <div class="sp-wrap">
                <div class="sp-banner">
                    <div class="sp-bn-top">
                        <div class="sp-big-emoji">{info["emoji"]}</div>
                        <div>
                            <div class="sp-bn-name">{best}</div>
                            <div style="margin-top:0.3rem;">{badge}</div>
                        </div>
                    </div>
                    <div class="sp-overview">{info["overview"]}</div>
                </div>
                <div class="sp-stats">
                    <div class="sp-stat"><div class="sp-stat-lbl">Habitat</div><div class="sp-stat-val">{info["habitat"]}</div></div>
                    <div class="sp-stat"><div class="sp-stat-lbl">Diet</div><div class="sp-stat-val">{info["diet"]}</div></div>
                    <div class="sp-stat"><div class="sp-stat-lbl">Lifespan</div><div class="sp-stat-val">{info["lifespan"]}</div></div>
                    <div class="sp-stat"><div class="sp-stat-lbl">Conservation</div><div class="sp-stat-val">{badge}</div></div>
                </div>
                <div class="sp-body">
                    <div class="sp-sec">
                        <div class="sp-sec-title">Behavior</div>
                        <div class="sp-sec-text">{info["behavior"]}</div>
                    </div>
                    <div class="sp-sec">
                        <div class="sp-sec-title">Fun Facts</div>
                        <div class="sp-facts-row">{facts_html}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # No result yet — show input UI
        method = st.radio("Input method", ["📸  Upload Image", "📷  Use Webcam"], horizontal=True, label_visibility="collapsed")

        image = None

        if method == "📸  Upload Image":
            uploaded = st.file_uploader("Drop an image here or click to browse", type=["jpg","jpeg","png"], label_visibility="visible")
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
        else:
            # Mobile guidance
            st.markdown("""
            <div class="mob-notice">
                <span class="mob-notice-icon">📱</span>
                <div class="mob-notice-text">
                    <strong>Mobile users:</strong> tap the camera below to take a photo.
                    If your browser blocks camera access, switch to
                    <strong>Upload Image</strong> mode instead — it works on all devices.
                </div>
            </div>
            """, unsafe_allow_html=True)
            try:
                cam = st.camera_input("Tap to take a photo")
                if cam:
                    image = Image.open(cam).convert("RGB")
            except Exception:
                st.markdown("""
                <div class="mob-notice" style="background:rgba(239,68,68,0.08);border-color:rgba(239,68,68,0.18);">
                    <span class="mob-notice-icon">🚫</span>
                    <div class="mob-notice-text" style="color:#fca5a5;">
                        <strong style="color:#fca5a5;">Camera not accessible</strong> on this browser or device.
                        Please switch to <strong style="color:#fca5a5;">Upload Image</strong> mode.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        if image:
            col_img, col_act = st.columns([1, 1.1], gap="large")
            with col_img:
                st.image(image, caption="Input Image", use_container_width=True)
            with col_act:
                st.markdown('<div style="margin-bottom:0.5rem;"></div>', unsafe_allow_html=True)
                run = st.button("⚡  Analyze Image", use_container_width=True, type="primary")

                if run:
                    try:
                        model = load_model()
                    except Exception as e:
                        st.error(f"Model failed to load: {e}")
                        st.stop()

                    # Multi-step loading experience
                    steps = [
                        ("🖼️", "Processing image…"),
                        ("🔍", "Analyzing features…"),
                        ("🧠", "Generating prediction…"),
                    ]
                    prog_bar = st.progress(0)
                    step_slot = st.empty()

                    try:
                        arr = preprocess_image(image)
                        for i, (icon, label) in enumerate(steps[:-1]):
                            step_slot.markdown(f'<div class="load-step active"><span class="ls-icon">{icon}</span>{label}</div>', unsafe_allow_html=True)
                            prog_bar.progress((i + 1) * 28)
                            time.sleep(0.35)

                        step_slot.markdown(f'<div class="load-step active"><span class="ls-icon">🧠</span>Generating prediction…</div>', unsafe_allow_html=True)
                        prog_bar.progress(75)
                        preds = model.predict(arr, verbose=0)[0]
                        prog_bar.progress(100)
                        step_slot.markdown('<div class="load-step done"><span class="ls-icon">✅</span>Analysis complete</div>', unsafe_allow_html=True)
                        time.sleep(0.3)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        st.stop()

                    prog_bar.empty()
                    step_slot.empty()

                    top3 = preds.argsort()[-3:][::-1]
                    best = CLASS_NAMES[top3[0]]
                    conf = float(preds[top3[0]])

                    st.session_state.total_predictions += 1
                    st.session_state.prediction_history.append({
                        "animal": best, "confidence": conf,
                        "time": datetime.now().strftime("%H:%M:%S")
                    })
                    st.session_state.animal_counts[best] = st.session_state.animal_counts.get(best, 0) + 1
                    st.session_state.last_result = {"animal": best, "conf": conf, "top3": top3, "preds": preds}
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# DASHBOARD
# ========================
elif st.session_state.page == "Dashboard":
    st.markdown('<div class="pw">', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.55rem;font-weight:700;color:#fff;letter-spacing:-0.4px;">Session Dashboard</div>
        <div style="font-size:0.84rem;color:#2c3050;margin-top:0.18rem;">Your detection activity this session</div>
    </div>
    """, unsafe_allow_html=True)

    total  = st.session_state.total_predictions
    unique = len(st.session_state.animal_counts)
    last   = st.session_state.prediction_history[-1]["animal"] if st.session_state.prediction_history else "—"
    le     = ANIMAL_INFO.get(last,{}).get("emoji","") if last != "—" else ""

    st.markdown(f"""
    <div class="dsg">
        <div class="ds"><div class="ds-ico">🔬</div><div class="ds-num">{total}</div><div class="ds-lbl">Total detections</div></div>
        <div class="ds"><div class="ds-ico">🌿</div><div class="ds-num">{unique}</div><div class="ds-lbl">Unique species</div></div>
        <div class="ds"><div class="ds-ico">📍</div><div class="ds-num" style="font-size:1.25rem;">{le} {last}</div><div class="ds-lbl">Last detected</div></div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.animal_counts:
        sc = sorted(st.session_state.animal_counts.items(), key=lambda x: x[1], reverse=True)
        mx = sc[0][1]
        st.markdown('<div class="slbl">Breakdown by species</div>', unsafe_allow_html=True)
        st.markdown('<div class="dbd">', unsafe_allow_html=True)
        for animal, count in sc:
            em = ANIMAL_INFO.get(animal,{}).get("emoji","🐾")
            st.markdown(f'<div class="dbd-row"><span class="dbd-animal">{em} {animal}</span><span class="dbd-cnt">{count}×</span></div>', unsafe_allow_html=True)
            st.progress(float(count)/float(mx))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="slbl" style="margin-top:0.3rem;">Detection history</div>', unsafe_allow_html=True)

    if st.session_state.prediction_history:
        st.markdown('<div class="dh">', unsafe_allow_html=True)
        for i, r in enumerate(reversed(st.session_state.prediction_history), 1):
            lvl = cc(r["confidence"])
            em  = ANIMAL_INFO.get(r["animal"],{}).get("emoji","🐾")
            st.markdown(f"""
            <div class="hr-row">
                <span class="hr-n">#{i}</span>
                <span class="hr-a">{em} {r['animal']}</span>
                <span class="hr-c-{lvl}">{r['confidence']*100:.1f}%</span>
                <span class="hr-t">{r['time']}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-top:0.9rem;"></div>', unsafe_allow_html=True)
        if st.button("Clear session data", type="secondary"):
            st.session_state.prediction_history = []
            st.session_state.total_predictions  = 0
            st.session_state.animal_counts = {}
            st.session_state.last_result   = None
            st.rerun()
    else:
        st.markdown("""
        <div class="empty">
            <div class="empty-ico">📋</div>
            <div class="empty-t">No detections yet</div>
            <div class="empty-s">Run a detection to see analytics here.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Detect →", type="primary"):
            navigate("Detect")

    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# ABOUT
# ========================
elif st.session_state.page == "About":
    st.markdown('<div class="pw">', unsafe_allow_html=True)

    st.markdown("""
    <div class="ab-hero">
        <div class="ab-t">About Ai Image Classifier</div>
        <div class="ab-b">
            Ai Image Classifier is a final-year BCA AI project — an animal classification system
            powered by a fine-tuned MobileNetV2 convolutional neural network. It classifies
            9 animal species from user-uploaded photos or live webcam captures, and pairs
            every prediction with detailed species profiles, confidence analytics, and a
            session-based history dashboard.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # — Model Insights section —
    st.markdown('<div class="slbl">Model insights</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ins-grid">
        <div class="ins-card">
            <div class="ins-hd">Why MobileNetV2</div>
            <div class="ins-body">
                MobileNetV2 uses <span class="ins-hl">inverted residuals</span> and
                <span class="ins-hl">linear bottlenecks</span> to achieve high accuracy with
                dramatically fewer parameters than VGG or ResNet variants.
                It runs efficiently on CPU — no GPU required — making it ideal for
                deployment in lightweight web applications like this one.
            </div>
        </div>
        <div class="ins-card">
            <div class="ins-hd">Transfer Learning</div>
            <div class="ins-body">
                The base MobileNetV2 was pre-trained on ImageNet (1.4M images, 1000 classes),
                giving it strong low-level and mid-level visual features.
                The final layers were replaced and <span class="ins-hl">fine-tuned</span> on
                the 9-class animal dataset — a technique that converges faster and requires
                far less data than training from scratch.
            </div>
        </div>
        <div class="ins-card">
            <div class="ins-hd">Model Performance <span class="acc-badge">~93% accuracy</span></div>
            <div class="ins-body">
                The fine-tuned model achieves approximately
                <span class="ins-hl">93% top-1 classification accuracy</span>
                on the held-out test set across all 9 classes.
                Confidence below 60% triggers a low-confidence warning —
                results are most reliable with clear, well-lit, single-subject images.
            </div>
        </div>
        <div class="ins-card">
            <div class="ins-hd">Dataset</div>
            <div class="ins-body">
                Trained on a curated dataset of
                <span class="ins-hl">9 animal categories</span>:
                Cat, Cow, Deer, Dog, Goat, Hen, NightVision, Rabbit, and Sheep.
                Images sourced from farm settings, wildlife photography, and
                infrared trail-camera footage, then preprocessed to
                <span class="ins-hl">180×180 RGB</span> and normalized to [0,1].
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)

    c_tech, c_arch = st.columns([1, 1], gap="large")

    with c_tech:
        st.markdown('<div class="slbl">Tech stack</div>', unsafe_allow_html=True)
        st.markdown('<div class="gc">', unsafe_allow_html=True)
        for icon, name, desc in [
            ("🧠","TensorFlow 2.x","Deep learning framework"),
            ("🏗️","Keras","Neural network API layer"),
            ("📊","Streamlit","Web application framework"),
            ("🖼️","Pillow (PIL)","Image loading and resizing"),
            ("🔢","NumPy","Numerical array operations"),
            ("🐍","Python 3.11","Core language runtime"),
        ]:
            st.markdown(f'<div class="tech-i"><div class="tech-ico">{icon}</div><div><div class="tech-n">{name}</div><div class="tech-d">{desc}</div></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_arch:
        st.markdown('<div class="slbl">Model architecture</div>', unsafe_allow_html=True)
        st.markdown('<div class="gc">', unsafe_allow_html=True)
        for k, v in [
            ("Base model","MobileNetV2 (weights=None)"),
            ("Input shape","180 × 180 × 3 (RGB)"),
            ("Pooling","GlobalAveragePooling2D"),
            ("Dense layer","128 units · ReLU"),
            ("Regularization","Dropout (p=0.2)"),
            ("Output","9-class Softmax"),
            ("Weights file","animal_weights.weights.h5"),
            ("Top-1 accuracy","≈ 93% (test set)"),
        ]:
            st.markdown(f'<div class="arch-i"><span class="arch-k">{k}</span><span class="arch-v">{v}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:2rem 0 0.5rem;font-size:0.72rem;color:#151825;">
         &middot; Final Year BCA AI Project &middot; Built with TensorFlow + Streamlit
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
