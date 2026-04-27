import streamlit as st
import anthropic
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import base64
import io
import json
import math

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisaIA · Análise de Rosto & Cortes",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --cream: #F5F0E8;
    --charcoal: #1A1A1A;
    --gold: #C9A84C;
    --muted: #7A7065;
    --card-bg: #FDFAF4;
    --border: #E0D9CC;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--cream) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }

h1, h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
}

.main-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.8rem;
    font-weight: 300;
    color: var(--charcoal);
    letter-spacing: 0.02em;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.main-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
    font-weight: 300;
}

.gold-divider {
    width: 60px;
    height: 2px;
    background: var(--gold);
    margin: 1rem 0 2rem 0;
}

.face-shape-badge {
    display: inline-block;
    background: var(--charcoal);
    color: var(--gold);
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-weight: 600;
    padding: 0.6rem 1.8rem;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
    border-radius: 2px;
}

.analysis-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
}

.analysis-card h4 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.2rem;
    color: var(--charcoal);
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.analysis-card p {
    font-size: 0.88rem;
    color: var(--muted);
    line-height: 1.7;
    margin: 0;
}

.haircut-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    transition: box-shadow 0.25s ease;
}

.haircut-card:hover {
    box-shadow: 0 8px 32px rgba(0,0,0,0.10);
}

.haircut-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--charcoal);
    margin-bottom: 0.3rem;
}

.haircut-why {
    font-size: 0.83rem;
    color: var(--muted);
    line-height: 1.6;
}

.landmark-info {
    font-size: 0.78rem;
    color: var(--muted);
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.05em;
}

.step-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    font-weight: 500;
    margin-bottom: 0.5rem;
}

.tip-box {
    background: linear-gradient(135deg, #1A1A1A 0%, #2D2D2D 100%);
    border-left: 3px solid var(--gold);
    border-radius: 2px;
    padding: 1.2rem 1.5rem;
    color: #E8E0D0;
    font-size: 0.88rem;
    line-height: 1.7;
}

.tip-box strong {
    color: var(--gold);
    font-family: 'Cormorant Garamond', serif;
    font-size: 1rem;
}

.stButton > button {
    background: var(--charcoal) !important;
    color: var(--gold) !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 2rem !important;
    width: 100%;
    transition: background 0.2s ease !important;
}

.stButton > button:hover {
    background: #2D2D2D !important;
}

[data-testid="stFileUploader"] {
    background: var(--card-bg) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 4px !important;
}

.ratio-bar-label {
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 2px;
}

.stProgress > div > div > div {
    background: var(--gold) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── MediaPipe Setup ───────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Key landmark indices for face shape analysis
LANDMARKS = {
    "top":          10,   # Forehead top
    "bottom":       152,  # Chin
    "left":         234,  # Left cheek
    "right":        454,  # Right cheek
    "jaw_left":     172,  # Left jaw
    "jaw_right":    397,  # Right jaw
    "cheek_left":   116,  # Left cheekbone
    "cheek_right":  345,  # Right cheekbone
    "temple_left":  70,   # Left temple
    "temple_right": 300,  # Right temple
    "forehead_l":   63,   # Forehead left
    "forehead_r":   293,  # Forehead right
}

HAIRCUT_IMAGES = {
    "Oval": [
        ("Undercut Clássico",      "https://images.unsplash.com/photo-1621605815971-fbc98d665033?w=400&q=80"),
        ("Fade + Textura no Topo", "https://images.unsplash.com/photo-1605497788044-5a32c7078486?w=400&q=80"),
        ("Quiff Moderno",          "https://images.unsplash.com/photo-1599351431202-1e0f0137899a?w=400&q=80"),
        ("Slick Back",             "https://images.unsplash.com/photo-1503951914875-452162b0f3f1?w=400&q=80"),
    ],
    "Redondo": [
        ("Topete Alto (Pompadour)", "https://images.unsplash.com/photo-1622286342621-4bd786c2447c?w=400&q=80"),
        ("Moicano Suave",           "https://images.unsplash.com/photo-1503951914875-452162b0f3f1?w=400&q=80"),
        ("Undercut Lateral",        "https://images.unsplash.com/photo-1621605815971-fbc98d665033?w=400&q=80"),
        ("Franja para o Lado",      "https://images.unsplash.com/photo-1605497788044-5a32c7078486?w=400&q=80"),
    ],
    "Quadrado": [
        ("Buzz Cut",          "https://images.unsplash.com/photo-1599351431202-1e0f0137899a?w=400&q=80"),
        ("Fade Baixo",        "https://images.unsplash.com/photo-1622286342621-4bd786c2447c?w=400&q=80"),
        ("Textura Soft",      "https://images.unsplash.com/photo-1503951914875-452162b0f3f1?w=400&q=80"),
        ("Crop + Franja",     "https://images.unsplash.com/photo-1621605815971-fbc98d665033?w=400&q=80"),
    ],
    "Coração": [
        ("Curtinho nas Laterais", "https://images.unsplash.com/photo-1605497788044-5a32c7078486?w=400&q=80"),
        ("Side Part Clássico",    "https://images.unsplash.com/photo-1599351431202-1e0f0137899a?w=400&q=80"),
        ("Franja Pesada",         "https://images.unsplash.com/photo-1503951914875-452162b0f3f1?w=400&q=80"),
        ("Wavy Medium Length",    "https://images.unsplash.com/photo-1622286342621-4bd786c2447c?w=400&q=80"),
    ],
    "Oblongo": [
        ("Fade + Volume Lateral", "https://images.unsplash.com/photo-1621605815971-fbc98d665033?w=400&q=80"),
        ("Franja Reta",           "https://images.unsplash.com/photo-1605497788044-5a32c7078486?w=400&q=80"),
        ("Crop Texturizado",      "https://images.unsplash.com/photo-1622286342621-4bd786c2447c?w=400&q=80"),
        ("Bob Masculino",         "https://images.unsplash.com/photo-1599351431202-1e0f0137899a?w=400&q=80"),
    ],
    "Diamante": [
        ("Fringe + Textura",   "https://images.unsplash.com/photo-1503951914875-452162b0f3f1?w=400&q=80"),
        ("Quiff Leve",         "https://images.unsplash.com/photo-1621605815971-fbc98d665033?w=400&q=80"),
        ("Side Swept Fringe",  "https://images.unsplash.com/photo-1622286342621-4bd786c2447c?w=400&q=80"),
        ("Curtinho Clássico",  "https://images.unsplash.com/photo-1605497788044-5a32c7078486?w=400&q=80"),
    ],
}


# ─── Helper Functions ──────────────────────────────────────────────────────────

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def image_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def draw_face_mesh(img_bgr: np.ndarray, results) -> np.ndarray:
    annotated = img_bgr.copy()
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
    return annotated


def compute_face_ratios(landmarks, w: int, h: int) -> dict:
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    top        = pt(LANDMARKS["top"])
    bottom     = pt(LANDMARKS["bottom"])
    left_c     = pt(LANDMARKS["left"])
    right_c    = pt(LANDMARKS["right"])
    jaw_l      = pt(LANDMARKS["jaw_left"])
    jaw_r      = pt(LANDMARKS["jaw_right"])
    cheek_l    = pt(LANDMARKS["cheek_left"])
    cheek_r    = pt(LANDMARKS["cheek_right"])
    fore_l     = pt(LANDMARKS["forehead_l"])
    fore_r     = pt(LANDMARKS["forehead_r"])

    face_height     = np.linalg.norm(bottom - top)
    face_width      = np.linalg.norm(right_c - left_c)
    jaw_width       = np.linalg.norm(jaw_r - jaw_l)
    cheekbone_width = np.linalg.norm(cheek_r - cheek_l)
    forehead_width  = np.linalg.norm(fore_r - fore_l)

    return {
        "face_height":     round(float(face_height), 1),
        "face_width":      round(float(face_width), 1),
        "jaw_width":       round(float(jaw_width), 1),
        "cheekbone_width": round(float(cheekbone_width), 1),
        "forehead_width":  round(float(forehead_width), 1),
        "ratio_h_w":       round(float(face_height / max(face_width, 1)), 3),
        "ratio_jaw_cheek": round(float(jaw_width / max(cheekbone_width, 1)), 3),
        "ratio_fore_jaw":  round(float(forehead_width / max(jaw_width, 1)), 3),
    }


def detect_face(pil_img: Image.Image):
    img_bgr = pil_to_cv2(pil_img)
    h, w = img_bgr.shape[:2]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return None, None, None

    landmarks = results.multi_face_landmarks[0].landmark
    ratios    = compute_face_ratios(landmarks, w, h)
    annotated = draw_face_mesh(img_bgr, results)
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    return annotated_pil, ratios, landmarks


def analyze_with_claude(pil_img: Image.Image, ratios: dict) -> dict:
    client = anthropic.Anthropic()
    b64    = image_to_b64(pil_img)

    system_prompt = """Você é um especialista em visagismo e análise facial.
Analise a imagem e os dados de proporções faciais fornecidos.
Retorne APENAS um JSON válido (sem markdown, sem texto extra) com esta estrutura:
{
  "formato_rosto": "Oval|Redondo|Quadrado|Coração|Oblongo|Diamante",
  "confianca": 0.0-1.0,
  "descricao_formato": "Descrição curta do formato (1-2 frases)",
  "caracteristicas_detectadas": ["característica 1", "característica 2", "característica 3"],
  "dica_principal": "Dica principal de visagismo (2-3 frases)",
  "estilos_recomendados": [
    {"nome": "Nome do corte", "justificativa": "Por que funciona para este formato"},
    {"nome": "Nome do corte 2", "justificativa": "Por que funciona"},
    {"nome": "Nome do corte 3", "justificativa": "Por que funciona"},
    {"nome": "Nome do corte 4", "justificativa": "Por que funciona"}
  ],
  "evitar": ["Estilo que deve evitar 1", "Estilo que deve evitar 2"]
}"""

    user_content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        },
        {
            "type": "text",
            "text": f"""Analise este rosto com base na imagem e nas proporções medidas pelo MediaPipe:

Proporções calculadas:
- Altura do rosto: {ratios['face_height']:.0f}px
- Largura do rosto: {ratios['face_width']:.0f}px
- Largura da mandíbula: {ratios['jaw_width']:.0f}px
- Largura dos maçãs: {ratios['cheekbone_width']:.0f}px
- Largura da testa: {ratios['forehead_width']:.0f}px
- Proporção altura/largura: {ratios['ratio_h_w']:.2f}
- Proporção mandíbula/maçãs: {ratios['ratio_jaw_cheek']:.2f}
- Proporção testa/mandíbula: {ratios['ratio_fore_jaw']:.2f}

Com base na imagem e nessas métricas, forneça a análise de visagismo completa.""",
        },
    ]

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1200,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ─── App Layout ───────────────────────────────────────────────────────────────

col_header, _ = st.columns([3, 1])
with col_header:
    st.markdown('<div class="main-title">VisaIA</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Análise Facial · Visagismo · Recomendação de Cortes</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

# ─── Upload Section ────────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 2], gap="large")

with col_upload:
    st.markdown('<div class="step-label">① Envie uma foto</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Foto do cliente",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        pil_img = Image.open(uploaded_file).convert("RGB")
        # Resize for performance
        max_dim = 800
        if max(pil_img.size) > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.LANCZOS)

        st.image(pil_img, caption="Foto enviada", use_container_width=True)

        st.markdown('<div class="step-label" style="margin-top:1.2rem;">② Análise</div>', unsafe_allow_html=True)
        run_btn = st.button("✦  Analisar Rosto com IA")

        st.markdown("""
        <div class="tip-box">
            <strong>📸 Dicas para melhor resultado</strong><br>
            Use foto frontal com boa iluminação.<br>
            Evite acessórios que cubram o rosto.<br>
            Rosto centralizado na imagem.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#7A7065;">
            <div style="font-size:3rem;margin-bottom:1rem;">📷</div>
            <div style="font-family:'Cormorant Garamond',serif;font-size:1.3rem;margin-bottom:0.5rem;">
                Aguardando foto
            </div>
            <div style="font-size:0.85rem;line-height:1.6;">
                Arraste ou clique para<br>enviar uma foto frontal do cliente
            </div>
        </div>
        """, unsafe_allow_html=True)
        run_btn = False

# ─── Analysis Results ──────────────────────────────────────────────────────────
with col_result:
    if uploaded_file and run_btn:

        with st.spinner("🔍 Detectando pontos faciais..."):
            annotated_img, ratios, landmarks = detect_face(pil_img)

        if annotated_img is None:
            st.error("❌ Nenhum rosto detectado. Tente uma foto mais clara e frontal.")
        else:
            with st.spinner("🤖 Analisando com IA · Visagismo..."):
                try:
                    analysis = analyze_with_claude(pil_img, ratios)
                except Exception as e:
                    st.error(f"Erro na análise IA: {e}")
                    st.stop()

            # ── Face Shape Badge ──
            fmt = analysis.get("formato_rosto", "Oval")
            conf = int(analysis.get("confianca", 0.85) * 100)

            st.markdown(f'<div class="face-shape-badge">Formato {fmt}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="landmark-info">Confiança da IA · {conf}% &nbsp;|&nbsp; '
                        f'Proporção A/L · {ratios["ratio_h_w"]:.2f} &nbsp;|&nbsp; '
                        f'Mandíbula/Maçãs · {ratios["ratio_jaw_cheek"]:.2f}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Two-column layout: annotated image + analysis cards ──
            img_col, info_col = st.columns([1, 1], gap="medium")

            with img_col:
                st.markdown('<div class="step-label">Mapeamento Facial</div>', unsafe_allow_html=True)
                st.image(annotated_img, use_container_width=True)

                # Proportion bars
                st.markdown('<div class="step-label" style="margin-top:1rem;">Proporções</div>', unsafe_allow_html=True)
                props = [
                    ("Altura / Largura", min(ratios["ratio_h_w"] / 1.8, 1.0)),
                    ("Mandíbula / Maçãs", min(ratios["ratio_jaw_cheek"], 1.0)),
                    ("Testa / Mandíbula",  min(ratios["ratio_fore_jaw"], 1.0)),
                ]
                for label, val in props:
                    st.markdown(f'<div class="ratio-bar-label">{label} — {val:.0%}</div>', unsafe_allow_html=True)
                    st.progress(float(val))

            with info_col:
                st.markdown('<div class="step-label">Análise</div>', unsafe_allow_html=True)

                desc = analysis.get("descricao_formato", "")
                st.markdown(f"""
                <div class="analysis-card">
                    <h4>Características do Formato {fmt}</h4>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)

                chars = analysis.get("caracteristicas_detectadas", [])
                if chars:
                    bullets = "".join(f"<li>{c}</li>" for c in chars)
                    st.markdown(f"""
                    <div class="analysis-card">
                        <h4>Características Detectadas</h4>
                        <p><ul style="margin:0;padding-left:1.1rem;color:#7A7065;font-size:0.85rem;line-height:1.8;">{bullets}</ul></p>
                    </div>
                    """, unsafe_allow_html=True)

                dica = analysis.get("dica_principal", "")
                if dica:
                    st.markdown(f"""
                    <div class="tip-box">
                        <strong>✦ Dica de Visagismo</strong><br>{dica}
                    </div>
                    """, unsafe_allow_html=True)

                evitar = analysis.get("evitar", [])
                if evitar:
                    ev_text = " · ".join(evitar)
                    st.markdown(f"""
                    <div class="analysis-card" style="border-left:3px solid #C9A84C;margin-top:0.8rem;">
                        <h4>⚠ Evitar</h4>
                        <p>{ev_text}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Haircut Recommendations ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="step-label">③ Recomendações de Cortes</div>', unsafe_allow_html=True)
            st.markdown(f"<div style='font-family:Cormorant Garamond,serif;font-size:1.6rem;font-weight:300;color:#1A1A1A;margin-bottom:1.2rem;'>Cortes ideais para rosto <em>{fmt}</em></div>", unsafe_allow_html=True)

            ai_styles = analysis.get("estilos_recomendados", [])
            fallback_images = HAIRCUT_IMAGES.get(fmt, HAIRCUT_IMAGES["Oval"])

            rec_cols = st.columns(4, gap="small")
            for i, col in enumerate(rec_cols):
                with col:
                    # Get image URL
                    img_name, img_url = fallback_images[i % len(fallback_images)]
                    # Get AI justification if available
                    if i < len(ai_styles):
                        cut_name = ai_styles[i].get("nome", img_name)
                        justif   = ai_styles[i].get("justificativa", "")
                    else:
                        cut_name = img_name
                        justif   = ""

                    st.markdown(f"""
                    <div class="haircut-card">
                        <img src="{img_url}" style="width:100%;aspect-ratio:3/4;object-fit:cover;display:block;" />
                        <div style="padding:0.9rem;">
                            <div class="haircut-title">{cut_name}</div>
                            <div class="haircut-why">{justif}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

    elif not uploaded_file:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:center;height:400px;">
            <div style="text-align:center;color:#7A7065;">
                <div style="font-family:'Cormorant Garamond',serif;font-size:2rem;font-weight:300;margin-bottom:0.5rem;">
                    Resultados aparecerão aqui
                </div>
                <div style="font-size:0.9rem;letter-spacing:0.1em;">
                    ENVIE UMA FOTO PARA COMEÇAR
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><hr style='border:none;border-top:1px solid #E0D9CC;margin:2rem 0 1rem;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#B0A89A;font-size:0.78rem;letter-spacing:0.1em;padding-bottom:1rem;">
    VISAIA · POWERED BY CLAUDE AI + MEDIAPIPE · ANÁLISE FACIAL & VISAGISMO
</div>
""", unsafe_allow_html=True)
