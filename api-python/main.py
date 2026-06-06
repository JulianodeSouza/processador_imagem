import io
import json
import os
import urllib.request

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
import mediapipe as mp
import numpy as np
from PIL import Image

# ─── Download automático do modelo do MediaPipe (só na primeira execução) ──────
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Baixando modelo do MediaPipe pela primeira vez (~30MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH
    )
    print("Modelo baixado com sucesso!")

app = FastAPI(
    title="VisaIA REST API",
    description="API REST para análise facial de visagismo e recomendação de cortes usando MediaPipe e Gemini AI",
    version="1.0.0"
)

# Permite que aplicações frontend (como React, Angular ou Vue) consultem a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Índices dos landmarks faciais ────────────────────────────────────────────
LANDMARKS = {
    "top":          10,
    "bottom":       152,
    "left":         234,
    "right":        454,
    "jaw_left":     172,
    "jaw_right":    397,
    "cheek_left":   116,
    "cheek_right":  345,
    "forehead_l":   63,
    "forehead_r":   293,
}

def compute_face_ratios(landmarks, w: int, h: int) -> dict:
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    top     = pt(LANDMARKS["top"])
    bottom  = pt(LANDMARKS["bottom"])
    left_c  = pt(LANDMARKS["left"])
    right_c = pt(LANDMARKS["right"])
    jaw_l   = pt(LANDMARKS["jaw_left"])
    jaw_r   = pt(LANDMARKS["jaw_right"])
    cheek_l = pt(LANDMARKS["cheek_left"])
    cheek_r = pt(LANDMARKS["cheek_right"])
    fore_l  = pt(LANDMARKS["forehead_l"])
    fore_r  = pt(LANDMARKS["forehead_r"])

    face_height     = np.linalg.norm(bottom - top)
    face_width      = np.linalg.norm(right_c - left_c)
    jaw_width       = np.linalg.norm(jaw_r - jaw_l)
    cheekbone_width = np.linalg.norm(cheek_r - cheek_l)
    forehead_width  = np.linalg.norm(fore_r - fore_l)

    return {
        "face_height_px":     round(float(face_height), 1),
        "face_width_px":      round(float(face_width), 1),
        "ratio_height_width": round(float(face_height / max(face_width, 1)), 3),
        "ratio_jaw_cheek":    round(float(jaw_width / max(cheekbone_width, 1)), 3),
        "ratio_forehead_jaw": round(float(forehead_width / max(jaw_width, 1)), 3),
    }

def analyze_with_gemini(ratios: dict) -> dict:
    client = genai.Client(api_key="sua-chave-aqui")

    system_prompt = """Você é um especialista em visagismo e análise facial.
Analise os dados de proporções faciais fornecidos.
Retorne um objeto JSON estrito com esta exata estrutura:
{
  "formato_rosto": "Oval|Redondo|Quadrado|Coração|Oblongo|Diamante",
  "confianca": 0.85,
  "descricao_formato": "Descrição curta do formato (1-2 frases)",
  "caracteristicas_detectadas": ["característica 1", "característica 2"],
  "dica_principal": "Dica principal de visagismo",
  "estilos_recomendados": [
    {"nome": "Nome do corte", "justificativa": "Por que funciona"},
    {"nome": "Nome do corte 2", "justificativa": "Por que funciona"},
    {"nome": "Nome do corte 3", "justificativa": "Por que funciona"},
    {"nome": "Nome do corte 4", "justificativa": "Por que funciona"}
  ],
  "evitar": ["Estilo que deve evitar 1", "Estilo que deve evitar 2"]
}"""

    user_text = f"""Analise este rosto com base nas proporções medidas pelo MediaPipe:
Proporções calculadas:
- Proporção altura/largura: {ratios['ratio_height_width']:.2f}
- Proporção mandíbula/maçãs: {ratios['ratio_jaw_cheek']:.2f}
- Proporção testa/mandíbula: {ratios['ratio_forehead_jaw']:.2f}
"""

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[user_text],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            temperature=0.2
        )
    )

    response_text = getattr(response, "text", None)
    if response_text is None and hasattr(response, "response"):
        response_text = getattr(response.response, "text", None)
    if response_text is None:
        raise ValueError("Resposta do Gemini não contém texto JSON esperado.")

    return json.loads(response_text.strip())

# ─── Endpoints da API REST ─────────────────────────────────────────────────────

@app.get("/", tags=["Geral"])
async def root():
    """Endpoint de checagem de saúde da API."""
    return {"status": "online", "api": "API Visagismo", "versao": "1.0.0"}

@app.post("/analisar", tags=["Análise Facial"])
async def analisar_rosto(file: UploadFile = File(...)):
    """
    Recebe uma imagem via Form-Data (chave 'file'), extrai os pontos faciais
    matemáticos e processa o diagnóstico de visagismo no Gemini AI.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo enviado precisa ser uma imagem válida (PNG/JPG).")

    try:
        # 1. Ler e preparar a imagem
        image_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if max(pil_img.size) > 800:
            pil_img.thumbnail((800, 800), Image.LANCZOS)

        img_rgb = np.array(pil_img)
        h, w = img_rgb.shape[:2]

        # 2. Detectar landmarks com a API do MediaPipe
        base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5
        )

        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = detector.detect(mp_image)

        if not results.face_landmarks:
            raise HTTPException(status_code=422, detail="Nenhum rosto humano pôde ser detectado na imagem enviada.")

        # 3. Extrair métricas matemáticas
        landmarks = results.face_landmarks[0]
        ratios = compute_face_ratios(landmarks, w, h)

        # 4. Enviar métricas para o Gemini
        diagnostico_ia = analyze_with_gemini(ratios)

        # 5. Retornar resposta estruturada
        return {
            "sucesso": True,
            "metricas_computadas": ratios,
            "analise_visagismo": diagnostico_ia
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno durante o processamento: {str(e)}")
