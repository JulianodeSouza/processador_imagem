✨ Funcionalidades
Detecção de 478 pontos faciais via MediaPipe Face Mesh
Análise por IA (Claude Vision) para identificar o formato do rosto
6 formatos suportados: Oval, Redondo, Quadrado, Coração, Oblongo, Diamante
Recomendações personalizadas com justificativa de visagismo
Interface elegante estilo barbearia premium
Métricas faciais: proporção altura/largura, mandíbula/maçãs, testa/mandíbula
🚀 Como Executar
1. Instalar dependências
pip install -r requirements.txt
2. Configurar chave da API Anthropic
# Linux / macOS
export ANTHROPIC_API_KEY="sua-chave-aqui"

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sua-chave-aqui"
Obtenha sua chave em: https://console.anthropic.com

3. Executar o app
streamlit run app.py
O app abrirá automaticamente em http://localhost:8501

🔬 Como Funciona
📷 Foto do cliente
        ↓
🧠 MediaPipe Face Mesh
   (478 pontos faciais)
        ↓
📐 Cálculo de proporções
   - Razão altura/largura
   - Largura mandíbula vs maçãs
   - Largura testa vs mandíbula
        ↓
🤖 Claude Vision API
   - Análise visual + métricas
   - Classificação do formato
   - Recomendações de cortes
        ↓
✂️ Exibição dos resultados
   - Mapa de pontos faciais
   - Formato identificado
   - 4 cortes recomendados com imagens
   - Dicas de visagismo
📋 Formatos de Rosto Detectados
Formato	Características
Oval	Proporcional, testa ligeiramente mais larga
Redondo	Largura ≈ Altura, bochechas cheias
Quadrado	Mandíbula forte, testa e maçãs similares
Coração	Testa larga, queixo estreito/pontudo
Oblongo	Rosto longo, largura uniforme
Diamante	Maçãs proeminentes, testa e queixo estreitos
📁 Estrutura
visagismo_app/
├── app.py           # Aplicação principal Streamlit
├── requirements.txt # Dependências Python
└── README.md        # Este arquivo
💡 Dicas para Melhores Resultados
Use foto frontal com rosto centralizado
Boa iluminação, preferencialmente natural
Sem óculos, chapéu ou cabelo cobrindo o rosto
Resolução mínima recomendada: 400x400px
