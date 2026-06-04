# 💈 VisaIA — Identificador de Formato de Rosto e Recomendações de Corte (Visagismo)

Um aplicativo desenvolvido em Python utilizando **FastAPI**, focado em **Visagismo**. Ele utiliza visão computacional e Inteligência Artificial para analisar a foto do usuário, identificar o formato de seu rosto com precisão matemática e sugerir os melhores cortes de cabelo de acordo com técnicas profissionais de visagismo.

---

## ✨ Funcionalidades do Projeto

- **Detecção Facial de Alta Precisão**: Realiza o mapeamento de 478 pontos faciais utilizando o modelo *MediaPipe Face Landmarker*.
- **Extração de Métricas Faciais**: Calcula matematicamente as proporções do rosto (razão altura/largura, mandíbula vs maçãs do rosto, testa vs mandíbula).
- **Análise por Inteligência Artificial**: Integração com a API do **Google Gemini** para processar as métricas extraídas, garantindo uma identificação precisa do formato do rosto.
- **6 Formatos de Rosto Suportados**: Capaz de classificar o rosto como Oval, Redondo, Quadrado, Coração, Oblongo ou Diamante.
- **Recomendações Personalizadas**: Sugere 4 opções de corte de cabelo ideais para o usuário, acompanhadas de justificativas com base no formato do rosto.
- **API REST**: Interface RESTful construída com FastAPI, com documentação automática via Swagger.

---

## 🚀 Como Executar o Projeto Localmente

Para rodar este projeto em sua própria máquina, siga os passos detalhados abaixo:

### ⚙️ Pré-requisitos
- Ter o [Python 3.11](https://www.python.org/downloads/) instalado.
- Ter uma chave de API do Gemini, gerada gratuitamente no [Google AI Studio](https://aistudio.google.com).

### 🛠️ Passo a Passo

**1. Clone o repositório e acesse a pasta**
```bash
git clone https://github.com/seu-usuario/processador_imagem.git
cd processador_imagem
```

**2. Crie um Ambiente Virtual com Python 3.11**
```bash
# Criar o ambiente virtual
py -3.11 -m venv venv

# Ativar no Windows:
venv\Scripts\activate

# Ativar no Linux / macOS:
source venv/bin/activate
```

**3. Instale as dependências do projeto**
```bash
pip install -r requirements.txt
```

**4. Configure a chave da API do Google Gemini**

Após gerar a chave no [Google AI Studio](https://aistudio.google.com), o método mais simples e confiável é inserir a chave diretamente no código.

Abra o `main.py` e localize a função `analyze_with_gemini`. Altere a linha do client na **linha 83**:

```python
# Antes:
client = genai.Client()

# Depois:
client = genai.Client(api_key="sua-chave-aqui")
```

> ⚠️ **Atenção:** Não compartilhe o arquivo `main.py` com a chave inserida publicamente (ex: GitHub). Adicione o `main.py` ao `.gitignore` ou remova a chave antes de commitar.

**5. Execute a API**
```bash
uvicorn main:app --reload
```

A API estará disponível em `http://localhost:8000`.
A documentação interativa estará em `http://localhost:8000/docs`.

> **Nota:** Na primeira execução, o modelo do MediaPipe (`face_landmarker.task`, ~30MB) será baixado automaticamente.

---

## 🔬 Como o Processo Funciona (Arquitetura)

1. **📷 Foto do cliente**: O usuário envia uma imagem via requisição POST para o endpoint `/analisar`.
2. **🧠 MediaPipe Face Landmarker**: A biblioteca escaneia a imagem e mapeia 478 pontos faciais.
3. **📐 Cálculo de proporções**: O código mede distâncias reais:
   - Razão entre altura e largura total;
   - Comparação da largura da mandíbula vs maçãs do rosto;
   - Comparação da largura da testa vs mandíbula.
4. **🤖 Google Gemini API**: A IA recebe as medidas precisas e realiza a **classificação do formato** e cria as **recomendações de cortes**.
5. **✂️ Resposta JSON estruturada**: A API retorna o formato de rosto identificado, dicas de visagismo e as 4 opções de corte.

---

## 📡 Endpoints da API

| Método | Rota | Descrição |
| :--- | :--- | :--- |
| `GET` | `/` | Checagem de saúde da API |
| `POST` | `/analisar` | Recebe uma imagem e retorna a análise de visagismo |

### Exemplo de requisição para `/analisar`

```bash
curl -X POST "http://localhost:8000/analisar" \
  -H "accept: application/json" \
  -F "file=@sua_foto.jpg"
```

### Exemplo de resposta

```json
{
  "sucesso": true,
  "metricas_computadas": {
    "face_height_px": 320.5,
    "face_width_px": 240.1,
    "ratio_height_width": 1.335,
    "ratio_jaw_cheek": 0.812,
    "ratio_forehead_jaw": 1.05
  },
  "analise_visagismo": {
    "formato_rosto": "Oval",
    "confianca": 0.88,
    "descricao_formato": "Rosto equilibrado com testa levemente mais larga que o queixo.",
    "caracteristicas_detectadas": ["proporção equilibrada", "queixo suavemente arredondado"],
    "dica_principal": "O rosto oval é o mais versátil — a maioria dos cortes funciona bem.",
    "estilos_recomendados": [
      {"nome": "Undercut", "justificativa": "Valoriza as proporções naturais do rosto."},
      {"nome": "Pompadour", "justificativa": "Adiciona volume no topo sem alargar as laterais."},
      {"nome": "Quiff", "justificativa": "Corte moderno que complementa a simetria facial."},
      {"nome": "Crop", "justificativa": "Corte limpo e atual, ideal para o formato oval."}
    ],
    "evitar": ["Franja pesada", "Volume excessivo nas laterais"]
  }
}
```

---

## 📋 Formatos de Rosto Detectados e Características

| Formato | Características Principais |
| :--- | :--- |
| **Oval** | Proporcional, com a testa ligeiramente mais larga que a pequena curva do queixo. |
| **Redondo** | Largura e altura quase iguais, traços suaves e bochechas cheias. |
| **Quadrado** | Linha da mandíbula forte e reta, com testa e maçãs de largura muito similar. |
| **Coração** | Testa mais larga, afinando gradativamente até um queixo mais pontudo. |
| **Oblongo** | Rosto longo e alongado, com largura uniforme ao longo de toda face. |
| **Diamante** | Maçãs do rosto são a área mais larga e proeminente, com testa e queixo estreitos. |

---

## 📁 Estrutura de Arquivos

```text
processador_imagem/
├── main.py              # Arquivo principal da API REST
├── requirements.txt     # Lista de dependências Python
├── face_landmarker.task # Modelo do MediaPipe (baixado automaticamente na 1ª execução)
└── README.md            # Esta documentação
```

---

## 💡 Dicas para Melhores Resultados

- Utilize uma **foto frontal** em que o rosto esteja centralizado.
- Procure por um ambiente com **boa iluminação**, preferencialmente luz natural.
- **Evite** utilizar óculos de sol, chapéus ou penteados que cubram partes do rosto ou da testa.
- A resolução recomendada para a imagem é de pelo menos **400x400 pixels**.
