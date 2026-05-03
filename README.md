# 💈 Identificador de Formato de Rosto e Recomendações de Corte (Visagismo)

Um aplicativo interativo desenvolvido em Python utilizando **Streamlit**, focado em **Visagismo**. Ele utiliza visão computacional e Inteligência Artificial para analisar a foto do usuário, identificar o formato de seu rosto com precisão matemática e sugerir os melhores cortes de cabelo de acordo com técnicas profissionais de visagismo.

---

## ✨ Funcionalidades do Projeto

- **Detecção Facial de Alta Precisão**: Realiza o mapeamento de 478 pontos faciais utilizando o modelo *MediaPipe Face Mesh*.
- **Extração de Métricas Faciais**: Calcula matematicamente as proporções do rosto (razão altura/largura, mandíbula vs maçãs do rosto, testa vs mandíbula).
- **Análise por Inteligência Artificial**: Integração com a API do **Claude Vision** para processar a imagem e as métricas extraídas, garantindo uma identificação humanizada e precisa do formato do rosto.
- **6 Formatos de Rosto Suportados**: Capaz de classificar o rosto como Oval, Redondo, Quadrado, Coração, Oblongo ou Diamante.
- **Recomendações Personalizadas**: Sugere 4 opções de corte de cabelo ideais para o usuário, acompanhadas de justificativas com base no formato do rosto.
- **Interface Fluida e Elegante**: Interface moderna e amigável desenvolvida em Streamlit, com visual arrojado estilo "barbearia premium".

---

## 🚀 Como Executar o Projeto Localmente

Para rodar este projeto em sua própria máquina, siga os passos detalhados abaixo:

### ⚙️ Pré-requisitos
- Ter o [Python](https://www.python.org/downloads/) (versão 3.8 ou superior) instalado.
- Ter uma conta na [Anthropic](https://console.anthropic.com/) para gerar uma chave de API (necessária para a análise do Claude Vision).

### 🛠️ Passo a Passo

**1. Clone o repositório e acesse a pasta**
```bash
git clone https://github.com/seu-usuario/processador_imagem.git
cd processador_imagem
```

**2. Crie um Ambiente Virtual (Opcional, mas altamente recomendado)**
```bash
# Criar o ambiente virtual (chame de 'venv')
python -m venv venv

# Ativar o ambiente virtual no Windows:
venv\Scripts\activate

# Ativar o ambiente virtual no Linux / macOS:
source venv/bin/activate
```

**3. Instale as dependências do projeto**
```bash
pip install -r requirements.txt
```

**4. Configure a chave da API do Claude (Anthropic)**
Obtenha sua chave de API criando uma conta no [Console da Anthropic](https://console.anthropic.com/). Após gerar a chave (`sk-ant-api03...`), defina-a como variável de ambiente:

```bash
# Windows (PowerShell):
$env:ANTHROPIC_API_KEY = "sua-chave-aqui"

# Windows (Prompt de Comando - CMD):
set ANTHROPIC_API_KEY=sua-chave-aqui

# Linux / macOS:
export ANTHROPIC_API_KEY="sua-chave-aqui"
```

**5. Execute a aplicação**
No seu terminal, digite o comando:
```bash
streamlit run app.py
```
O servidor será iniciado e a aplicação abrirá automaticamente no seu navegador padrão (geralmente através do endereço `http://localhost:8501`).

---

## 🔬 Como o Processo Funciona (Arquitetura)

1. **📷 Foto do cliente**: O usuário faz o upload de uma imagem.
2. **🧠 MediaPipe Face Mesh**: A biblioteca escaneia a imagem e mapeia 478 pontos faciais tridimensionais.
3. **📐 Cálculo de proporções**: O código mede distâncias reais:
   - Razão entre altura e largura total;
   - Comparação da largura da mandíbula vs maçãs do rosto;
   - Comparação da largura da testa vs mandíbula.
4. **🤖 Claude Vision API**: A IA recebe a foto juntamente com as medidas precisas, unindo tecnologia visual e dados para realizar a **classificação do formato** e criar as **recomendações de cortes**.
5. **✂️ Exibição dos resultados**: A tela final exibe o mapa de pontos faciais gerado, qual formato de rosto foi identificado, dicas extras de visagismo e as 4 opções de corte visualizadas em tela.

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
├── app.py           # Arquivo principal da aplicação (Dashboard e Lógica)
├── requirements.txt # Lista de bibliotecas e dependências Python
└── README.md        # Esta documentação
```

---

## 💡 Dicas para Melhores Resultados ao Usar o App
- Utilize uma **foto frontal** em que o rosto esteja centralizado.
- Procure por um ambiente com **boa iluminação**, preferencialmente luz natural.
- **Evite** utilizar óculos de sol, chapéus ou penteados que cubram partes do rosto ou da testa.
- A resolução recomendada para a imagem é de pelo menos **400x400 pixels**.
