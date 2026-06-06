# 🔗 Integração API Python → API Node.js

Este guia explica como integrar a API Python (processamento visual de imagens) com a API Node.js (gerenciamento e banco de dados).

---

## 📐 Arquitetura do Fluxo

```
┌──────────┐   upload da imagem   ┌─────────────┐
│  Cliente │ ──────────────────►  │  API Python │
└──────────┘                      └──────┬──────┘
                                         │ processa a imagem
                                         │ chama o Node automaticamente
                                         ▼
                                  ┌─────────────┐
                                  │  API Node   │
                                  │  (webhook)  │
                                  └──────┬──────┘
                                         │ salva resultado
                                         ▼
                                    PostgreSQL
```

---

## 📁 Arquivos da Integração

| Arquivo | Destino no projeto | Função |
|---|---|---|
| `webhookController.ts` | `src/api/controllers/` | Recebe o JSON da Python e salva no banco |
| `webhookRoutes.ts` | `src/api/routes/` | Define os endpoints do webhook |
| `.env.example` | raiz do projeto | Variáveis de ambiente necessárias |

---

## ⚙️ Configuração

### 1. Variáveis de ambiente

Copie o arquivo de exemplo e preencha:

```bash
cp .env.example .env
```

Gere um token secreto seguro para proteger o webhook:

```bash
openssl rand -hex 32
```

Cole o valor gerado em `WEBHOOK_SECRET` no seu `.env`.

### 2. Registrar a rota no `app.ts`

```ts
import webhookRoutes from './api/routes/webhookRoutes';

app.use('/api/webhook', webhookRoutes);
```

---

## 🐍 O que a API Python precisa fazer

Após processar a imagem, a Python deve chamar o Node via `POST` com o resultado:

```python
import requests

def enviar_resultado_para_node(image_id, status, confianca, detalhes):
    url = "http://localhost:3333/api/webhook/image-result"

    payload = {
        "image_id": image_id,
        "status": status,         # ex: "aprovado", "reprovado"
        "confianca": confianca,   # ex: 0.97
        "detalhes": detalhes      # dict com dados extras
    }

    headers = {
        "Content-Type": "application/json",
        "x-webhook-secret": "seu_token_secreto_aqui"
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()
```

---

## 🛢️ Salvando no Banco de Dados

Descomente o bloco `ImageAnalysis.create(...)` no controller após criar o model.

**Exemplo de migration:**

```ts
module.exports = {
  up: async (queryInterface, Sequelize) => {
    await queryInterface.createTable('image_analyses', {
      id:         { type: Sequelize.INTEGER, autoIncrement: true, primaryKey: true },
      image_id:   { type: Sequelize.STRING, allowNull: false, unique: true },
      status:     { type: Sequelize.STRING, allowNull: false },
      confianca:  { type: Sequelize.FLOAT, allowNull: true },
      detalhes:   { type: Sequelize.JSONB, allowNull: true },
      created_at: { type: Sequelize.DATE, defaultValue: Sequelize.NOW },
    });
  },
  down: async (queryInterface) => {
    await queryInterface.dropTable('image_analyses');
  },
};
```

```bash
npx sequelize-cli db:migrate
```

---

## 🧪 Testando o Webhook Manualmente

Simule uma chamada da API Python com `curl`:

```bash
curl -X POST http://localhost:3333/api/webhook/image-result \
  -H "Content-Type: application/json" \
  -H "x-webhook-secret: seu_token_secreto_aqui" \
  -d '{
    "image_id": "foto_001",
    "status": "aprovado",
    "confianca": 0.97,
    "detalhes": { "objeto_detectado": "rosto", "qualidade": "alta" }
  }'
```

**Resposta esperada:**
```json
{
  "message": "Resultado recebido e salvo com sucesso.",
  "image_id": "foto_001",
  "status": "aprovado"
}
```

Consultando um resultado salvo:

```bash
curl http://localhost:3333/api/webhook/image-result/foto_001
```

---

## 🚨 Tratamento de Erros

| Situação | Status HTTP | Causa |
|---|---|---|
| `image_id` ou `status` ausente | `400` | Payload incompleto enviado pela Python |
| Token secreto errado ou ausente | `401` | A Python não enviou o `x-webhook-secret` correto |
| Erro interno no Node | `500` | Verifique os logs do servidor |

---

## 🔐 Segurança

- **Nunca exponha** o `WEBHOOK_SECRET` no código-fonte ou repositório.
- Em produção, use **HTTPS** para que o token não trafegue em texto puro.
- Se Node e Python estiverem no **mesmo servidor**, restrinja o endpoint a `localhost` via firewall.

---

## 📌 Variáveis de Ambiente — Resumo

| Variável | Onde usar | Descrição |
|---|---|---|
| `WEBHOOK_SECRET` | Node `.env` | Token que valida chamadas da Python |
| `NODE_WEBHOOK_URL` | Python `.env` | URL do endpoint Node para a Python chamar |
