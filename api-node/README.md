api-node/
├── src/
│ ├── api/
│ │ ├── controllers/ # Camada de controle (manipulação de req/res e lógica de rotas)
│ │ └── routes/ # Definição dos endpoints HTTP
│ ├── infra/
│ │ ├── config/ # Configuração de conexão com o banco de dados
│ │ ├── migrations/ # Versionamento do esquema do banco de dados
│ │ ├── models/ # Definição das entidades e associações do Sequelize
│ │ └── seeders/ # Alimentação inicial de dados (ex: tabela de cortes)
│ └── app.ts # Inicialização do Express, CORS e middlewares

## ⚙️ Instalação e Execução Local

### 1. Pré-requisitos

Certifique-se de ter instalado o **Node.js** e um servidor de banco de dados (ex: PostgreSQL) em execução.

### 2. Passos para Configuração

No terminal da pasta `api-node`:

```bash
# Instalar as dependências do projeto
npm install

# Configurar as variáveis de ambiente (.env)
# Crie um arquivo .env na raiz do projeto com as seguintes chaves:
PORT=3333
DB_HOST=localhost
DB_USER=seu_usuario
DB_PASS=sua_senha
DB_NAME=barbershop_db
DB_DIALECT=postgres
```

# Executar as migrations (Criar as tabelas)

```bash
npx sequelize-cli db:migrate
```

# Executar as seeders (Popular tabela de cortes/serviços)

```bash
npx sequelize-cli db:seed:all
```

# Iniciar em ambiente de desenvolvimento (recarregamento automático)

```bash
npm run start:dev
```
