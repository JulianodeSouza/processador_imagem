import express from 'express';
import cors from 'cors';
import path from 'path';
import dotenv from 'dotenv';
import { syncDatabase } from './infra/models';
import routes from './api/routes';

dotenv.config();

const app = express();

// Middlewares
app.use(cors()); // Permite requisições do frontend Next.js
app.use(express.json());

// Rota estática para servir as imagens que o Multer faz upload
app.use('/uploads', express.static(path.resolve(__dirname, '..', 'uploads')));

// Integração das Rotas
app.use(routes);

const PORT = process.env.PORT;

app.listen(PORT, async () => {
  console.log(`🚀 Servidor rodando na porta ${PORT}`);
  await syncDatabase(); // Cria/Atualiza as tabelas no MySQL automaticamente
});