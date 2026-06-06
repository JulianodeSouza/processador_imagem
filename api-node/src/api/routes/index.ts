import { Router } from "express";
import multer from "multer";

import { ClienteController } from "../../services/ClienteController";
import { BarbeiroController } from "../../services/BarbeiroController";
import { AgendamentoController } from "../../services/AgendamentoController";
import { HistoricoCorteController } from "../../services/HistoricoCorteController";
import { VisagismoController } from "../../services/VisagismoController";
import { AnalyticsController } from "../../services/AnalyticsController";
import { CorteController } from "../../services/CorteController";

const routes = Router();

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, "uploads/"),
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});
const upload = multer({ storage });

// Analytics
routes.get("/analytics/dashboard", AnalyticsController.getDashboard);
routes.get(
  "/analytics/cortes-populares",
  AnalyticsController.getCortesPopulares,
);
routes.get("/analytics/horarios-pico", AnalyticsController.getHorariosPico);
routes.get(
  "/analytics/barbeiros-destaque",
  AnalyticsController.getBarbeirosDestaque,
);

// Agendamentos
routes.get("/agendamentos", AgendamentoController.getAll);
routes.get("/agendamentos/:id", AgendamentoController.getById);
routes.post("/agendamentos", AgendamentoController.create);
routes.put("/agendamentos/:id", AgendamentoController.update);
routes.delete("/agendamentos/:id", AgendamentoController.delete);

// Clientes
routes.get("/clientes", ClienteController.getAll);
routes.get("/clientes/:id", ClienteController.getById);
routes.post("/clientes", ClienteController.create);
routes.put("/clientes/:id", ClienteController.update);
routes.delete("/clientes/:id", ClienteController.delete);

// Barbeiros
routes.get("/barbeiros/:id/agenda", BarbeiroController.getAgenda);
routes.get("/barbeiros", BarbeiroController.getAll);
routes.get("/barbeiros/:id", BarbeiroController.getById);
routes.post("/barbeiros", BarbeiroController.create);
routes.put("/barbeiros/:id", BarbeiroController.update);
routes.delete("/barbeiros/:id", BarbeiroController.delete);

// Histórico
routes.get("/historico-cortes", HistoricoCorteController.getAll);
routes.get("/historico-cortes/:id", HistoricoCorteController.getById);
routes.post("/historico-cortes", HistoricoCorteController.create);

// Visagismo
routes.post(
  "/visagismo/analisar",
  upload.single("file"),
  VisagismoController.analisar,
);
routes.get("/visagismo/resultados", VisagismoController.getAll);
routes.get("/visagismo/resultado/:id", VisagismoController.getResultado);

// Serviços
routes.get("/cortes", CorteController.getAll);

export default routes;
