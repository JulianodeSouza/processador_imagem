import { Request, Response } from "express";
import { Agendamento, Cliente, Barbeiro } from "../infra/models";

export const AnalyticsController = {
  async getDashboard(req: Request, res: Response) {
    try {
      const totalAgendamentos = await Agendamento.count();
      const novosClientes = await Cliente.count();

      return res.json({
        todayAppointments: totalAgendamentos.toString(),
        newClients: novosClientes.toString(),
        estimatedRevenue: `R$ ${totalAgendamentos * 80},00`,
      });
    } catch (error) {
      return res.status(500).json({ error: "Error loading dashboard metrics" });
    }
  },

  async getCortesPopulares(req: Request, res: Response) {
    return res.json([
      { service: "Corte Clássico", total: 45 },
      { service: "Fade Texturizado", total: 32 },
    ]);
  },

  async getHorariosPico(req: Request, res: Response) {
    return res.json([
      { time: "18:00", appointments: 15 },
      { time: "19:00", appointments: 12 },
    ]);
  },

  async getBarbeirosDestaque(req: Request, res: Response) {
    try {
      const barbeiros = await Barbeiro.findAll({ limit: 5 });
      const mapped = barbeiros.map((b: any) => ({
        id: b.id,
        name: b.nome,
      }));
      return res.json(mapped);
    } catch (error) {
      return res
        .status(500)
        .json({ error: "Error loading highlighted barbers" });
    }
  },
};
