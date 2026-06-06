import { Request, Response } from "express";
import { Agendamento, Cliente, Barbeiro } from "../infra/models";

export const AgendamentoController = {
  async getAll(req: Request, res: Response) {
    try {
      const agendamentos = await Agendamento.findAll({
        order: [["data_hora", "ASC"]],
        include: [Cliente, Barbeiro],
      });

      const mapped = agendamentos.map((a: any) => {
        const dt = new Date(a.data_hora);
        return {
          id: a.id,
          clientId: a.cliente_id,
          barberId: a.barbeiro_id,
          date: dt.toISOString().split("T")[0],
          time: dt.toTimeString().split(" ")[0].substring(0, 5),
          status: a.status.toLowerCase().replace(" ", "_"),
          clientName: a.Cliente ? a.Cliente.nome : "Cliente Desconhecido",
          service: "Corte",
          notes: "Observação interna",
          duration: "45 min",
        };
      });

      return res.json(mapped);
    } catch (error) {
      return res.status(500).json({ error: "Error fetching appointments" });
    }
  },

  async getById(req: Request, res: Response) {
    try {
      const id = req.params.id as string;
      const a = await Agendamento.findByPk(id, {
        include: [Cliente, Barbeiro],
      });
      if (!a) return res.status(404).json({ error: "Appointment not found" });

      const dt = new Date((a as any).data_hora);
      return res.json({
        id: (a as any).id,
        clientId: (a as any).cliente_id,
        barberId: (a as any).barbeiro_id,
        date: dt.toISOString().split("T")[0],
        time: dt.toTimeString().split(" ")[0].substring(0, 5),
        status: (a as any).status,
        clientName: (a as any).Cliente ? (a as any).Cliente.nome : null,
      });
    } catch (error) {
      return res.status(500).json({ error: "Error fetching appointment" });
    }
  },

  async create(req: Request, res: Response) {
    try {
      const { clientName, clientEmail, clientPhone, barber, date, time } =
        req.body;

      if (!clientName || !date || !time || !barber) {
        return res.status(400).json({ error: "Missing required fields" });
      }

      const data_hora_formatada = new Date(`${date}T${time}:00`);

      let cliente = await Cliente.findOne({
        where: { email: clientEmail || "" },
      });
      if (!cliente) {
        cliente = await Cliente.create({
          nome: clientName,
          email: clientEmail || null,
          telefone: clientPhone || null,
        });
      }

      const barbeiroObj = await Barbeiro.findOne({ where: { nome: barber } });
      if (!barbeiroObj)
        return res.status(404).json({ error: "Barber not found" });

      const novoAgendamento = await Agendamento.create({
        cliente_id: (cliente as any).id,
        barbeiro_id: (barbeiroObj as any).id,
        data_hora: data_hora_formatada,
        status: "Agendado",
      });

      return res.status(201).json({
        id: (novoAgendamento as any).id,
        clientId: (novoAgendamento as any).cliente_id,
        barberId: (novoAgendamento as any).barbeiro_id,
        status: (novoAgendamento as any).status,
      });
    } catch (error) {
      return res.status(500).json({ error: "Error creating appointment" });
    }
  },

  async update(req: Request, res: Response) {
    try {
      const id = req.params.id as string;
      const { barber, date, time, status } = req.body;

      const updateData: any = {};
      if (date && time) updateData.data_hora = new Date(`${date}T${time}:00`);
      if (status) updateData.status = status;

      if (barber) {
        const barbeiroObj = await Barbeiro.findOne({ where: { nome: barber } });
        if (barbeiroObj) updateData.barbeiro_id = (barbeiroObj as any).id;
      }

      await Agendamento.update(updateData, { where: { id } });
      const a = await Agendamento.findByPk(id);

      return res.json({
        id: (a as any).id,
        status: (a as any).status,
      });
    } catch (error) {
      return res.status(500).json({ error: "Error updating appointment" });
    }
  },

  async delete(req: Request, res: Response) {
    try {
      const id = req.params.id as string;
      await Agendamento.destroy({ where: { id } });
      return res.status(204).send();
    } catch (error) {
      return res.status(500).json({ error: "Error deleting appointment" });
    }
  },
};
