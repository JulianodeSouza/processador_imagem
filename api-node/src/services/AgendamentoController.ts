import { Request, Response } from "express";
import { Agendamento, Cliente, Barbeiro } from "../infra/models";
import { Op } from "sequelize";

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
      const agora = new Date();

      const diferencaMilissegundos =
        data_hora_formatada.getTime() - agora.getTime();
      const diferencaHoras = diferencaMilissegundos / (1000 * 60 * 60);

      if (diferencaHoras < 1) {
        return res.status(400).json({
          error:
            "O agendamento deve ser realizado com no mínimo 1 hora de antecedência.",
        });
      }

      const horaAgendamento = parseInt(time.split(":")[0], 10);
      const jornadaInicio = 9; // 09:00
      const jornadaFim = 18; // 18:00

      if (horaAgendamento < jornadaInicio || horaAgendamento >= jornadaFim) {
        return res.status(400).json({
          error: `O horário selecionado (${time}) está fora da jornada de trabalho do barbeiro (${jornadaInicio}:00 às ${jornadaFim}:00).`,
        });
      }

      // Busca ou cria o cliente
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

      const agendamentosAtivos = await Agendamento.count({
        where: {
          cliente_id: (cliente as any).id,
          status: "Agendado", // Considerando "Agendado" como status ativo
        },
      });

      if (agendamentosAtivos >= 2) {
        return res.status(400).json({
          error:
            "O cliente já possui o limite máximo de 2 agendamentos ativos.",
        });
      }

      const barbeiroObj = await Barbeiro.findOne({ where: { nome: barber } });
      if (!barbeiroObj)
        return res.status(404).json({ error: "Barber not found" });

      // Criação final do agendamento
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

      if (date && time) {
        const data_hora_formatada = new Date(`${date}T${time}:00`);
        const agora = new Date();

        // RN01 para atualizações
        const diferencaHoras =
          (data_hora_formatada.getTime() - agora.getTime()) / (1000 * 60 * 60);
        if (diferencaHoras < 1) {
          return res
            .status(400)
            .json({
              error: "A remarcação deve ter no mínimo 1 hora de antecedência.",
            });
        }

        // RN05 para atualizações
        const horaAgendamento = parseInt(time.split(":")[0], 10);
        if (horaAgendamento < 9 || horaAgendamento >= 18) {
          return res
            .status(400)
            .json({
              error: "Horário fora da jornada de trabalho do barbeiro.",
            });
        }

        updateData.data_hora = data_hora_formatada;
      }

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
  async getDisponibilidade(req: Request, res: Response) {
    try {
      const { date, barber } = req.query;

      if (!date || !barber) {
        return res
          .status(400)
          .json({ error: "Data e barbeiro são obrigatórios" });
      }

      // Busca o ID do barbeiro pelo nome
      const barbeiroObj = await Barbeiro.findOne({
        where: { nome: barber as string },
      });
      if (!barbeiroObj)
        return res.status(404).json({ error: "Barbeiro não encontrado" });

      // Define o início e o fim do dia selecionado
      const dataInicio = new Date(`${date}T00:00:00`);
      const dataFim = new Date(`${date}T23:59:59`);

      // Busca agendamentos desse barbeiro neste dia
      const agendamentos = await Agendamento.findAll({
        where: {
          barbeiro_id: (barbeiroObj as any).id,
          status: "Agendado",
          data_hora: {
            [Op.between]: [dataInicio, dataFim],
          },
        },
      });

      // Extrai apenas as horas já ocupadas (ex: "10:00")
      const horariosOcupados = agendamentos.map((a: any) => {
        const dt = new Date(a.data_hora);
        return dt.toTimeString().split(" ")[0].substring(0, 5);
      });

      // Gera a jornada de trabalho padrão (09:00 às 17:00, pois 18h fecha)
      const horariosPossiveis = [];
      for (let i = 9; i < 18; i++) {
        horariosPossiveis.push(`${i.toString().padStart(2, "0")}:00`);
      }

      // Remove os horários que já estão no banco de dados
      const horariosDisponiveis = horariosPossiveis.filter(
        (h) => !horariosOcupados.includes(h),
      );

      // RN01: Se a data selecionada for hoje, esconde os horários que já passaram
      // ou estão a menos de 1 hora de distância
      const agora = new Date();
      const hojeStr = agora.toISOString().split("T")[0];

      const horariosFinais = horariosDisponiveis.filter((h) => {
        if (date === hojeStr) {
          const horaSlot = new Date(`${date}T${h}:00`);
          const diferencaHoras =
            (horaSlot.getTime() - agora.getTime()) / (1000 * 60 * 60);
          return diferencaHoras >= 1;
        }
        return true;
      });

      return res.json(horariosFinais);
    } catch (error) {
      console.error(error);
      return res.status(500).json({ error: "Erro ao buscar disponibilidade" });
    }
  },
};
