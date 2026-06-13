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
          
          // DADOS AGORA DINÂMICOS LIDOS DA BASE DE DADOS:
          service: a.servico || "Corte Clássico", 
          notes: a.observacoes || "",
          duration: "30 min", // Mantido fixo a 30 min para alinhar com os blocos da agenda, ou pode criar uma coluna 'duracao' no BD
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
        service: (a as any).servico,
        notes: (a as any).observacoes
      });
    } catch (error) {
      return res.status(500).json({ error: "Error fetching appointment" });
    }
  },

  async create(req: Request, res: Response) {
    try {
      // Adicionado service e notes que vêm do payload do front-end
      const { clientId, clientName, clientEmail, clientPhone, barber, date, time, service, notes } = req.body;

      if ((!clientId && !clientName) || !date || !time || !barber) {
        return res.status(400).json({ error: "Missing required fields" });
      }

      const data_hora_formatada = new Date(`${date}T${time}:00`);
      const agora = new Date();

      const diferencaMilissegundos = data_hora_formatada.getTime() - agora.getTime();
      const diferencaHoras = diferencaMilissegundos / (1000 * 60 * 60);

      if (diferencaHoras < 1) {
        return res.status(400).json({
          error: "O agendamento deve ser realizado com no mínimo 1 hora de antecedência.",
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

      let cliente;
      if (clientId) {
        cliente = await Cliente.findByPk(clientId);
      } else if (clientEmail) {
        cliente = await Cliente.findOne({ where: { email: clientEmail } });
      }

      if (!cliente && clientName) {
        cliente = await Cliente.create({
          nome: clientName,
          email: clientEmail || null,
          telefone: clientPhone || null,
        });
      }

      if (!cliente) {
        return res.status(404).json({ error: "Cliente não encontrado na base de dados." });
      }

      const agendamentosAtivos = await Agendamento.count({
        where: {
          cliente_id: (cliente as any).id,
          status: "Agendado", 
        },
      });

      if (agendamentosAtivos >= 2) {
        return res.status(400).json({
          error: "O cliente já possui o limite máximo de 2 agendamentos ativos.",
        });
      }

      const barbeiroObj = await Barbeiro.findOne({ where: { nome: barber } });
      if (!barbeiroObj)
        return res.status(404).json({ error: "Barber not found" });

      // Criação final do agendamento com serviço e notas
      const novoAgendamento = await Agendamento.create({
        cliente_id: (cliente as any).id,
        barbeiro_id: (barbeiroObj as any).id,
        data_hora: data_hora_formatada,
        status: "Agendado",
        servico: service || "Corte Clássico", 
        observacoes: notes || "", 
      });

      return res.status(201).json({
        id: (novoAgendamento as any).id,
        clientId: (novoAgendamento as any).cliente_id,
        barberId: (novoAgendamento as any).barbeiro_id,
        status: (novoAgendamento as any).status,
        service: (novoAgendamento as any).servico,
        notes: (novoAgendamento as any).observacoes
      });
    } catch (error) {
      console.error("[AgendamentoController]", error);
      return res.status(500).json({ error: "Error creating appointment" });
    }
  },

  async update(req: Request, res: Response) {
    try {
      const id = req.params.id as string;
      // Adicionado service e notes para suportar edições e remarcações avançadas
      const { barber, date, time, status, service, notes } = req.body;

      const updateData: any = {};

      if (date && time) {
        const data_hora_formatada = new Date(`${date}T${time}:00`);
        const agora = new Date();

        const diferencaHoras = (data_hora_formatada.getTime() - agora.getTime()) / (1000 * 60 * 60);
        if (diferencaHoras < 1) {
          return res.status(400).json({ error: "A remarcação deve ter no mínimo 1 hora de antecedência." });
        }

        const horaAgendamento = parseInt(time.split(":")[0], 10);
        if (horaAgendamento < 9 || horaAgendamento >= 18) {
          return res.status(400).json({ error: "Horário fora da jornada de trabalho do barbeiro." });
        }

        updateData.data_hora = data_hora_formatada;
      }

      if (status) updateData.status = status;
      if (service) updateData.servico = service;
      if (notes !== undefined) updateData.observacoes = notes;

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
        return res.status(400).json({ error: "Data e barbeiro são obrigatórios" });
      }

      const barbeiroObj = await Barbeiro.findOne({ where: { nome: barber as string } });
      if (!barbeiroObj) return res.status(404).json({ error: "Barbeiro não encontrado" });

      const dataInicio = new Date(`${date}T00:00:00`);
      const dataFim = new Date(`${date}T23:59:59`);

      const agendamentos = await Agendamento.findAll({
        where: {
          barbeiro_id: (barbeiroObj as any).id,
          status: "Agendado",
          data_hora: { [Op.between]: [dataInicio, dataFim] },
        },
      });

      const horariosOcupados = agendamentos.map((a: any) => {
        const dt = new Date(a.data_hora);
        // Regra atualizada para extrair hora e minuto para evitar conflitos de 30 em 30 min
        return `${String(dt.getHours()).padStart(2, '0')}:${String(dt.getMinutes()).padStart(2, '0')}`;
      });

      const horariosPossiveis = [];
      for (let i = 9; i < 18; i++) {
        const horaFormatada = i.toString().padStart(2, "0");
        horariosPossiveis.push(`${horaFormatada}:00`);
        horariosPossiveis.push(`${horaFormatada}:30`);
      }

      const horariosDisponiveis = horariosPossiveis.filter((h) => !horariosOcupados.includes(h));

      const agora = new Date();
      const hojeStr = agora.toISOString().split("T")[0];

      const horariosFinais = horariosDisponiveis.filter((h) => {
        if (date === hojeStr) {
          const [hora, minuto] = h.split(':');
          const horaSlot = new Date(`${date}T${hora}:${minuto}:00`);
          const diferencaHoras = (horaSlot.getTime() - agora.getTime()) / (1000 * 60 * 60);
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