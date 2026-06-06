import { Request, Response } from "express";
import { Agendamento, Barbeiro, Cliente } from "../infra/models";

export const BarbeiroController = {
  async getAll(req: Request, res: Response) {
    try {
      const barbeiros = await Barbeiro.findAll({ order: [["nome", "ASC"]] });
      const mapped = barbeiros.map((b: any) => ({
        id: b.id,
        name: b.nome,
        phone: b.telefone,
        email: b.email,
        isActive: b.ativo,
        status: b.ativo ? "Ativo" : "Inativo",
      }));
      return res.json(mapped);
    } catch (error) {
      return res.status(500).json({ error: "Erro ao buscar barbeiros" });
    }
  },

  async getById(req: Request, res: Response) {
    try {
      const id = req.params.id as string;
      const barbeiro = await Barbeiro.findByPk(id);
      if (!barbeiro)
        return res.status(404).json({ error: "Barbeiro não encontrado" });

      return res.json({
        id: (barbeiro as any).id,
        name: (barbeiro as any).nome,
        phone: (barbeiro as any).telefone,
        email: (barbeiro as any).email,
        isActive: (barbeiro as any).ativo,
      });
    } catch (error) {
      return res.status(500).json({ error: "Erro ao buscar barbeiro" });
    }
  },

  async create(req: Request, res: Response) {
    try {
      const { name, email, phone, isActive } = req.body;
      const novoBarbeiro = await Barbeiro.create({
        nome: name,
        email: email || null,
        telefone: phone || null,
        ativo: isActive !== undefined ? isActive : true,
      });
      return res.status(201).json(novoBarbeiro);
    } catch (error) {
      return res.status(500).json({ error: "Erro ao criar barbeiro" });
    }
  },

  async update(req: Request, res: Response) {
    try {
      const id = req.params.id as string;
      const { name, email, phone, isActive } = req.body;
      await Barbeiro.update(
        {
          nome: name,
          email,
          telefone: phone,
          ativo: isActive,
        },
        { where: { id } },
      );
      return res.json({ message: "Barbeiro atualizado com sucesso" });
    } catch (error) {
      return res.status(500).json({ error: "Erro ao atualizar barbeiro" });
    }
  },

  async delete(req: Request, res: Response) {
    try {
      const id = req.params.id as string;
      await Barbeiro.destroy({ where: { id } });
      return res.status(204).send();
    } catch (error) {
      return res.status(500).json({ error: "Erro ao remover barbeiro" });
    }
  },

  async getAgenda(req: Request, res: Response) {
    try {
      const id = req.params.id as string;
      const agendamentos = await Agendamento.findAll({
        where: { barbeiro_id: id },
        include: [Cliente],
        order: [["data_hora", "ASC"]],
      });

      const mapped = agendamentos.map((a: any) => ({
        id: a.id,
        clientName: a.Cliente?.nome || "Cliente",
        date: a.data_hora,
        status: a.status,
        service: "Corte",
        notes: "Nenhuma",
      }));
      return res.json(mapped);
    } catch (error) {
      return res.status(500).json({ error: "Erro ao buscar agenda" });
    }
  },
};
