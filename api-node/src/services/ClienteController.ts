import { Request, Response } from "express";
import { Cliente, HistoricoCorte } from "../infra/models";

export const ClienteController = {
  async getAll(req: Request, res: Response) {
    try {
      const clientes = await Cliente.findAll({ order: [["nome", "ASC"]] });
      const mapped = clientes.map((c: any) => ({
        id: c.id,
        name: c.nome,
        phone: c.telefone,
        email: c.email,
        history: [],
      }));
      return res.json(mapped);
    } catch (error) {
      return res.status(500).json({ error: "Internal server error" });
    }
  },

  async getById(req: Request, res: Response) {
    try {
      const id = req.params.id as string;
      const cliente = await Cliente.findByPk(id, {
        include: [{ model: HistoricoCorte, as: "historicos" }],
      });
      if (!cliente) return res.status(404).json({ error: "Client not found" });
      return res.json({
        id: (cliente as any).id,
        name: (cliente as any).nome,
        phone: (cliente as any).telefone,
        email: (cliente as any).email,
      });
    } catch (error) {
      return res.status(500).json({ error: "Error fetching client" });
    }
  },

  async create(req: Request, res: Response) {
    try {
      const { name, clientName, email, clientEmail, phone, clientPhone } =
        req.body;
      const nomeFinal = name || clientName;
      const emailFinal = email || clientEmail;
      const telefoneFinal = phone || clientPhone;

      if (!nomeFinal)
        return res.status(400).json({ error: "Name is required" });

      const novoCliente = await Cliente.create({
        nome: nomeFinal,
        email: emailFinal || null,
        telefone: telefoneFinal || null,
      });

      return res.status(201).json({
        id: (novoCliente as any).id,
        name: (novoCliente as any).nome,
        email: (novoCliente as any).email,
        phone: (novoCliente as any).telefone,
      });
    } catch (error) {
      return res.status(500).json({ error: "Error creating client" });
    }
  },

  async update(req: Request, res: Response) {
    try {
      const id = req.params.id as string;
      const { name, email, phone } = req.body;
      const updateData: any = {};
      if (name !== undefined) updateData.nome = name;
      if (email !== undefined) updateData.email = email;
      if (phone !== undefined) updateData.telefone = phone;

      await Cliente.update(updateData, { where: { id } });
      const cliente = await Cliente.findByPk(id);
      if (!cliente) return res.status(404).json({ error: "Client not found" });

      return res.json({
        id: (cliente as any).id,
        name: (cliente as any).nome,
        email: (cliente as any).email,
        phone: (cliente as any).telefone,
      });
    } catch (error) {
      return res.status(500).json({ error: "Error updating client" });
    }
  },

  async delete(req: Request, res: Response) {
    try {
      const id = req.params.id as string;
      await Cliente.destroy({ where: { id } });
      return res.status(204).send();
    } catch (error) {
      return res.status(500).json({ error: "Error deleting client" });
    }
  },
};
