import { Request, Response } from "express";
import { Corte } from "../infra/models";

export const CorteController = {
  async getAll(req: Request, res: Response) {
    try {
      const cortes = await Corte.findAll({ order: [["nome", "ASC"]] });
      const mapped = cortes.map((c: any) => ({
        id: c.id,
        name: c.nome,
        nome: c.nome,
        description: c.descricao,
        price: c.preco,
      }));
      return res.json(mapped);
    } catch (error) {
      return res.status(500).json({ error: "Error fetching services" });
    }
  },
};
