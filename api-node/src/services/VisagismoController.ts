import { Request, Response } from "express";
import { RecomendacaoVisagismo, Cliente, Corte } from "../infra/models";

export const VisagismoController = {
  async getAll(req: Request, res: Response) {
    try {
      const resultados = await RecomendacaoVisagismo.findAll({
        include: [
          { model: Cliente, attributes: ["nome"] },
          { model: Corte, attributes: ["nome", "preco"] },
        ],
        order: [["criado_em", "DESC"]],
      });

      const mapped = resultados.map((r: any) => ({
        id: r.id,
        clientId: r.cliente_id,
        clientName: r.Cliente ? r.Cliente.nome : "Unknown",
        faceShape: r.formato_rosto_identificado,
        suggestedCutId: r.corte_sugerido_id,
        suggestedCutName: r.Corte ? r.Corte.nome : null,
        justification: r.justificativa,
        createdAt: r.criado_em,
      }));

      return res.json(mapped);
    } catch (error) {
      return res.status(500).json({ error: "Error fetching visagism history" });
    }
  },

  async getResultado(req: Request, res: Response) {
    try {
      const id = req.params.id as string;
      const r = await RecomendacaoVisagismo.findByPk(id, {
        include: [Cliente, Corte],
      });
      if (!r) return res.status(404).json({ error: "Record not found" });

      return res.json({
        id: (r as any).id,
        clientId: (r as any).cliente_id,
        clientName: (r as any).Cliente ? (r as any).Cliente.nome : "Unknown",
        faceShape: (r as any).formato_rosto_identificado,
        suggestedCutId: (r as any).corte_sugerido_id,
        justification: (r as any).justificativa,
        createdAt: (r as any).criado_em,
      });
    } catch (error) {
      return res.status(500).json({ error: "Error fetching record" });
    }
  },

  async analisar(req: Request, res: Response) {
    try {
      if (!req.file)
        return res.status(400).json({ error: "No image file uploaded" });
      const { clientId } = req.body;
      if (!clientId)
        return res.status(400).json({ error: "clientId is required" });

      const imageUrl = `http://localhost:3333/uploads/${req.file.filename}`;

      const resultadoPython = {
        faceShape: "Rosto Oval",
        justificativa: "Proporções ideais equilibradas verticalmente.",
        corteSugerido: "Corte Clássico",
      };

      const corteSugeridoDb = await Corte.findOne({
        where: { nome: resultadoPython.corteSugerido },
      });

      const novaRecomendacao = await RecomendacaoVisagismo.create({
        cliente_id: clientId,
        formato_rosto_identificado: resultadoPython.faceShape,
        corte_sugerido_id: corteSugeridoDb ? (corteSugeridoDb as any).id : null,
        justificativa: resultadoPython.justificativa,
      });

      return res.status(201).json({
        message: "Analysis completed successfully",
        recommendation: {
          id: (novaRecomendacao as any).id,
          faceShape: (novaRecomendacao as any).formato_rosto_identificado,
          justification: (novaRecomendacao as any).justificativa,
        },
        photoUrl: imageUrl,
      });
    } catch (error) {
      return res
        .status(500)
        .json({ error: "Error processing visagism analysis" });
    }
  },
};
