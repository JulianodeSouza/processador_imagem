import { Request, Response } from "express";
import { RecomendacaoVisagismo, Cliente, Corte } from "../infra/models";
import FormData from "form-data";
import fs from "fs";
import axios from "axios";

const PYTHON_API_URL = process.env.PYTHON_API_URL || "http://localhost:8000";

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
      // 1. Valida se o arquivo foi enviado
      if (!req.file)
        return res.status(400).json({ error: "No image file uploaded" });

      const { clientId } = req.body;
      if (!clientId)
        return res.status(400).json({ error: "clientId is required" });

      // 2. Monta o FormData para enviar o arquivo à API Python
      const form = new FormData();
      form.append("file", fs.createReadStream(req.file.path), {
        filename: req.file.originalname,
        contentType: req.file.mimetype,
      });

      // 3. Chama a API Python no endpoint /analisar
      const pythonResponse = await axios.post(
        `${PYTHON_API_URL}/analisar`,
        form,
        {
          headers: form.getHeaders(),
        },
      );

      if (pythonResponse.status !== 200) {
        const errorData = pythonResponse;
        return res.status(502).json({
          error: "Erro ao processar imagem na API Python.",
          details:
            errorData?.data || "No additional error information provided.",
        });
      }

      const resultadoPython = pythonResponse.data;

      // 4. Extrai os dados retornados pela API Python
      // Estrutura esperada: resultadoPython.analise_visagismo
      const analise = resultadoPython.analise_visagismo;
      const formatoRosto = analise.formato_rosto;

      // Pega o nome do primeiro corte sugerido para buscar no banco
      const primeiroCorte = analise.estilos_recomendados?.[0]?.nome || null;

      // Monta a justificativa com todos os cortes sugeridos
      const justificativa = [
        analise.dica_principal,
        ...(analise.estilos_recomendados || []).map(
          (e: any) => `${e.nome}: ${e.justificativa}`,
        ),
      ].join(" | ");

      // 5. Busca o corte no banco de dados pelo nome
      const corteSugeridoDb = primeiroCorte
        ? await Corte.findOne({ where: { nome: primeiroCorte } })
        : null;

      // 6. Salva a recomendação no banco
      const novaRecomendacao = await RecomendacaoVisagismo.create({
        cliente_id: clientId,
        formato_rosto_identificado: formatoRosto,
        corte_sugerido_id: corteSugeridoDb ? (corteSugeridoDb as any).id : null,
        justificativa,
      });

      const imageUrl = `http://localhost:${process.env.PORT || 3333}/uploads/${req.file.filename}`;

      // 7. Retorna o resultado completo ao cliente
      return res.status(201).json({
        message: "Analysis completed successfully",
        recommendation: {
          id: (novaRecomendacao as any).id,
          faceShape: formatoRosto,
          confidence: analise.confianca,
          description: analise.descricao_formato,
          characteristics: analise.caracteristicas_detectadas,
          justification: analise.dica_principal,
          suggestedCuts: analise.estilos_recomendados,
          avoid: analise.evitar,
        },
        metrics: resultadoPython.metricas_computadas,
        photoUrl: imageUrl,
      });
    } catch (error) {
      console.error("[VisagismoController] Erro:", error);
      return res
        .status(500)
        .json({ error: "Error processing visagism analysis" });
    }
  },
};
