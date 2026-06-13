import { Request, Response } from "express";
import {
  Agendamento,
  Cliente,
  Corte,
  HistoricoCorte,
  RecomendacaoVisagismo,
} from "../infra/models";

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
        include: [
          // Traz o histórico de cortes (se já existir na sua estrutura)
          {
            model: HistoricoCorte,
            required: false,
          },
          // NOVO: Traz o histórico de visagismo com o Corte atrelado
          {
            model: RecomendacaoVisagismo,
            include: [Corte],
            required: false,
          },
        ],
        // Ordena para trazer a análise de IA mais recente primeiro
        order: [[RecomendacaoVisagismo, "criado_em", "DESC"]],
      });

      if (!cliente) {
        return res.status(404).json({ error: "Cliente não encontrado" });
      }

      // Mapeia e remonta o JSON das recomendações de visagismo
      const recomendacoesDb = (cliente as any).RecomendacaoVisagismos || [];

      const historicoAnalises = recomendacoesDb.map((r: any) => {
        const partes = r.justificativa ? r.justificativa.split(" | ") : [];
        const dicaPrincipal = partes[0] || "Mantenha o equilíbrio do rosto.";

        const estilos = partes.slice(1).map((estiloStr: string) => {
          const [nome, ...just] = estiloStr.split(": ");
          return {
            nome: nome || "Corte Recomendado",
            justificativa:
              just.join(": ") ||
              "Recomendado com base nas proporções do rosto.",
          };
        });

        return {
          message: "Análise carregada do histórico do cliente",
          recommendation: {
            id: r.id,
            clientId: r.cliente_id,
            clientName: (cliente as any).nome,
            faceShape: r.formato_rosto_identificado,
            suggestedCutId: r.corte_sugerido_id,
            suggestedCutName: r.Corte ? r.Corte.nome : "Corte Personalizado",
            confidence: 0.85,
            description: `Análise resgatada do banco (feita em ${new Date(r.criado_em).toLocaleDateString("pt-BR")}).`,
            characteristics: [],
            justification: dicaPrincipal,
            suggestedCuts:
              estilos.length > 0
                ? estilos
                : [
                    {
                      nome: r.Corte ? r.Corte.nome : "Corte Clássico",
                      justificativa: dicaPrincipal,
                    },
                  ],
            avoid: [],
            createdAt: r.criado_em,
          },
          metrics: {
            face_height_px: 0,
            face_width_px: 0,
            ratio_height_width: 0,
            ratio_jaw_cheek: 0,
            ratio_forehead_jaw: 0,
          },
          photoUrl: "",
        };
      });

      // Retorna o cliente com as análises já injetadas!
      return res.json({
        id: (cliente as any).id,
        name: (cliente as any).nome,
        phone: (cliente as any).telefone,
        email: (cliente as any).email,
        history: (cliente as any).HistoricoCortes || [],
        historicoAnalises: historicoAnalises, // <- Mandamos o array pronto aqui
      });
    } catch (error) {
      console.error("[ClienteController] Erro ao buscar cliente:", error);
      return res.status(500).json({ error: "Erro ao buscar cliente" });
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

      // 1. Remove os registos dependentes primeiro para evitar o erro de Chave Estrangeira
      await RecomendacaoVisagismo.destroy({ where: { cliente_id: id } });
      await HistoricoCorte.destroy({ where: { cliente_id: id } });

      // Se tiver uma tabela de Agendamentos atrelada ao cliente, remova também:
      await Agendamento.destroy({ where: { cliente_id: id } });

      // 2. Agora sim, remove o cliente com segurança
      await Cliente.destroy({ where: { id } });

      return res.status(204).send();
    } catch (error) {
      console.error("[ClienteController] Erro ao deletar cliente:", error);
      return res.status(500).json({ error: "Error deleting client" });
    }
  },
};
