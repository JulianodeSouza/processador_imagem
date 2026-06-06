import { Request, Response } from 'express';
import { HistoricoCorte, FotoResultado } from '../infra/models';

export const HistoricoCorteController = {
  async getAll(req: Request, res: Response) {
    try {
      const historicos = await HistoricoCorte.findAll({ 
        order: [['realizado_em', 'DESC']],
        include: ['photos'] 
      });
      const mapped = historicos.map((h: any) => ({
        id: h.id,
        clientId: h.cliente_id,
        appointmentId: h.agendamento_id,
        serviceId: h.corte_id,
        notes: h.observacoes,
        realizedAt: h.realizado_em
      }));
      return res.json(mapped);
    } catch (error) {
      return res.status(500).json({ error: "Error fetching history" });
    }
  },

  async getById(req: Request, res: Response) {
    try {
      const id = req.params.id as string;
      const h = await HistoricoCorte.findByPk(id, { include: ['photos'] });
      if (!h) return res.status(404).json({ error: 'History not found' });
      
      return res.json({
        id: (h as any).id,
        clientId: (h as any).cliente_id,
        appointmentId: (h as any).agendamento_id,
        serviceId: (h as any).corte_id,
        notes: (h as any).observacoes,
        realizedAt: (h as any).realizado_em
      });
    } catch (error) {
      return res.status(500).json({ error: "Error fetching history record" });
    }
  },

  async create(req: Request, res: Response) {
    try {
      const { clientId, appointmentId, serviceId, notes, photos } = req.body;
      if (!clientId) return res.status(400).json({ error: "clientId is required" });

      const historico = await HistoricoCorte.create({ 
        cliente_id: clientId,
        agendamento_id: appointmentId || null,
        corte_id: serviceId || null,
        observacoes: notes || ""
      });
      
      if (photos && Array.isArray(photos) && photos.length > 0 && appointmentId) {
        for (const url of photos) {
          await FotoResultado.create({ url_foto: url, agendamento_id: appointmentId });
        }
      }
      
      return res.status(201).json({
        id: (historico as any).id,
        clientId: (historico as any).cliente_id,
        notes: (historico as any).observacoes
      });
    } catch (error) {
      return res.status(500).json({ error: "Error creating history record" });
    }
  }
};