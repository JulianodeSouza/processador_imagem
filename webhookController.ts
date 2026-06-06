import { Request, Response } from 'express';

// Adapte para o seu model Sequelize real, ex:
// import { ImageAnalysis } from '../../infra/models/ImageAnalysis';

export class WebhookController {
  /**
   * POST /api/webhook/image-result
   *
   * Chamado automaticamente pela API Python após processar a imagem.
   * Recebe o JSON com o resultado e salva no banco.
   *
   * Exemplo de body esperado:
   * {
   *   "image_id": "abc123",       // identificador da imagem
   *   "status": "aprovado",       // resultado da análise
   *   "confianca": 0.97,          // nível de confiança
   *   "detalhes": { ... }         // dados extras retornados pela Python
   * }
   */
  async receiveResult(req: Request, res: Response): Promise<void> {
    try {
      // 1. Valida o payload recebido
      const { image_id, status, confianca, detalhes } = req.body;

      if (!image_id || !status) {
        res.status(400).json({
          error: 'Payload inválido. Os campos image_id e status são obrigatórios.',
        });
        return;
      }

      // 2. (Opcional) Valida token secreto para garantir que veio da API Python
      const secret = req.headers['x-webhook-secret'];
      if (secret !== process.env.WEBHOOK_SECRET) {
        res.status(401).json({ error: 'Não autorizado.' });
        return;
      }

      // 3. Salva no banco via Sequelize
      // Descomente e adapte ao seu model:
      //
      // const record = await ImageAnalysis.create({
      //   image_id,
      //   status,
      //   confianca,
      //   detalhes: JSON.stringify(detalhes),
      // });

      console.log(`[Webhook] Resultado recebido para imagem ${image_id}:`, status);

      // 4. Responde 200 para a API Python saber que recebeu com sucesso
      res.status(200).json({
        message: 'Resultado recebido e salvo com sucesso.',
        image_id,
        status,
        // id: record.id, // descomente após configurar o model
      });
    } catch (error) {
      console.error('[WebhookController] Erro:', error);
      res.status(500).json({ error: 'Erro interno no servidor.' });
    }
  }

  /**
   * GET /api/webhook/image-result/:imageId
   *
   * Consulta o resultado de uma imagem já processada.
   */
  async getResult(req: Request, res: Response): Promise<void> {
    try {
      const { imageId } = req.params;

      // const record = await ImageAnalysis.findOne({ where: { image_id: imageId } });
      // if (!record) {
      //   res.status(404).json({ error: 'Resultado não encontrado para essa imagem.' });
      //   return;
      // }
      // res.status(200).json(record);

      // Placeholder — remova após configurar o model:
      res.status(200).json({
        image_id: imageId,
        message: 'Configure o model Sequelize para retornar o registro real.',
      });
    } catch (error) {
      console.error('[WebhookController] Erro ao buscar:', error);
      res.status(500).json({ error: 'Erro interno no servidor.' });
    }
  }
}
