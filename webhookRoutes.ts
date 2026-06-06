import { Router } from 'express';
import { WebhookController } from '../controllers/webhookController';

const router = Router();
const webhookController = new WebhookController();

// POST /api/webhook/image-result
// Chamado pela API Python com o resultado da análise
router.post('/image-result', (req, res) => webhookController.receiveResult(req, res));

// GET /api/webhook/image-result/:imageId
// Consulta o resultado de uma imagem específica
router.get('/image-result/:imageId', (req, res) => webhookController.getResult(req, res));

export default router;
