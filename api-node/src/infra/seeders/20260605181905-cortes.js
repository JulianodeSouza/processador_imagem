'use strict';
const crypto = require('crypto');

/** @type {import('sequelize-cli').Migration} */
module.exports = {
  async up(queryInterface, Sequelize) {
    await queryInterface.bulkInsert('cortes', [
      {
        id: crypto.randomUUID(),
        nome: 'Corte Clássico',
        descricao: 'Corte tradicional com tesoura e máquina de acabamento.',
        preco: 50.00,
        criado_em: new Date()
      },
      {
        id: crypto.randomUUID(),
        nome: 'The Executive Ritual',
        descricao: 'Corte premium com toalha quente, barboterapia e massagem facial.',
        preco: 120.00,
        criado_em: new Date()
      },
      {
        id: crypto.randomUUID(),
        nome: 'Escultura de Barba + Spa',
        descricao: 'Alinhamento detalhado de barba com vapor de ozônio e hidratação.',
        preco: 70.00,
        criado_em: new Date()
      },
      {
        id: crypto.randomUUID(),
        nome: 'Análise de Imagem + Corte Visagista',
        descricao: 'Consultoria completa de imagem e corte projetado especificamente para as proporções do rosto.',
        preco: 150.00,
        criado_em: new Date()
      }
    ], {});
  },

  async down(queryInterface, Sequelize) {
    await queryInterface.bulkDelete('cortes', null, {});
  }
};