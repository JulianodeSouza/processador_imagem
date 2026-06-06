'use strict';

/** @type {import('sequelize-cli').Migration} */
module.exports = {
  async up(queryInterface, Sequelize) {
    await queryInterface.createTable('barbeiros', {
      id: {
        type: Sequelize.UUID,
        defaultValue: Sequelize.UUIDV4,
        primaryKey: true,
        allowNull: false
      },
      nome: { type: Sequelize.STRING, allowNull: false },
      telefone: { type: Sequelize.STRING },
      email: { type: Sequelize.STRING },
      ativo: { type: Sequelize.BOOLEAN, defaultValue: true },
      criado_em: {
        type: Sequelize.DATE,
        allowNull: false,
        defaultValue: Sequelize.fn('NOW')
      }
    });
  },
  async down(queryInterface) {
    await queryInterface.dropTable('barbeiros');
  }
};