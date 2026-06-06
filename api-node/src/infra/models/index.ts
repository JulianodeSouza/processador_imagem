import { DataTypes } from "sequelize";
import { sequelize } from "../config/database";

export const Cliente = sequelize.define(
  "Cliente",
  {
    id: {
      type: DataTypes.UUID,
      defaultValue: DataTypes.UUIDV4,
      primaryKey: true,
    },
    nome: { type: DataTypes.STRING, allowNull: false },
    telefone: { type: DataTypes.STRING },
    email: { type: DataTypes.STRING },
    criado_em: { type: DataTypes.DATE, defaultValue: DataTypes.NOW },
  },
  {
    tableName: "clientes",
    timestamps: false,
  },
);

export const Barbeiro = sequelize.define(
  "Barbeiro",
  {
    id: {
      type: DataTypes.UUID,
      defaultValue: DataTypes.UUIDV4,
      primaryKey: true,
    },
    nome: { type: DataTypes.STRING, allowNull: false },
    telefone: { type: DataTypes.STRING },
    email: { type: DataTypes.STRING },
    ativo: { type: DataTypes.BOOLEAN, defaultValue: true },
    criado_em: { type: DataTypes.DATE, defaultValue: DataTypes.NOW },
  },
  {
    tableName: "barbeiros",
    timestamps: false,
  },
);

export const Corte = sequelize.define(
  "Corte",
  {
    id: {
      type: DataTypes.UUID,
      defaultValue: DataTypes.UUIDV4,
      primaryKey: true,
    },
    nome: { type: DataTypes.STRING, allowNull: false },
    descricao: { type: DataTypes.STRING },
    preco: { type: DataTypes.DECIMAL(10, 2) },
    criado_em: { type: DataTypes.DATE, defaultValue: DataTypes.NOW },
  },
  {
    tableName: "cortes",
    timestamps: false,
  },
);

export const Agendamento = sequelize.define(
  "Agendamento",
  {
    id: {
      type: DataTypes.UUID,
      defaultValue: DataTypes.UUIDV4,
      primaryKey: true,
    },
    cliente_id: { type: DataTypes.UUID, allowNull: false },
    barbeiro_id: { type: DataTypes.UUID, allowNull: false },
    data_hora: { type: DataTypes.DATE, allowNull: false },
    status: { type: DataTypes.STRING, allowNull: false },
    criado_em: { type: DataTypes.DATE, defaultValue: DataTypes.NOW },
  },
  {
    tableName: "agendamentos",
    timestamps: false,
  },
);

export const HistoricoCorte = sequelize.define(
  "HistoricoCorte",
  {
    id: {
      type: DataTypes.UUID,
      defaultValue: DataTypes.UUIDV4,
      primaryKey: true,
    },
    cliente_id: { type: DataTypes.UUID, allowNull: false },
    agendamento_id: { type: DataTypes.UUID },
    corte_id: { type: DataTypes.UUID },
    observacoes: { type: DataTypes.TEXT },
    realizado_em: { type: DataTypes.DATE, defaultValue: DataTypes.NOW },
  },
  {
    tableName: "historico_cortes",
    timestamps: false,
  },
);

export const FotoResultado = sequelize.define(
  "FotoResultado",
  {
    id: {
      type: DataTypes.UUID,
      defaultValue: DataTypes.UUIDV4,
      primaryKey: true,
    },
    agendamento_id: { type: DataTypes.UUID, allowNull: false },
    url_foto: { type: DataTypes.STRING, allowNull: false },
    enviado_em: { type: DataTypes.DATE, defaultValue: DataTypes.NOW },
  },
  {
    tableName: "fotos_resultado",
    timestamps: false,
  },
);

export const RecomendacaoVisagismo = sequelize.define(
  "RecomendacaoVisagismo",
  {
    id: {
      type: DataTypes.UUID,
      defaultValue: DataTypes.UUIDV4,
      primaryKey: true,
    },
    cliente_id: {
      type: DataTypes.UUID,
      allowNull: false,
    },
    formato_rosto_identificado: {
      type: DataTypes.STRING,
      allowNull: false,
    },
    corte_sugerido_id: {
      type: DataTypes.UUID,
    },
    justificativa: {
      type: DataTypes.TEXT,
    },
    criado_em: {
      type: DataTypes.DATE,
      defaultValue: DataTypes.NOW,
    },
  },
  {
    tableName: "recomendacoes_visagismo",
    timestamps: false,
  },
);

// ========================
// RELACIONAMENTOS (Associações)
// ========================

// Agendamento pertence a um Cliente e a um Barbeiro
Cliente.hasMany(Agendamento, { foreignKey: "cliente_id" });
Agendamento.belongsTo(Cliente, { foreignKey: "cliente_id" });

Barbeiro.hasMany(Agendamento, { foreignKey: "barbeiro_id" });
Agendamento.belongsTo(Barbeiro, { foreignKey: "barbeiro_id" });

// Histórico de Cortes
Cliente.hasMany(HistoricoCorte, { foreignKey: "cliente_id" });
HistoricoCorte.belongsTo(Cliente, { foreignKey: "cliente_id" });

Agendamento.hasOne(HistoricoCorte, { foreignKey: "agendamento_id" });
HistoricoCorte.belongsTo(Agendamento, { foreignKey: "agendamento_id" });

Corte.hasMany(HistoricoCorte, { foreignKey: "corte_id" });
HistoricoCorte.belongsTo(Corte, { foreignKey: "corte_id" });

// Fotos de Resultados (Ligadas ao Agendamento conforme a tua migration)
Agendamento.hasMany(FotoResultado, {
  foreignKey: "agendamento_id",
  as: "photos",
});
FotoResultado.belongsTo(Agendamento, { foreignKey: "agendamento_id" });

export async function syncDatabase() {
  await sequelize.sync();
  console.log("Banco de dados sincronizado!");
}

// Relacionamentos da Recomendação de Visagismo
Cliente.hasMany(RecomendacaoVisagismo, { foreignKey: "cliente_id" });
RecomendacaoVisagismo.belongsTo(Cliente, { foreignKey: "cliente_id" });

Corte.hasMany(RecomendacaoVisagismo, { foreignKey: "corte_sugerido_id" });
RecomendacaoVisagismo.belongsTo(Corte, { foreignKey: "corte_sugerido_id" });
