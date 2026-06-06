require('dotenv').config(); // Puxa as variáveis do seu arquivo .env

module.exports = {
  development: {
    username: process.env.DB_USER || "avnadmin",
    password: String(process.env.DB_PASS), 
    database: process.env.DB_NAME || "defaultdb",
    host: process.env.DB_HOST || "barbershopdb-testebancodedadosaula.d.aivencloud.com",
    port: Number(process.env.DB_PORT) || 5432,
    dialect: "postgres",
    dialectOptions: {
      ssl: {
        require: true,
        rejectUnauthorized: false
      }
    }
  }
};