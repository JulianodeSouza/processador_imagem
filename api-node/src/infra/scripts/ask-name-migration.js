const readSync = require("readline-sync");
const { execSync } = require("child_process");

const migrationName = readSync.question("Write migration name: ");

// Adicionado o { stdio: 'inherit' } para que você consiga ver 
// o output de sucesso (a cor verde do sequelize) direto no seu terminal!
execSync(`npx sequelize-cli migration:generate --name ${migrationName}`, { stdio: 'inherit' });