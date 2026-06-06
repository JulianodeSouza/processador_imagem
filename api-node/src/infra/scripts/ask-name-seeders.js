const readSync = require("readline-sync");
const { execSync } = require("child_process");

const seedersName = readSync.question("Write seeders name: ");

// Adicionado o { stdio: 'inherit' } para que você consiga ver 
// o output de sucesso (a cor verde do sequelize) direto no seu terminal!
execSync(`npx sequelize-cli seed:generate --name ${seedersName}`, { stdio: 'inherit' });