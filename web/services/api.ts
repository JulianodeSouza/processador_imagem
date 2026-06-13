import axios from "axios";

// O baseURL deve apontar para o seu backend Node.js. 
// Você pode configurar isso em um arquivo .env.local na raiz do projeto (NEXT_PUBLIC_API_URL=http://localhost:3000)
export const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:3333",
  timeout: 60000,
});

// Interceptor para injetar tokens de autenticação futuramente, se necessário
api.interceptors.request.use(
  (config) => {
    // const token = localStorage.getItem('@Barbershop:token');
    // if (token) { config.headers.Authorization = `Bearer ${token}`; }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);