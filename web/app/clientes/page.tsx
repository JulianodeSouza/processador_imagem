"use client";

import { ReactNode, useState, useEffect } from "react";
import { api } from "../../services/api";
import ImageUploadModule from "@/components/ImageUploadModule";

interface ClientPhoto {
  id: string;
  url: string;
  date: string;
}

interface CutHistory {
  id: string;
  date: string;
  service: string;
  barberName: string;
  photos: ClientPhoto[];
}

interface VisagismProfile {
  faceShape: string;
  skinTone: string;
  stylePreference: string[];
  recommendedCuts: string[];
}

interface Client {
  id: string;
  name: string;
  phone: string;
  email: string;
  history: CutHistory[];
  visagism?: VisagismProfile;
}

export default function ClientesPage(): ReactNode {
  const [clients, setClients] = useState<Client[]>([]);
  const [selectedClient, setSelectedClient] = useState<Client | null>(null);
  const [isLoadingList, setIsLoadingList] = useState(true);

  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [newPhone, setNewPhone] = useState("");

  // Buscar todos os clientes da API ao montar o componente
  useEffect(() => {
    async function fetchClients() {
      try {
        const response = await api.get("/clientes");
        setClients(response.data || []);
      } catch (error) {
        console.error("Erro ao buscar clientes:", error);
      } finally {
        setIsLoadingList(false);
      }
    }
    fetchClients();
  }, []);

  const handleSaveClient = async () => {
    if (!newName.trim()) return;
    try {
      const payload = { name: newName, phone: newPhone };
      const response = await api.post("/clientes", payload);
      // Atualiza a lista com o novo cliente retornado da API
      setClients((prev) => [...prev, { ...response.data, history: [] }]);
      setIsAddModalOpen(false);
      setNewName("");
      setNewPhone("");
    } catch (error) {
      console.error("Erro ao criar cliente:", error);
    }
  };

  const handleSelectClient = async (client: Client) => {
    try {
      // Opcional: Se a listagem não traz os detalhes, busca da API os dados completos
      const response = await api.get(`/clientes/${client.id}`);
      setSelectedClient(response.data);
    } catch (error) {
      console.error(
        "Erro ao buscar detalhes do cliente, usando dados locais.",
        error,
      );
      setSelectedClient(client);
    }
  };

  return (
    <div className="w-full bg-surface min-h-screen">
      <main className="lg:pl-72 pt-28 pb-20 px-6 lg:px-12 max-w-7xl mx-auto flex flex-col lg:flex-row gap-8">
        {/* Coluna Esquerda: Lista de Clientes */}
        <section className="w-full lg:w-1/3 flex flex-col">
          <div className="flex justify-between items-end mb-8">
            <div>
              <p className="text-primary font-label mb-2 tracking-widest uppercase">
                Diretório
              </p>
              <h1 className="text-3xl font-headline font-bold text-on-surface">
                Clientes
              </h1>
            </div>
            <button
              onClick={() => setIsAddModalOpen(true)}
              className="p-3 bg-primary text-black rounded-xl hover:bg-primary/90 transition-colors"
            >
              <span className="material-symbols-outlined">person_add</span>
            </button>
          </div>

          <div className="flex flex-col gap-4">
            {isLoadingList ? (
              <p className="text-on-surface-variant animate-pulse">
                Carregando...
              </p>
            ) : clients.length === 0 ? (
              <p className="text-on-surface-variant">
                Nenhum cliente cadastrado.
              </p>
            ) : (
              clients.map((client) => (
                <button
                  key={client.id}
                  onClick={() => handleSelectClient(client)}
                  className={`p-5 rounded-2xl border text-left transition-all ${selectedClient?.id === client.id ? "bg-surface-container-high border-primary/50" : "bg-surface-container border-outline-variant/20 hover:border-primary/30"}`}
                >
                  <h3 className="font-headline font-bold text-lg text-on-surface">
                    {client.name}
                  </h3>
                  <p className="font-label text-xs text-on-surface-variant mt-1">
                    {client.phone}
                  </p>
                </button>
              ))
            )}
          </div>
        </section>

        {/* Coluna Direita: Ficha Técnica do Cliente */}
        <section className="w-full lg:w-2/3">
          {selectedClient ? (
            <div className="bg-surface-container border border-outline-variant/20 rounded-[2rem] p-8 min-h-[600px]">
              <div className="flex justify-between items-start mb-8 pb-8 border-b border-outline-variant/20">
                <div>
                  <h2 className="text-3xl font-headline font-bold text-on-surface">
                    {selectedClient.name}
                  </h2>
                  <p className="text-on-surface-variant font-label mt-2">
                    {selectedClient.phone} •{" "}
                    {selectedClient.email || "Sem email"}
                  </p>
                </div>

              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Módulo de Visagismo e Preferências */}
                <div>
                  <h3 className="flex items-center gap-2 text-xl font-headline font-bold text-on-surface mb-6">
                    <span className="material-symbols-outlined text-primary">
                      face
                    </span>
                    Perfil Visagista
                  </h3>

                  <div className="bg-primary/10 border border-primary/20 p-4 rounded-xl">
                    <p className="text-[10px] uppercase tracking-widest text-primary font-label mb-2">
                      Análise por IA
                    </p>

                    <ImageUploadModule
                      clientId={selectedClient.id}
                      onAnalysisComplete={(result) => {
                        alert(
                          `Análise concluída! Formato: ${result.faceShape}`,
                        );
                        // Aqui você pode atualizar o estado do cliente localmente
                      }}
                    />

                    <ul className="list-disc list-inside text-sm text-on-surface space-y-1 mt-4">
                      {selectedClient.visagism?.recommendedCuts?.map(
                        (cut, i) => <li key={i}>{cut}</li>,
                      ) || <li>Aguardando análise de imagem...</li>}
                    </ul>
                  </div>
                </div>

                {/* Histórico e Fotos */}
                <div>
                  <h3 className="flex items-center gap-2 text-xl font-headline font-bold text-on-surface mb-6">
                    <span className="material-symbols-outlined text-secondary">
                      history
                    </span>
                    Histórico e Resultados
                  </h3>

                  <div className="space-y-6">
                    {selectedClient.history &&
                      selectedClient.history.length > 0 ? (
                      selectedClient.history.map((record) => (
                        <div
                          key={record.id}
                          className="relative pl-6 border-l-2 border-outline-variant/20"
                        >
                          <div className="absolute w-3 h-3 bg-secondary rounded-full -left-[7px] top-1"></div>
                          <p className="text-xs font-label text-secondary mb-1">
                            {new Date(record.date).toLocaleDateString("pt-BR")}
                          </p>
                          <p className="font-headline font-bold text-on-surface">
                            {record.service}
                          </p>
                          <p className="text-xs text-on-surface-variant mt-1 mb-3">
                            Atendido por: {record.barberName}
                          </p>

                          <div className="flex gap-2">
                            {record.photos?.map((photo) => (
                              <img
                                key={photo.id}
                                src={photo.url}
                                alt="Resultado do corte"
                                className="w-16 h-16 rounded-lg object-cover border border-outline-variant/30 cursor-pointer hover:opacity-80 transition-opacity"
                              />
                            ))}
                            <button className="w-16 h-16 rounded-lg border border-dashed border-outline-variant/50 flex items-center justify-center text-on-surface-variant hover:text-primary hover:border-primary transition-colors">
                              <span className="material-symbols-outlined">
                                add_a_photo
                              </span>
                            </button>
                          </div>
                        </div>
                      ))
                    ) : (
                      <p className="text-sm text-on-surface-variant">
                        Nenhum histórico registrado.
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full min-h-[600px] flex flex-col items-center justify-center border border-dashed border-outline-variant/20 rounded-[2rem] text-on-surface-variant">
              <span className="material-symbols-outlined text-4xl mb-4 opacity-50">
                badge
              </span>
              <p>Selecione um cliente para visualizar a ficha técnica.</p>
            </div>
          )}
        </section>
      </main>

      {/* Modal de Cadastro */}
      {isAddModalOpen && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4 backdrop-blur-sm">
          <div className="bg-surface-container p-8 rounded-[2rem] w-full max-w-md border border-outline-variant/20 shadow-2xl">
            <h2 className="text-2xl font-headline font-bold text-on-surface mb-6">
              Novo Cliente
            </h2>
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-label text-on-surface-variant mb-2 uppercase tracking-wider">
                  Nome Completo
                </label>
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  className="w-full bg-surface-container-high border border-outline-variant/30 text-on-surface rounded-xl px-4 py-3 outline-none focus:border-primary transition-colors"
                  placeholder="Ex: Carlos Oliveira"
                />
              </div>
              <div>
                <label className="block text-xs font-label text-on-surface-variant mb-2 uppercase tracking-wider">
                  Telefone
                </label>
                <input
                  type="text"
                  value={newPhone}
                  onChange={(e) => setNewPhone(e.target.value)}
                  className="w-full bg-surface-container-high border border-outline-variant/30 text-on-surface rounded-xl px-4 py-3 outline-none focus:border-primary transition-colors"
                  placeholder="(00) 00000-0000"
                />
              </div>
            </div>
            <div className="flex justify-end gap-3 mt-8">
              <button
                onClick={() => setIsAddModalOpen(false)}
                className="px-6 py-2.5 rounded-xl font-label text-xs text-on-surface-variant hover:bg-surface-container-high transition-colors uppercase tracking-wider"
              >
                Cancelar
              </button>
              <button
                onClick={handleSaveClient}
                disabled={!newName.trim()}
                className="px-6 py-2.5 rounded-xl font-label text-xs bg-primary text-black hover:bg-primary/90 disabled:opacity-50 transition-colors uppercase tracking-wider"
              >
                Salvar Cliente
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
