"use client";

import { ReactNode, useState, useEffect, useCallback } from "react";
import { api } from "../../services/api";

interface Barber {
  id: string;
  name: string;
  phone: string;
  email: string;
  specialties: string[];
  rating: number;
  totalCuts: number;
  status: "Ativo" | "Inativo";
}

interface BarberAgenda {
  id: string;
  clientName: string;
  date: string;
  status: string;
  notes: string;
}

export default function EquipePage(): ReactNode {
  const [barbers, setBarbers] = useState<Barber[]>([]);
  const [selectedBarber, setSelectedBarber] = useState<Barber | null>(null);
  const [isLoadingList, setIsLoadingList] = useState(true);

  // Estados dos Modais
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isAgendaModalOpen, setIsAgendaModalOpen] = useState(false);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);

  // Estados de Cadastro
  const [newName, setNewName] = useState("");
  const [newPhone, setNewPhone] = useState("");
  const [newEmail, setNewEmail] = useState("");


  // ... seus outros states (newName, newPhone, etc)

  // Função para aplicar a máscara de telefone (Fixo ou Celular)
  const handlePhoneChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    let value = e.target.value.replace(/\D/g, ""); // Remove tudo o que não for número

    if (value.length <= 10) {
      // Máscara para telefone fixo: (XX) XXXX-XXXX
      value = value.replace(/^(\d{2})(\d)/g, "($1) $2");
      value = value.replace(/(\d{4})(\d)/, "$1-$2");
    } else {
      // Máscara para celular: (XX) XXXXX-XXXX
      value = value.replace(/^(\d{2})(\d)/g, "($1) $2");
      value = value.replace(/(\d{5})(\d)/, "$1-$2");
    }

    setNewPhone(value.substring(0, 15)); // Limita o tamanho máximo a 15 caracteres
  };

  // Estado da Agenda
  const [barberAgenda, setBarberAgenda] = useState<BarberAgenda[]>([]);

  const fetchBarbers = useCallback(async () => {
    try {
      setIsLoadingList(true);
      const response = await api.get<Barber[]>("/barbeiros");
      setBarbers(response.data || []);
    } catch (error) {
      console.error("Erro ao buscar barbeiros:", error);
    } finally {
      setIsLoadingList(false);
    }
  }, []);

  useEffect(() => {
    const loadData = async () => {
      try {
        setIsLoadingList(true);
        const response = await api.get<Barber[]>("/barbeiros");
        setBarbers(response.data || []);
      } catch (error) {
        console.error("Erro ao buscar barbeiros:", error);
      } finally {
        setIsLoadingList(false);
      }
    }
    loadData();
  }, [fetchBarbers]);

  const handleSaveBarber = async () => {
    if (!newName.trim()) return;
    try {
      const payload = { name: newName, phone: newPhone, email: newEmail };
      await api.post("/barbeiros", payload);
      setIsAddModalOpen(false);
      setNewName("");
      setNewPhone("");
      setNewEmail("");
      fetchBarbers();
    } catch (error) {
      console.error("Erro ao criar barbeiro:", error);
    }
  };

  const handleUpdate = async () => {
    if (!selectedBarber) return;
    try {
      // Traduzimos os dados para o backend (Inglês -> Português já é feito no Controller, 
      // aqui enviamos o objeto mapeado)
      await api.put(`/barbeiros/${selectedBarber.id}`, selectedBarber);
      setIsEditModalOpen(false);
      fetchBarbers();
    } catch (error) {
      console.error("Erro ao atualizar barbeiro:", error);
    }
  };

  const handleDelete = async () => {
    if (!selectedBarber) return;
    if (confirm(`Deseja realmente excluir o perfil de ${selectedBarber.name}?`)) {
      try {
        await api.delete(`/barbeiros/${selectedBarber.id}`);
        setIsEditModalOpen(false);
        setSelectedBarber(null);
        fetchBarbers();
      } catch (error) {
        console.error("Erro ao excluir barbeiro:", error);
      }
    }
  };

  const handleOpenAgenda = async (barber: Barber) => {
    try {
      const { data } = await api.get<BarberAgenda[]>(`/barbeiros/${barber.id}/agenda`);
      setBarberAgenda(data);
      setIsAgendaModalOpen(true);
    } catch (error) {
      console.error("Erro ao carregar agenda:", error);
    }
  };

  return (
    <div className="w-full bg-surface min-h-screen">
      <main className="lg:pl-72 pt-28 pb-20 px-6 lg:px-12 max-w-7xl mx-auto flex flex-col lg:flex-row gap-8">

        {/* LISTA DE PROFISSIONAIS */}
        <section className="w-full lg:w-1/3 flex flex-col">
          <div className="flex justify-between items-end mb-8">
            <div>
              <p className="text-secondary font-label mb-2 tracking-widest uppercase">Equipe</p>
              <h1 className="text-3xl font-headline font-bold text-on-surface">Profissionais</h1>
            </div>
            <button onClick={() => setIsAddModalOpen(true)} className="p-3 bg-secondary text-black rounded-xl hover:bg-secondary/90 transition-colors">
              <span className="material-symbols-outlined">person_add</span>
            </button>
          </div>

          <div className="flex flex-col gap-4">
            {isLoadingList ? (
              <p className="text-on-surface-variant animate-pulse font-label">Carregando equipe...</p>
            ) : (
              barbers.map(barber => (
                <button
                  key={barber.id}
                  onClick={() => setSelectedBarber(barber)}
                  className={`p-5 rounded-2xl border text-left transition-all ${selectedBarber?.id === barber.id ? 'bg-surface-container-high border-secondary/50' : 'bg-surface-container border-outline-variant/20 hover:border-secondary/30'}`}
                >
                  <div className="flex justify-between items-start">
                    <h3 className="font-headline font-bold text-lg text-on-surface">{barber.name}</h3>
                    {barber.rating > 0 && (
                      <div className="flex items-center gap-1 text-secondary">
                        <span className="material-symbols-outlined text-sm">star</span>
                        <span className="text-xs font-bold font-label">{barber?.rating?.toFixed(1)}</span>
                      </div>
                    )}
                  </div>
                  <p className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant mt-2 flex items-center gap-2">
                    <span className={`w-1.5 h-1.5 rounded-full ${barber.status === "Ativo" ? "bg-green-500" : "bg-red-500"}`}></span>
                    {barber.status}
                  </p>
                </button>
              ))
            )}
          </div>
        </section>

        {/* DETALHES DO PROFISSIONAL */}
        <section className="w-full lg:w-2/3">
          {selectedBarber ? (
            <div className="bg-surface-container border border-outline-variant/20 rounded-[2rem] p-8 min-h-[600px]">
              <div className="flex justify-between items-start mb-8 pb-8 border-b border-outline-variant/20">
                <div>
                  <h2 className="text-4xl font-headline font-bold text-on-surface">{selectedBarber.name}</h2>
                  <p className="text-on-surface-variant font-label mt-2 uppercase tracking-widest text-xs">
                    {selectedBarber.phone} • {selectedBarber.email || "E-mail não cadastrado"}
                  </p>
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={() => setIsEditModalOpen(true)}
                    className="px-6 py-2 bg-surface-container-high border border-outline-variant/30 rounded-xl text-[10px] font-label uppercase tracking-widest text-on-surface hover:bg-outline-variant/10 transition-colors"
                  >
                    Editar Perfil
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
                <div>
                  <h3 className="flex items-center gap-3 text-xl font-headline font-bold text-on-surface mb-8">
                    <span className="material-symbols-outlined text-secondary">content_cut</span>
                    Especialidades
                  </h3>
                  <div className="flex flex-wrap gap-2 mb-8">
                    {selectedBarber.specialties?.map((spec, i) => (
                      <span key={i} className="px-4 py-1.5 bg-surface-container-high rounded-full text-[10px] font-label uppercase tracking-widest text-on-surface-variant border border-outline-variant/20">
                        {spec}
                      </span>
                    )) || <p className="text-on-surface-variant text-sm font-label">Nenhuma especialidade definida.</p>}
                  </div>

                  <div className="bg-surface-container-high p-6 rounded-2xl border border-outline-variant/10 flex justify-between items-center">
                    <div>
                      <p className="text-[10px] uppercase tracking-widest text-secondary font-label mb-2">Desempenho</p>
                      <p className="text-2xl font-headline font-bold text-on-surface">{selectedBarber.totalCuts} Cortes</p>
                    </div>
                    <div className="text-right">
                      <p className="text-[10px] uppercase tracking-widest text-secondary font-label mb-2">Rating</p>
                      <p className="text-2xl font-headline font-bold text-on-surface flex items-center gap-2">
                        {selectedBarber?.rating?.toFixed(1)} <span className="material-symbols-outlined text-secondary">star</span>
                      </p>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="flex items-center gap-3 text-xl font-headline font-bold text-on-surface mb-8">
                    <span className="material-symbols-outlined text-secondary">calendar_today</span>
                    Agenda & Turnos
                  </h3>
                  <div className="p-6 bg-surface-container-high rounded-2xl border border-outline-variant/10 mb-6">
                    <p className="text-xs font-label text-on-surface-variant mb-4 uppercase tracking-widest">Disponibilidade</p>
                    <p className="text-on-surface font-headline text-lg font-bold">Terça a Sábado</p>
                    <p className="text-secondary font-label text-sm">09:00 - 19:00</p>
                  </div>
                  <button
                    onClick={() => handleOpenAgenda(selectedBarber)}
                    className="w-full py-4 bg-secondary/10 border border-secondary/20 text-secondary font-label text-[10px] uppercase tracking-widest rounded-2xl hover:bg-secondary/20 transition-all flex items-center justify-center gap-3"
                  >
                    <span className="material-symbols-outlined text-sm">event_note</span>
                    Ver Agenda Completa
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full min-h-[600px] flex flex-col items-center justify-center border border-dashed border-outline-variant/30 rounded-[2rem] text-on-surface-variant/40">
              <span className="material-symbols-outlined text-6xl mb-4">badge</span>
              <p className="font-label uppercase tracking-widest text-sm">Selecione um profissional para gerenciar</p>
            </div>
          )}
        </section>
      </main>

      {/* MODAL DE EDIÇÃO (CORRIGIDO) */}
      {isEditModalOpen && selectedBarber && (
        <div className="fixed inset-0 bg-black/80 z-[100] flex items-center justify-center p-4 backdrop-blur-md">
          <div className="bg-surface-container p-8 rounded-[2rem] w-full max-w-lg border border-outline-variant/30 shadow-2xl">
            <div className="flex items-center gap-4 mb-8">
              <div className="p-3 bg-secondary/10 rounded-2xl text-secondary">
                <span className="material-symbols-outlined text-3xl">edit_square</span>
              </div>
              <div>
                <h2 className="text-2xl font-headline font-bold text-on-surface">Editar Perfil</h2>
                <p className="text-on-surface-variant font-label text-xs uppercase tracking-widest">Atualizar dados do profissional</p>
              </div>
            </div>

            <div className="space-y-6">
              <div>
                <label className="block text-[10px] font-label text-secondary mb-2 uppercase tracking-widest">Nome do Profissional</label>
                <input
                  type="text"
                  value={selectedBarber.name}
                  onChange={e => setSelectedBarber({ ...selectedBarber, name: e.target.value })}
                  className="w-full bg-surface-container-high border border-outline-variant/30 text-on-surface rounded-2xl px-5 py-4 outline-none focus:border-secondary transition-colors font-label"
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-[10px] font-label text-secondary mb-2 uppercase tracking-widest">Telefone</label>
                  <input
                    type="text"
                    value={selectedBarber.phone}
                    onChange={e => setSelectedBarber({ ...selectedBarber, phone: e.target.value })}
                    className="w-full bg-surface-container-high border border-outline-variant/30 text-on-surface rounded-2xl px-5 py-4 outline-none focus:border-secondary transition-colors font-label"
                  />
                </div>
                <div>
                  <label className="block text-[10px] font-label text-secondary mb-2 uppercase tracking-widest">Status na Unidade</label>
                  <select
                    value={selectedBarber.status}
                    onChange={e => setSelectedBarber({ ...selectedBarber, status: e.target.value as "Ativo" | "Inativo" })}
                    className="w-full bg-surface-container-high border border-outline-variant/30 text-on-surface rounded-2xl px-5 py-4 outline-none focus:border-secondary transition-colors font-label appearance-none"
                  >
                    <option value="Ativo">Ativo</option>
                    <option value="Inativo">Inativo</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-[10px] font-label text-secondary mb-2 uppercase tracking-widest">E-mail</label>
                <input
                  type="email"
                  value={selectedBarber.email}
                  onChange={e => setSelectedBarber({ ...selectedBarber, email: e.target.value })}
                  className="w-full bg-surface-container-high border border-outline-variant/30 text-on-surface rounded-2xl px-5 py-4 outline-none focus:border-secondary transition-colors font-label"
                />
              </div>
            </div>

            <div className="flex justify-between items-center mt-12 pt-8 border-t border-outline-variant/20">
              <button
                onClick={handleDelete}
                className="text-red-500 font-label text-[10px] uppercase tracking-widest font-bold hover:underline"
              >
                Excluir Perfil
              </button>
              <div className="flex gap-4">
                <button
                  onClick={() => setIsEditModalOpen(false)}
                  className="px-6 py-3 rounded-2xl font-label text-[10px] text-on-surface-variant hover:bg-surface-container-high transition-colors uppercase tracking-widest"
                >
                  Cancelar
                </button>
                <button
                  onClick={handleUpdate}
                  className="px-8 py-3 rounded-2xl font-label text-[10px] bg-secondary text-black hover:bg-secondary/80 transition-colors uppercase tracking-widest font-bold"
                >
                  Salvar Alterações
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* MODAL DE CADASTRO (REESTILIZADO) */}
      {isAddModalOpen && (
        <div className="fixed inset-0 bg-black/80 z-[100] flex items-center justify-center p-4 backdrop-blur-md">
          <div className="bg-surface-container p-8 rounded-[2rem] w-full max-w-md border border-outline-variant/30 shadow-2xl">
            <h2 className="text-2xl font-headline font-bold text-on-surface mb-8">Novo Profissional</h2>
            <div className="space-y-6">
              <input type="text" value={newName} onChange={(e) => setNewName(e.target.value)} className="w-full bg-surface-container-high border border-outline-variant/30 text-on-surface rounded-2xl px-5 py-4 outline-none focus:border-secondary transition-colors font-label" placeholder="Nome Completo" />
              <input type="text" value={newPhone} onChange={handlePhoneChange} />
              <input
                type="text"
                value={newPhone}
                onChange={handlePhoneChange}
                className="w-full bg-surface-container-high border border-outline-variant/30 text-on-surface rounded-2xl px-5 py-4 outline-none focus:border-secondary transition-colors font-label" placeholder="Telefone"
              />
              <input type="email" value={newEmail} onChange={(e) => setNewEmail(e.target.value)} className="w-full bg-surface-container-high border border-outline-variant/30 text-on-surface rounded-2xl px-5 py-4 outline-none focus:border-secondary transition-colors font-label" placeholder="E-mail" />
            </div>
            <div className="flex justify-end gap-3 mt-8">
              <button onClick={() => setIsAddModalOpen(false)} className="px-6 py-3 text-on-surface-variant font-label text-[10px] uppercase tracking-widest">Cancelar</button>
              <button onClick={handleSaveBarber} className="px-8 py-3 bg-secondary text-black font-label text-[10px] font-bold rounded-2xl uppercase tracking-widest">Cadastrar</button>
            </div>
          </div>
        </div>
      )}

      {/* MODAL DE AGENDA (REESTILIZADO) */}
      {isAgendaModalOpen && (
        <div className="fixed inset-0 bg-black/80 z-[100] flex items-center justify-center p-4 backdrop-blur-md">
          <div className="bg-surface-container p-8 rounded-[2rem] w-full max-w-2xl border border-outline-variant/30 shadow-2xl max-h-[85vh] overflow-hidden flex flex-col">
            <h2 className="text-2xl font-headline font-bold text-on-surface mb-6">Agenda: {selectedBarber?.name}</h2>
            <div className="overflow-y-auto flex-1 pr-2">
              <table className="w-full text-left">
                <thead className="sticky top-0 bg-surface-container border-b border-outline-variant/20">
                  <tr className="text-[10px] font-label text-secondary uppercase tracking-widest">
                    <th className="py-4 px-2">Cliente</th>
                    <th className="py-4 px-2">Data</th>
                    <th className="py-4 px-2">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/10">
                  {barberAgenda.map((a) => (
                    <tr key={a.id} className="text-sm font-label text-on-surface-variant hover:bg-outline-variant/5 transition-colors">
                      <td className="py-4 px-2 font-bold text-on-surface">{a.clientName}</td>
                      <td className="py-4 px-2">{new Date(a.date).toLocaleDateString()}</td>
                      <td className="py-4 px-2 italic">{a.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <button onClick={() => setIsAgendaModalOpen(false)} className="mt-8 w-full py-4 bg-surface-container-high text-on-surface font-label text-[10px] font-bold uppercase tracking-widest rounded-2xl border border-outline-variant/20">
              Fechar Visualização
            </button>
          </div>
        </div>
      )}
    </div>
  );
}