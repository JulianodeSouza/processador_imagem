"use client";

import { ReactNode, useState, useEffect, useMemo } from "react";
import { api } from "../../services/api";

type AppointmentStatus = "confirmado" | "em_andamento" | "concluido" | "livre";

interface AppointmentData {
  id: string;
  date: string;
  time: string;
  status: AppointmentStatus;
  clientName?: string;
  service?: string;
  duration?: string;
  notes?: string;
  avatarSrc?: string;
  isBookable?: boolean; // Nova propriedade para o frontend
}

// 1. Gera a jornada padrão dinamicamente (09:00 às 17:00)
const gerarJornada = () => {
  const times = [];
  for (let i = 9; i < 18; i++) {
    times.push(`${i.toString().padStart(2, "0")}:00`);
  }
  return times;
};

const DAILY_TIMES = gerarJornada();
const formatDateStr = (d: Date) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;

export default function AgendaPage(): ReactNode {
  const [selectedDate, setSelectedDate] = useState<Date>(new Date());
  const [appointments, setAppointments] = useState<AppointmentData[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Estados dos Modais
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isRescheduleModalOpen, setIsRescheduleModalOpen] = useState(false);
  
  // Estados de Formulário
  const [selectedTimeSlot, setSelectedTimeSlot] = useState("");
  const [selectedAppointmentId, setSelectedAppointmentId] = useState("");
  const [newClientName, setNewClientName] = useState("");
  const [newService, setNewService] = useState("");
  const [newRescheduleTime, setNewRescheduleTime] = useState("");

  const currentDateStr = formatDateStr(selectedDate);
  const displayDay = selectedDate.toLocaleDateString("pt-BR", { day: "2-digit", month: "long" });
  const displayWeekday = selectedDate.toLocaleDateString("pt-BR", { weekday: "long" });

  // Buscar Agendamentos da API
  useEffect(() => {
    async function fetchAppointments() {
      setIsLoading(true);
      try {
        const response = await api.get(`/agendamentos?data=${currentDateStr}`);
        setAppointments(response.data || []);
      } catch (error) {
        console.error("Erro ao buscar agendamentos:", error);
        setAppointments([]);
      } finally {
        setIsLoading(false);
      }
    }
    fetchAppointments();
  }, [currentDateStr]);

  const handlePrevDay = () => {
    const newDate = new Date(selectedDate);
    newDate.setDate(newDate.getDate() - 1);
    setSelectedDate(newDate);
  };

  const handleNextDay = () => {
    const newDate = new Date(selectedDate);
    newDate.setDate(newDate.getDate() + 1);
    setSelectedDate(newDate);
  };

  // 2. Aplica as validações na construção da agenda
  const dailySchedule = useMemo(() => {
    const agora = new Date();
    const hojeStr = formatDateStr(agora);
    const isToday = currentDateStr === hojeStr;
    const isPastDay = currentDateStr < hojeStr;

    return DAILY_TIMES.map((time) => {
      const apt = appointments.find((a) => a.time === time);
      if (apt) return apt;
      
      // Validação RN01 (Antecedência de 1 Hora)
      let isBookable = true;
      if (isPastDay) {
        isBookable = false;
      } else if (isToday) {
        const [hora] = time.split(':');
        const horaSlot = new Date();
        horaSlot.setHours(parseInt(hora, 10), 0, 0, 0);
        
        const diferencaHoras = (horaSlot.getTime() - agora.getTime()) / (1000 * 60 * 60);
        if (diferencaHoras < 1) {
          isBookable = false;
        }
      }

      return {
        id: `free-${currentDateStr}-${time}`,
        date: currentDateStr,
        time,
        status: "livre" as AppointmentStatus,
        isBookable
      };
    });
  }, [appointments, currentDateStr]);

  // 3. Filtra horários disponíveis considerando a regra de 1 hora para o Modal de Remarcar
  const availableTimesForToday = useMemo(() => {
    const agora = new Date();
    const hojeStr = formatDateStr(agora);
    const isToday = currentDateStr === hojeStr;

    return DAILY_TIMES.filter(time => {
      // Se já tem agendamento, não tá disponível
      if (appointments.some((a) => a.time === time)) return false;

      // Se for hoje, tem que respeitar 1h de antecedência (RN01)
      if (isToday) {
        const [hora] = time.split(':');
        const horaSlot = new Date();
        horaSlot.setHours(parseInt(hora, 10), 0, 0, 0);
        
        const diferencaHoras = (horaSlot.getTime() - agora.getTime()) / (1000 * 60 * 60);
        return diferencaHoras >= 1;
      }
      
      // Se for dia passado, retorna falso
      if (currentDateStr < hojeStr) return false;

      return true;
    });
  }, [appointments, currentDateStr]);

  const handleOpenAddModal = (time: string, isBookable: boolean) => {
    if (!isBookable) return;
    setSelectedTimeSlot(time);
    setNewClientName("");
    setNewService("Corte Clássico");
    setIsAddModalOpen(true);
  };

  const handleSaveAppointment = async () => {
    if (!newClientName.trim()) return;
    try {
      const payload = {
        date: currentDateStr,
        time: selectedTimeSlot,
        clientName: newClientName,
        service: newService,
        status: "confirmado"
      };
      const response = await api.post("/agendamentos", payload);
      setAppointments((prev) => [...prev, response.data]);
      setIsAddModalOpen(false);
    } catch (error) {
      console.error("Erro ao criar agendamento:", error);
    }
  };

  const handleCancelAppointment = async (id: string) => {
    if (confirm("Tem certeza que deseja cancelar este agendamento?")) {
      try {
        await api.delete(`/agendamentos/${id}`);
        setAppointments((prev) => prev.filter((apt) => apt.id !== id));
      } catch (error) {
        console.error("Erro ao cancelar agendamento:", error);
      }
    }
  };

  const handleOpenRescheduleModal = (id: string) => {
    setSelectedAppointmentId(id);
    setNewRescheduleTime("");
    setIsRescheduleModalOpen(true);
  };

  const handleSaveReschedule = async () => {
    if (!newRescheduleTime) return;
    try {
      const response = await api.put(`/agendamentos/${selectedAppointmentId}`, { time: newRescheduleTime });
      setAppointments((prev) => prev.map((apt) => apt.id === selectedAppointmentId ? response.data : apt));
      setIsRescheduleModalOpen(false);
    } catch (error) {
      console.error("Erro ao remarcar agendamento:", error);
    }
  };

  const getStatusStyles = (status: AppointmentStatus) => {
    switch (status) {
      case "em_andamento":
        return { chipBg: "bg-primary/10", chipText: "text-primary", dot: "bg-primary shadow-[0_0_8px_rgba(129,236,255,0.8)] animate-pulse", label: "Em Cadeira", cardBorder: "border-primary/30" };
      case "confirmado":
        return { chipBg: "bg-secondary/10", chipText: "text-secondary", dot: "bg-secondary shadow-[0_0_8px_rgba(254,183,0,0.8)]", label: "Confirmado", cardBorder: "border-outline-variant/20" };
      case "concluido":
        return { chipBg: "bg-surface-container-high", chipText: "text-on-surface-variant", dot: "bg-outline", label: "Finalizado", cardBorder: "border-outline-variant/10", opacity: "opacity-60" };
      default:
        return { chipBg: "bg-transparent", chipText: "text-on-surface-variant", dot: "bg-transparent", label: "", cardBorder: "border-transparent" };
    }
  };

  return (
    <div className="w-full bg-surface min-h-screen relative">
      <main className="lg:pl-72 pt-28 pb-20">
        <div className="px-6 lg:px-12 max-w-5xl mx-auto">
          
          <section className="mb-14 flex flex-col md:flex-row justify-between items-start md:items-end gap-6">
            <div>
              <p className="text-primary font-label mb-2 tracking-widest uppercase">Sua Linha do Tempo</p>
              <h1 className="text-4xl lg:text-5xl font-headline font-bold text-on-surface">Agenda do Atelier</h1>
            </div>

            <div className="flex items-center gap-4 bg-surface-container p-2 rounded-2xl border border-outline-variant/20">
              <button onClick={handlePrevDay} className="p-2 rounded-xl hover:bg-surface-container-high text-on-surface transition-colors">
                <span className="material-symbols-outlined">chevron_left</span>
              </button>
              <div className="px-4 flex flex-col items-center min-w-[120px]">
                <span className="font-headline font-bold text-lg capitalize">{displayDay}</span>
                <span className="font-label text-[10px] text-on-surface-variant capitalize">{displayWeekday}</span>
              </div>
              <button onClick={handleNextDay} className="p-2 rounded-xl hover:bg-surface-container-high text-on-surface transition-colors">
                <span className="material-symbols-outlined">chevron_right</span>
              </button>
            </div>
          </section>

          {isLoading ? (
            <div className="flex justify-center items-center py-20">
              <p className="text-on-surface-variant animate-pulse">Carregando agendamentos...</p>
            </div>
          ) : (
            <section className="relative">
              <div className="absolute left-[39px] md:left-[47px] top-4 bottom-4 w-px bg-gradient-to-b from-surface via-outline-variant/20 to-surface"></div>
              <div className="space-y-6">
                {dailySchedule.map((slot) => {
                  const isFree = slot.status === "livre";
                  const styles = getStatusStyles(slot.status);

                  return (
                    <div key={slot.id} className="relative flex items-start gap-6 group">
                      <div className="w-20 md:w-24 pt-4 flex flex-col items-end flex-shrink-0 relative z-10">
                        <span className={`font-headline text-xl md:text-2xl font-bold ${isFree ? "text-on-surface-variant/50" : "text-on-surface"}`}>{slot.time}</span>
                        {!isFree && <span className="font-label text-[10px] text-on-surface-variant mt-1">{slot.duration || '45 min'}</span>}
                      </div>

                      <div className="relative z-10 pt-5">
                        <div className={`w-3 h-3 rounded-full border-[2px] border-surface flex items-center justify-center ${isFree ? 'bg-outline-variant/30' : styles.dot}`}></div>
                      </div>

                      <div className="flex-1 min-w-0">
                        {isFree ? (
                          <button 
                            onClick={() => handleOpenAddModal(slot.time, !!slot.isBookable)} 
                            disabled={!slot.isBookable}
                            className={`w-full text-left p-6 rounded-[1.5rem] border border-dashed flex justify-between items-center transition-all
                              ${slot.isBookable 
                                ? "border-outline-variant/30 bg-surface hover:bg-surface-container/50 group-hover:border-primary/40 cursor-pointer" 
                                : "border-outline-variant/10 bg-transparent opacity-40 cursor-not-allowed"}`}
                          >
                            <span className={`font-label transition-colors ${slot.isBookable ? "text-on-surface-variant group-hover:text-primary" : "text-on-surface-variant/50"}`}>
                              {slot.isBookable ? "Horário Disponível" : "Horário Indisponível (Expirado)"}
                            </span>
                            {slot.isBookable && (
                              <span className="material-symbols-outlined text-outline-variant group-hover:text-primary transition-colors">add_circle</span>
                            )}
                          </button>
                        ) : (
                          <div className={`p-6 rounded-[1.5rem] bg-surface-container border ${styles.cardBorder} shadow-[0_0_30px_rgba(0,0,0,0.2)] transition-all hover:-translate-y-1 ${styles.opacity || ''}`}>
                            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-4">
                              <div className="flex items-center gap-4">
                                {slot.avatarSrc ? (
                                  <img src={slot.avatarSrc} alt={slot.clientName} className="w-12 h-12 rounded-full grayscale opacity-80" />
                                ) : (
                                  <div className="w-12 h-12 rounded-full bg-surface-container-high flex items-center justify-center border border-outline-variant/20">
                                    <span className="material-symbols-outlined text-on-surface-variant">person</span>
                                  </div>
                                )}
                                <div>
                                  <h3 className="font-headline font-bold text-xl text-on-surface">{slot.clientName}</h3>
                                  <p className="font-label text-xs text-on-surface-variant mt-1">{slot.service}</p>
                                </div>
                              </div>
                              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${styles.chipBg}`}>
                                <span className={`w-1.5 h-1.5 rounded-full ${styles.dot}`}></span>
                                <span className={`font-label text-[9px] font-bold ${styles.chipText}`}>{styles.label}</span>
                              </div>
                            </div>
                            {slot.notes && (
                              <div className="mt-4 pt-4 border-t border-outline-variant/10">
                                <p className="font-body text-sm text-on-surface-variant flex items-start gap-2">
                                  <span className="material-symbols-outlined text-[16px] mt-0.5 text-secondary/70">sticky_note_2</span>
                                  {slot.notes}
                                </p>
                              </div>
                            )}
                            {slot.status !== "concluido" && (
                              <div className="mt-6 flex gap-3">
                                <button onClick={() => handleCancelAppointment(slot.id)} className="flex-1 bg-red-500/10 hover:bg-red-500/20 text-red-400 font-label text-[10px] py-2.5 rounded-xl transition-colors uppercase tracking-wider">
                                  Cancelar
                                </button>
                                <button onClick={() => handleOpenRescheduleModal(slot.id)} className="flex-1 bg-surface-container-high hover:bg-outline-variant/30 text-on-surface font-label text-[10px] py-2.5 rounded-xl transition-colors uppercase tracking-wider">
                                  Remarcar
                                </button>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>
          )}
        </div>
      </main>

      {/* MODAL: Novo Agendamento */}
      {isAddModalOpen && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4 backdrop-blur-sm">
          <div className="bg-surface-container p-8 rounded-[2rem] w-full max-w-md border border-outline-variant/20 shadow-2xl">
            <h2 className="text-2xl font-headline font-bold text-on-surface mb-2">Novo Agendamento</h2>
            <p className="text-on-surface-variant font-label text-sm mb-6">Para o horário das <strong className="text-primary">{selectedTimeSlot}</strong></p>
            
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-label text-on-surface-variant mb-2 uppercase tracking-wider">Nome do Cliente</label>
                <input type="text" value={newClientName} onChange={(e) => setNewClientName(e.target.value)} className="w-full bg-surface-container-high border border-outline-variant/30 text-on-surface rounded-xl px-4 py-3 outline-none focus:border-primary transition-colors" placeholder="Ex: João Silva" />
              </div>
              <div>
                <label className="block text-xs font-label text-on-surface-variant mb-2 uppercase tracking-wider">Serviço</label>
                <select value={newService} onChange={(e) => setNewService(e.target.value)} className="w-full bg-surface-container-high border border-outline-variant/30 text-on-surface rounded-xl px-4 py-3 outline-none focus:border-primary transition-colors appearance-none">
                  <option value="Corte Clássico">Corte Clássico</option>
                  <option value="Escultura de Barba + Spa">Escultura de Barba + Spa</option>
                  <option value="The Executive Ritual">The Executive Ritual</option>
                  <option value="Análise de Imagem + Corte Visagista">Análise de Imagem + Corte Visagista</option>
                </select>
              </div>
            </div>

            <div className="flex justify-end gap-3 mt-8">
              <button onClick={() => setIsAddModalOpen(false)} className="px-6 py-2.5 rounded-xl font-label text-xs text-on-surface-variant hover:bg-surface-container-high transition-colors uppercase tracking-wider">Voltar</button>
              <button onClick={handleSaveAppointment} disabled={!newClientName.trim()} className="px-6 py-2.5 rounded-xl font-label text-xs bg-primary text-black hover:bg-primary/90 disabled:opacity-50 transition-colors uppercase tracking-wider">Confirmar</button>
            </div>
          </div>
        </div>
      )}

      {/* MODAL: Remarcar */}
      {isRescheduleModalOpen && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4 backdrop-blur-sm">
          <div className="bg-surface-container p-8 rounded-[2rem] w-full max-w-md border border-outline-variant/20 shadow-2xl">
            <h2 className="text-2xl font-headline font-bold text-on-surface mb-2">Remarcar Horário</h2>
            <p className="text-on-surface-variant font-label text-sm mb-6">Escolha um novo horário disponível para hoje</p>
            
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-label text-on-surface-variant mb-2 uppercase tracking-wider">Novo Horário</label>
                <select value={newRescheduleTime} onChange={(e) => setNewRescheduleTime(e.target.value)} className="w-full bg-surface-container-high border border-outline-variant/30 text-on-surface rounded-xl px-4 py-3 outline-none focus:border-primary transition-colors appearance-none">
                  <option value="" disabled>Selecione um horário...</option>
                  {availableTimesForToday.map(time => (<option key={time} value={time}>{time}</option>))}
                </select>
                {availableTimesForToday.length === 0 && <p className="text-error text-xs mt-2">Não há mais horários disponíveis hoje.</p>}
              </div>
            </div>

            <div className="flex justify-end gap-3 mt-8">
              <button onClick={() => setIsRescheduleModalOpen(false)} className="px-6 py-2.5 rounded-xl font-label text-xs text-on-surface-variant hover:bg-surface-container-high transition-colors uppercase tracking-wider">Voltar</button>
              <button onClick={handleSaveReschedule} disabled={!newRescheduleTime} className="px-6 py-2.5 rounded-xl font-label text-xs bg-primary text-black hover:bg-primary/90 disabled:opacity-50 transition-colors uppercase tracking-wider">Salvar Alteração</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}