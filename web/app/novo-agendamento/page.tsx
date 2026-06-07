"use client";

import { useState, useEffect, ReactNode, FormEvent } from "react";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import { api } from "@/services/api";

interface FormData {
  clientName: string;
  clientEmail: string;
  clientPhone: string;
  service: string;
  barber: string;
  date: string;
  time: string;
  notes: string;
}

interface Barber {
  id: string;
  name: string;
}

interface ServiceType {
  id: string;
  nome: string;
  descricao: string;
  preco: number;
}

export default function NovoAgendamentoPage(): ReactNode {
  const [formData, setFormData] = useState<FormData>({
    clientName: "",
    clientEmail: "",
    clientPhone: "",
    service: "",
    barber: "",
    date: "",
    time: "",
    notes: "",
  });

  const [availableTimes, setAvailableTimes] = useState<string[]>([]);
  const [loadingTimes, setLoadingTimes] = useState(false);
  const [barbers, setBarbers] = useState<Barber[]>([]);
  const [services, setServices] = useState<ServiceType[]>([]);
  const [submitted, setSubmitted] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Busca barbeiros e serviços simultaneamente
  useEffect(() => {
    async function fetchData() {
      try {
        const [barbersRes, servicesRes] = await Promise.all([
          api.get("/barbeiros"),
          api.get("/cortes"),
        ]);
        setBarbers(barbersRes.data || []);
        setServices(servicesRes.data || []);
      } catch (error) {
        console.error("Erro ao carregar dados do formulário:", error);
      }
    }
    fetchData();
  }, []);

  const handleChange = (
    e: React.ChangeEvent<
      HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement
    >,
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      await api.post("/agendamentos", formData);

      setSubmitted(true);
      setTimeout(() => {
        setFormData({
          clientName: "",
          clientEmail: "",
          clientPhone: "",
          service: "",
          barber: "",
          date: "",
          time: "",
          notes: "",
        });
        setSubmitted(false);
      }, 3000);
    } catch (error) {
      console.error("Erro ao criar agendamento:", error);
      alert("Ocorreu um erro ao criar o agendamento. Tente novamente.");
    } finally {
      setIsLoading(false);
    }
  };

  const servicoSelecionado = services.find((s) => s.nome === formData.service);

  const gerarHorariosDisponiveis = () => {
    const horarios = [];
    const horaAbertura = 9;
    const horaFechamento = 18;

    for (let i = horaAbertura; i <= horaFechamento; i++) {
      const horaFormatada = i.toString().padStart(2, '0');

      // Adiciona a hora cheia (ex: 09:00)
      horarios.push(`${horaFormatada}:00`);

      // Se quiser agendamentos a cada 30 minutos, adicione esta lógica:
      // (Evita adicionar 18:30 se o fechamento é exatamente às 18:00)
      if (i < horaFechamento) {
        horarios.push(`${horaFormatada}:30`);
      }
    }
    return horarios;
  };

  useEffect(() => {
    async function fetchDisponibilidade() {
      if (formData.date && formData.barber) {
        setLoadingTimes(true);
        try {
          const res = await api.get("/agendamentos/disponibilidade", {
            params: { date: formData.date, barber: formData.barber }
          });
          setAvailableTimes(res.data);

          // Se o horário que estava selecionado não estiver mais disponível, limpa ele
          if (formData.time && !res.data.includes(formData.time)) {
            setFormData(prev => ({ ...prev, time: "" }));
          }
        } catch (error) {
          console.error("Erro ao buscar horários", error);
        } finally {
          setLoadingTimes(false);
        }
      } else {
        setAvailableTimes([]);
      }
    }
    fetchDisponibilidade();
  }, [formData]);

  const horariosOpcoes = gerarHorariosDisponiveis();

  return (
    <div className="w-full bg-surface min-h-screen">
      <Header title="BarberShop Atelier" />
      <Sidebar />

      <main className="lg:pl-72 pt-24 min-h-screen pb-12">
        <div className="px-6 lg:px-12 py-8 max-w-4xl mx-auto">
          {/* Header Section */}
          <section className="mb-12">
            <h1 className="text-4xl lg:text-5xl font-bold italic tracking-tight text-on-surface mb-3 font-headline">
              Novo Agendamento
            </h1>
            <p className="text-on-surface-variant font-label text-sm lg:text-base">
              Agende um novo cliente no seu atelier. Preencha os dados abaixo e
              confirme a disponibilidade.
            </p>
          </section>

          {/* Success Message */}
          {submitted && (
            <div className="mb-8 p-4 lg:p-6 rounded-lg bg-primary/20 border border-primary/50 flex items-start gap-3 animate-pulse">
              <span className="material-symbols-outlined text-primary text-2xl flex-shrink-0">
                check_circle
              </span>
              <div>
                <p className="text-primary font-label font-semibold">
                  Agendamento realizado com sucesso!
                </p>
                <p className="text-primary/80 text-sm mt-1">
                  Um email de confirmação foi enviado para{" "}
                  {formData.clientEmail}
                </p>
              </div>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-8">
            {/* Dados do Cliente */}
            <div className="glass-card rounded-xl p-6 lg:p-8 border border-outline-variant/10">
              <h2 className="text-xl lg:text-2xl font-bold text-on-surface mb-6 font-headline flex items-center gap-3">
                <span className="material-symbols-outlined text-primary">
                  person
                </span>
                Dados do Cliente
              </h2>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="lg:col-span-2">
                  <label className="block text-on-surface font-label text-sm font-semibold mb-2">
                    Nome Completo *
                  </label>
                  <input
                    type="text"
                    name="clientName"
                    value={formData.clientName}
                    onChange={handleChange}
                    required
                    placeholder="João Silva"
                    className="w-full px-4 py-3 rounded-lg bg-surface-container border border-outline-variant/30 text-on-surface focus:border-primary transition-all duration-300"
                  />
                </div>
                <div>
                  <label className="block text-on-surface font-label text-sm font-semibold mb-2">
                    Email *
                  </label>
                  <input
                    type="email"
                    name="clientEmail"
                    value={formData.clientEmail}
                    onChange={handleChange}
                    required
                    placeholder="joao@example.com"
                    className="w-full px-4 py-3 rounded-lg bg-surface-container border border-outline-variant/30 text-on-surface focus:border-primary transition-all duration-300"
                  />
                </div>
                <div>
                  <label className="block text-on-surface font-label text-sm font-semibold mb-2">
                    Telefone *
                  </label>
                  <input
                    type="tel"
                    name="clientPhone"
                    value={formData.clientPhone}
                    onChange={handleChange}
                    required
                    placeholder="(11) 99999-9999"
                    className="w-full px-4 py-3 rounded-lg bg-surface-container border border-outline-variant/30 text-on-surface focus:border-primary transition-all duration-300"
                  />
                </div>
              </div>
            </div>

            {/* Serviço e Profissional */}
            <div className="glass-card rounded-xl p-6 lg:p-8 border border-outline-variant/10">
              <h2 className="text-xl lg:text-2xl font-bold text-on-surface mb-6 font-headline flex items-center gap-3">
                <span className="material-symbols-outlined text-secondary">
                  content_cut
                </span>
                Serviço e Profissional
              </h2>

              {/* Serviço e Profissional */}
              <div className="glass-card rounded-xl p-6 lg:p-8 border border-outline-variant/10">
                <h2 className="text-xl lg:text-2xl font-bold text-on-surface mb-6 font-headline flex items-center gap-3">
                  <span className="material-symbols-outlined text-secondary">
                    content_cut
                  </span>
                  Serviço e Profissional
                </h2>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* COLUNA DO SERVIÇO */}
                  <div className="flex flex-col gap-3">
                    <div>
                      <label className="block text-on-surface font-label text-sm font-semibold mb-2">
                        Serviço *
                      </label>
                      <div className="relative">
                        <select
                          name="service"
                          value={formData.service}
                          onChange={handleChange}
                          required
                          className="w-full px-4 py-3 rounded-lg bg-surface-container border border-outline-variant/30 text-on-surface focus:border-primary appearance-none pr-10"
                        >
                          <option value="">Selecione um serviço</option>
                          {services.map((service) => (
                            <option key={service.id} value={service.nome}>
                              {service.nome}
                            </option>
                          ))}
                        </select>
                        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-on-surface-variant">
                          <span className="material-symbols-outlined text-xl">
                            expand_more
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* CAIXINHA DE DESCRIÇÃO DO CORTE */}
                    {servicoSelecionado && servicoSelecionado.descricao && (
                      <div className="p-4 rounded-lg bg-secondary/10 border border-secondary/20 transition-all duration-300 animate-fade-in">
                        <h4 className="text-secondary font-semibold text-sm mb-1 flex items-center gap-1">
                          <span className="material-symbols-outlined text-[16px]">
                            info
                          </span>
                          Detalhes do Serviço
                        </h4>
                        <p className="text-on-surface-variant text-sm mb-2 leading-relaxed">
                          {servicoSelecionado.descricao}
                        </p>
                        {servicoSelecionado.preco && (
                          <span className="inline-block px-3 py-1 bg-secondary/20 text-secondary rounded-full text-xs font-bold">
                            R${" "}
                            {Number(servicoSelecionado.preco)
                              .toFixed(2)
                              .replace(".", ",")}
                          </span>
                        )}
                      </div>
                    )}
                  </div>

                  {/* COLUNA DO BARBEIRO */}
                  <div>
                    <label className="block text-on-surface font-label text-sm font-semibold mb-2">
                      Barbeiro *
                    </label>
                    <div className="relative">
                      <select
                        name="barber"
                        value={formData.barber}
                        onChange={handleChange}
                        required
                        className="w-full px-4 py-3 rounded-lg bg-surface-container border border-outline-variant/30 text-on-surface focus:border-primary appearance-none pr-10"
                      >
                        <option value="">Selecione um barbeiro</option>
                        {barbers.map((barber) => (
                          <option key={barber.id} value={barber.name}>
                            {barber.name}
                          </option>
                        ))}
                      </select>
                      <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-on-surface-variant">
                        <span className="material-symbols-outlined text-xl">
                          expand_more
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Data e Hora */}
            <div className="glass-card rounded-xl p-6 lg:p-8 border border-outline-variant/10">
              <h2 className="text-xl lg:text-2xl font-bold text-on-surface mb-6 font-headline flex items-center gap-3">
                <span className="material-symbols-outlined text-primary">
                  schedule
                </span>
                Data e Hora
              </h2>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <label className="block text-on-surface font-label text-sm font-semibold mb-2">
                    Data *
                  </label>
                  <input
                    type="date"
                    name="date"
                    value={formData.date}
                    onChange={handleChange}
                    required
                    className="w-full px-4 py-3 rounded-lg bg-surface-container border border-outline-variant/30 text-on-surface focus:border-primary"
                  />
                </div>

                <div>
                  <label className="block text-on-surface font-label text-sm font-semibold mb-2">
                    Hora *
                  </label>
                  <select id="time" name="time" required className="w-full px-4 py-3 rounded-lg bg-surface-container border border-outline-variant/30 text-on-surface focus:border-primary appearance-none disabled:opacity-60 disabled:cursor-not-allowed">
                    <option value="">Selecione o horário...</option>
                    {horariosOpcoes.map((horario) => (
                      <option key={horario} value={horario}>
                        {horario}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                </div>
              </div>
            </div>

            {/* Observações */}
            <div className="glass-card rounded-xl p-6 lg:p-8 border border-outline-variant/10">
              <h2 className="text-xl lg:text-2xl font-bold text-on-surface mb-6 font-headline flex items-center gap-3">
                <span className="material-symbols-outlined text-tertiary">
                  note
                </span>
                Observações
              </h2>
              <textarea
                name="notes"
                value={formData.notes}
                onChange={handleChange}
                placeholder="Informações adicionais..."
                rows={4}
                className="w-full px-4 py-3 rounded-lg bg-surface-container border border-outline-variant/30 text-on-surface focus:border-primary resize-none"
              />
            </div>

            {/* Form Actions */}
            <div className="flex flex-col sm:flex-row gap-4 pt-4">
              <button
                type="submit"
                disabled={isLoading}
                className="flex-1 px-6 py-4 rounded-lg bg-gradient-to-r from-primary to-primary/80 text-on-primary font-label text-sm font-bold uppercase tracking-wide hover:opacity-90 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
              >
                <span className="material-symbols-outlined">
                  {isLoading ? "hourglass_empty" : "check_circle"}
                </span>
                {isLoading ? "A processar..." : "Confirmar Agendamento"}
              </button>
              <button
                type="reset"
                onClick={() =>
                  setFormData({
                    clientName: "",
                    clientEmail: "",
                    clientPhone: "",
                    service: "",
                    barber: "",
                    date: "",
                    time: "",
                    notes: "",
                  })
                }
                className="flex-1 px-6 py-4 rounded-lg bg-surface-container border border-outline-variant/30 text-on-surface font-label text-sm font-bold uppercase hover:bg-surface-container-high transition-all flex items-center justify-center gap-2"
              >
                <span className="material-symbols-outlined">refresh</span>
                Limpar Formulário
              </button>
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}
