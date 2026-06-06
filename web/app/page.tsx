"use client";

import { ReactNode, useEffect, useState } from "react";
import MetricsCard from "@/components/MetricsCard";
import AppointmentsList from "@/components/AppointmentsList";
import { AppointmentData } from "@/components/AppointmentCard";
import OccupancyChart from "@/components/OccupancyChart";
import FloatingActionButton from "@/components/FloatingActionButton";
import { api } from "@/services/api";

export default function Dashboard(): ReactNode {
  const [appointments, setAppointments] = useState<AppointmentData[]>([]);
  const [metrics, setMetrics] = useState({
    agendamentosHoje: "0",
    novosClientes: "0",
    faturamentoEstimado: "R$ 0"
  });
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchDashboardData() {
      try {
        // Data atual formatada (YYYY-MM-DD)
        const today = new Date().toISOString().split('T')[0];
        
        // Busca os agendamentos de hoje
        const appointmentsResponse = await api.get(`/agendamentos?data=${today}`);
        setAppointments(appointmentsResponse.data || []);

        // Busca métricas gerais (ajuste a rota conforme o seu backend)
        const metricsResponse = await api.get('/analytics/dashboard');
        if (metricsResponse.data) {
          setMetrics(metricsResponse.data);
        }
      } catch (error) {
        console.error("Erro ao buscar dados do dashboard:", error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchDashboardData();
  }, []);

  return (
    <div className="w-full bg-surface min-h-screen">
      <main className="lg:pl-72 pt-28 pb-20">
        <div className="px-6 lg:px-12 max-w-7xl mx-auto">
          {/* Header Section */}
          <section className="mb-12 flex flex-col lg:flex-row justify-between items-start lg:items-end gap-6 lg:gap-0">
            <div className="max-w-2xl">
              <h1 className="text-4xl lg:text-5xl font-headline font-bold text-on-surface mb-3 tracking-wide">
                Bem-vindo de volta, Mestre.
              </h1>
              <p className="text-on-surface-variant font-label text-sm uppercase tracking-wider">
                O seu atelier digital está pronto para os rituais de hoje.
              </p>
            </div>
            
            {/* Status Neon Chip */}
            <div className="flex items-center gap-3 bg-surface-container-high px-5 py-2.5 rounded-full border border-outline-variant/20 shadow-[0_0_20px_rgba(129,236,255,0.05)]">
              <span className="w-2.5 h-2.5 rounded-full bg-primary animate-pulse shadow-[0_0_12px_#81ecff]"></span>
              <span className="font-label text-xs font-bold text-primary uppercase tracking-widest">
                Operações ao Vivo
              </span>
            </div>
          </section>

          {/* Metrics Grid */}
          <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
            <MetricsCard
              icon="calendar_today"
              value={metrics.agendamentosHoje}
              label="Agendamentos Hoje"
              badge="+12% VS ONTEM"
              badgeColor="text-primary"
            />
            <MetricsCard
              icon="person_add"
              value={metrics.novosClientes}
              label="Novos Clientes"
              badge="HOJE"
              badgeColor="text-secondary"
            />
            <MetricsCard
              icon="trending_up"
              value={metrics.faturamentoEstimado}
              label="Faturação Estimada"
              badge="PROJETADO"
              badgeColor="text-tertiary"
              backgroundIcon="payments"
            />
          </section>

          {/* Main Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 space-y-4">
              {isLoading ? (
                <p className="text-on-surface-variant animate-pulse">A carregar agendamentos...</p>
              ) : (
                <AppointmentsList appointments={appointments} />
              )}
            </div>
            <section className="space-y-8">
              <OccupancyChart />
            </section>
          </div>
        </div>
      </main>
      <FloatingActionButton href="/novo-agendamento" />
    </div>
  );
}