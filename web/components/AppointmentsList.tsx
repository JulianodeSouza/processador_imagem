"use client";

import Link from "next/link";
import { ReactNode } from "react";
import AppointmentCard, { AppointmentData } from "./AppointmentCard";

interface AppointmentsListProps {
  appointments: AppointmentData[];
  title?: string;
  showViewAll?: boolean;
  viewAllHref?: string;
  onAppointmentMore?: (id: string) => void;
}

export default function AppointmentsList({
  appointments,
  title = "Próximos Agendamentos",
  showViewAll = true,
  viewAllHref = "/agenda",
  onAppointmentMore,
}: AppointmentsListProps): ReactNode {
  return (
    <section className="lg:col-span-2 space-y-4 lg:space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center gap-4">
        <h2 className="text-xl lg:text-2xl font-bold font-headline">{title}</h2>
        {showViewAll && (
          <Link
            href={viewAllHref}
            className="text-xs font-label text-primary hover:text-primary/80 transition-all whitespace-nowrap"
          >
            Ver Agenda
          </Link>
        )}
      </div>

      {/* Appointments List */}
      <div className="space-y-3 lg:space-y-4">
        {appointments.length > 0 ? (
          appointments.map((appointment) => (
            <AppointmentCard
              key={appointment.id}
              appointment={appointment}
              onMore={onAppointmentMore}
            />
          ))
        ) : (
          <div className="text-center py-8 text-on-surface-variant">
            <p>Nenhum agendamento próximo</p>
          </div>
        )}
      </div>
    </section>
  );
}
