"use client";
import { ReactNode } from "react";

export interface AppointmentData {
  id: string;
  time: string;
  timeLabel: string;
  clientName: string;
  service: string;
  status: "concluido" | "agendado" | "cancelado";
  avatarSrc?: string;
  avatarAlt?: string;
}

interface AppointmentCardProps {
  appointment: AppointmentData;
}

const statusConfig = {
  concluido: {
    bgColor: "bg-surface-container-high",
    borderColor: "border-primary/30",
    textColor: "text-primary",
    dotColor: "bg-primary shadow-[0_0_8px_rgba(129,236,255,0.8)]",
    label: "Concluído",
  },
  agendado: {
    bgColor: "bg-surface-container-high",
    borderColor: "border-secondary/30",
    textColor: "text-secondary",
    dotColor: "bg-secondary shadow-[0_0_8px_rgba(254,183,0,0.8)]",
    label: "Agendado",
  },
  cancelado: {
    bgColor: "bg-surface",
    borderColor: "border-outline-variant/30",
    textColor: "text-on-surface-variant",
    dotColor: "bg-outline-variant",
    label: "Cancelado",
  },
};

export default function AppointmentCard({ appointment }: AppointmentCardProps): ReactNode {
  const config = statusConfig[appointment.status];
  const isPlaceholder = !appointment.avatarSrc;

  return (
    <div className={`flex items-center gap-6 p-5 rounded-[1.5rem] bg-surface-container border border-outline-variant/10 shadow-[0_0_40px_rgba(255,255,255,0.02)] transition-all group ${appointment.status === "cancelado" ? "opacity-60" : ""}`}>
      
      {/* Time */}
      <div className="flex flex-col items-center justify-center min-w-[70px]">
        <span className={`text-2xl font-bold font-headline ${appointment.status === "agendado" ? "text-on-surface-variant" : "text-primary"}`}>
          {appointment.time}
        </span>
        <span className="text-[10px] font-bold text-on-surface-variant font-label mt-1">
          {appointment.timeLabel}
        </span>
      </div>

      {/* Avatar */}
      <div className="h-12 w-12 rounded-full overflow-hidden border border-outline-variant/30 flex-shrink-0 relative">
        {isPlaceholder ? (
          <div className="w-full h-full bg-surface-container-high flex items-center justify-center">
            <span className="material-symbols-outlined text-on-surface-variant">person</span>
          </div>
        ) : (
          <img alt={appointment.avatarAlt} className="h-full w-full object-cover grayscale opacity-90 group-hover:grayscale-0 transition-all duration-500" src={appointment.avatarSrc} />
        )}
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0 flex flex-col justify-center">
        <h4 className="font-bold font-headline text-on-surface text-lg truncate mb-1">
          {appointment.clientName}
        </h4>
        <p className="text-xs text-on-surface-variant font-label truncate">
          {appointment.service}
        </p>
      </div>

      {/* Status Chip Neon */}
      <div className="flex items-center gap-4 flex-shrink-0">
        <div className={`flex items-center gap-2 px-4 py-1.5 rounded-full border ${config?.borderColor} ${config?.bgColor}`}>
          <span className={`w-1.5 h-1.5 rounded-full ${config?.dotColor}`}></span>
          <span className={`text-[10px] font-bold font-label tracking-widest ${config?.textColor}`}>
            {config?.label}
          </span>
        </div>
      </div>
    </div>
  );
}