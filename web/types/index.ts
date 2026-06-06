/**
 * Tipos compartilhados da aplicação
 */

export interface SidebarItem {
  id: string;
  label: string;
  href: string;
  icon: string;
  active?: boolean;
}

export type AppointmentStatus = "confirmado" | "chegando" | "pendente";

export interface Appointment {
  id: string;
  time: string;
  timeLabel: string;
  clientName: string;
  service: string;
  status: AppointmentStatus;
  avatarSrc?: string;
  avatarAlt?: string;
}

export interface MetricsData {
  icon: string;
  iconBgColor: string;
  iconColor: string;
  value: string | number;
  label: string;
  badge?: string;
  badgeColor?: string;
  backgroundIcon?: string;
}

export interface OccupancyData {
  hour: string;
  percentage: number;
  color?: "primary" | "secondary" | "muted";
}

export interface StatusIndicator {
  active: boolean;
  label: string;
}
