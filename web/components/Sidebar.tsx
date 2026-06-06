"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ReactNode } from "react";

interface SidebarItem {
  id: string;
  label: string;
  href: string;
  icon: string;
}

interface SidebarProps {
  title?: string;
  subtitle?: string;
  items?: SidebarItem[];
  onNewSchedule?: () => void;
  newScheduleText?: string;
  isOpen?: boolean;
  onClose?: () => void;
}

const defaultItems: SidebarItem[] = [
  {
    id: "dashboard",
    label: "Dashboard",
    href: "/",
    icon: "dashboard",
  },
  {
    id: "agenda",
    label: "Agenda",
    href: "/agenda",
    icon: "calendar_today",
  },
  {
    id: "novo-agendamento",
    label: "Novo Agendamento",
    href: "/novo-agendamento",
    icon: "add_circle",
  },
  {
    id: "clientes",
    label: "Clientes",
    href: "/clientes",
    icon: "group", 
  },
  {
    id: "equipe",
    label: "Equipe",
    href: "/equipe",
    icon: "badge",
  },
  {
    id: "visagismo",
    label: "Visagismo",
    href: "/visagismo",
    icon: "face_6",
  },
];

export default function Sidebar({
  title = "Painel de Gestão",
  subtitle = "Estética Inteligente",
  items = defaultItems,
  onNewSchedule,
  newScheduleText = "NOVO AGENDAMENTO",
  isOpen = false,
  onClose,
}: SidebarProps): ReactNode {
  const pathname = usePathname();

  const isActive = (href: string) => {
    if (href === "/") {
      return pathname === "/";
    }
    return pathname.startsWith(href);
  };
  return (
    <aside
      className={`h-screen w-64 lg:w-72 fixed left-0 top-0 z-40 border-r border-outline-variant/20 bg-surface/95 backdrop-blur-xl shadow-[8px_0_24px_rgba(0,0,0,0.4)] flex flex-col pt-24 pb-8 transition-all duration-300 lg:flex ${
        isOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
      }`}
    >
      {/* Header */}
      <div className="px-6 lg:px-8 mb-8 lg:mb-12">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-headline text-lg lg:text-xl text-secondary font-bold">
            {title}
          </h2>
          <button
            onClick={onClose}
            className="lg:hidden p-2 text-on-surface-variant hover:text-primary rounded-lg transition-all duration-200"
          >
            <span className="material-symbols-outlined text-xl">close</span>
          </button>
        </div>
        <p className="font-label text-[10px] lg:text-xs text-on-surface-variant">
          {subtitle}
        </p>
      </div>

      {/* Navigation Items */}
      <nav className="flex-1 flex flex-col gap-2 px-4 lg:px-6">
        {items.map((item) => {
          const active = isActive(item.href);
          return (
            <Link
              key={item.id}
              href={item.href}
              onClick={onClose}
              className={`flex items-center gap-4 px-4 lg:px-6 py-3 lg:py-4 rounded-lg transition-all duration-300 group relative ${
                active
                  ? "text-primary bg-primary/10 shadow-lg shadow-primary/20"
                  : "text-on-surface-variant hover:bg-surface-container/50 hover:text-on-surface"
              }`}
            >
              {/* Barra esquerda ativa */}
              {active && (
                <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-primary via-primary to-primary/50 rounded-r-full" />
              )}

              <span
                className={`material-symbols-outlined text-xl lg:text-2xl flex-shrink-0 transition-all duration-300 ${
                  active ? "text-primary" : ""
                }`}
              >
                {item.icon}
              </span>
              <span className="font-label text-xs hidden sm:inline transition-colors duration-300">
                {item.label}
              </span>

              {/* Efeito hover glow */}
              {!active && (
                <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-primary/0 via-primary/0 to-primary/0 group-hover:from-primary/5 group-hover:to-primary/10 transition-all duration-300 pointer-events-none" />
              )}
            </Link>
          );
        })}
      </nav>

      {/* Novo Agendamento Button */}
      <div className="px-4 lg:px-6 mt-auto border-t border-outline-variant/20 pt-6">
        <button
          onClick={() => {
            onNewSchedule?.();
            onClose?.();
          }}
          className="w-full flex items-center justify-center gap-2 px-6 py-3 rounded-lg bg-gradient-to-r from-secondary to-secondary/80 text-on-secondary font-label text-xs font-bold tracking-wide hover:from-secondary hover:to-secondary/90 hover:shadow-lg hover:shadow-secondary/30 transition-all duration-300 active:scale-95"
        >
          <span className="material-symbols-outlined text-lg">add_circle</span>
          <span>{newScheduleText}</span>
        </button>
      </div>
    </aside>
  );
}
