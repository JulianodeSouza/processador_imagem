"use client";

import Link from "next/link";
import { ReactNode } from "react";

interface FloatingActionButtonProps {
  onClick?: () => void;
  href?: string;
  icon?: string;
  label?: string;
  tooltip?: string;
}

export default function FloatingActionButton({
  onClick,
  href = "/novo-agendamento",
  icon = "add",
  label = "Rápido Agendamento",
  tooltip = "Rápido Agendamento",
}: FloatingActionButtonProps): ReactNode {
  const content = (
    <button
      onClick={onClick}
      className="fixed bottom-8 right-8 h-16 w-16 rounded-full bg-primary text-on-primary-fixed shadow-[0_0_30px_rgba(129,236,255,0.4)] flex items-center justify-center transition-all hover:scale-110 active:scale-95 group z-50"
    >
      <span className="material-symbols-outlined text-3xl font-bold">
        {icon}
      </span>
      <span className="absolute right-full mr-4 bg-surface-container px-4 py-2 rounded-lg text-xs font-bold uppercase tracking-wider whitespace-nowrap opacity-0 translate-x-4 group-hover:opacity-100 group-hover:translate-x-0 transition-all pointer-events-none border border-outline-variant/20">
        {tooltip}
      </span>
    </button>
  );

  return href ? <Link href={href}>{content}</Link> : content;
}
