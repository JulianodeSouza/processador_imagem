"use client";

import { ReactNode } from "react";

interface HeaderProps {
  title?: string;
  onSidebarToggle?: () => void;
}

export default function Header({
  title = "BarberShop Atelier",
  onSidebarToggle,
}: HeaderProps): ReactNode {
  return (
    <header className="fixed top-0 right-0 w-full z-40 bg-surface/60 backdrop-blur-xl border-b border-outline-variant/20 shadow-[0_4px_20px_rgba(0,0,0,0.3)]">
      <div className="flex justify-between items-center w-full px-6 lg:px-12 py-4 gap-4">
        <button
          onClick={onSidebarToggle}
          className="lg:hidden p-2 text-on-surface-variant hover:text-primary hover:bg-surface-container/50 rounded-lg transition-all duration-300"
        >
          <span className="material-symbols-outlined text-2xl">menu</span>
        </button>

        <div className="flex items-center gap-6 lg:gap-12 min-w-0">
          <span className="font-headline text-xl lg:text-2xl font-bold text-primary whitespace-nowrap">
            {title}
          </span>
        </div>
      </div>
    </header>
  );
}
