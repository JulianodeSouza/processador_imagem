"use client";
import { ReactNode } from "react";

interface MetricsCardProps {
  icon: string;
  value: string | number;
  label: string;
  badge?: string;
  badgeColor?: string;
  backgroundIcon?: string;
}

export default function MetricsCard({
  icon,
  value,
  label,
  badge,
  badgeColor = "text-primary",
  backgroundIcon,
}: MetricsCardProps): ReactNode {
  return (
    <div className="bg-surface-container rounded-[1.5rem] p-8 flex flex-col justify-between group relative overflow-hidden border border-outline-variant/20 shadow-[0_0_40px_rgba(255,255,255,0.04)] hover:shadow-[0_0_40px_rgba(255,255,255,0.08)] transition-all duration-500">
      
      {backgroundIcon && (
        <div className="absolute -bottom-4 -right-4 p-8 opacity-[0.03] pointer-events-none transition-transform group-hover:scale-110 duration-500">
          <span className="material-symbols-outlined text-9xl">
            {backgroundIcon}
          </span>
        </div>
      )}

      <div className="flex justify-between items-start mb-10 relative z-10">
        <div className="p-3 bg-surface-container-high rounded-xl border border-outline-variant/10">
          <span className="material-symbols-outlined text-2xl text-on-surface">
            {icon}
          </span>
        </div>
        {badge && (
          <span className={`font-label text-[10px] font-bold uppercase tracking-widest ${badgeColor}`}>
            {badge}
          </span>
        )}
      </div>

      <div className="relative z-10 flex flex-col items-start">
        <div className="text-5xl font-bold font-headline text-on-surface tracking-tight mb-2">
          {value}
        </div>
        <h3 className="text-on-surface-variant font-label text-xs uppercase tracking-widest">
          {label}
        </h3>
      </div>
    </div>
  );
}