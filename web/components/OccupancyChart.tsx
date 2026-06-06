"use client";

import { ReactNode } from "react";

interface OccupancyBarData {
  hour: string;
  percentage: number;
  color?: "primary" | "secondary" | "muted";
}

interface OccupancyChartProps {
  title?: string;
  bars?: OccupancyBarData[];
  peakHour?: string;
  averageWaitTime?: string;
}

const defaultBars: OccupancyBarData[] = [
  { hour: "08h", percentage: 40, color: "muted" },
  { hour: "10h", percentage: 75, color: "primary" },
  { hour: "12h", percentage: 95, color: "primary" },
  { hour: "14h", percentage: 60, color: "primary" },
  { hour: "16h", percentage: 100, color: "secondary" },
  { hour: "18h", percentage: 80, color: "primary" },
  { hour: "20h", percentage: 30, color: "muted" },
];

const getBarColor = (color?: "primary" | "secondary" | "muted") => {
  switch (color) {
    case "primary":
      return "bg-primary";
    case "secondary":
      return "bg-secondary";
    default:
      return "bg-surface-bright";
  }
};

const getBarGlow = (color?: "primary" | "secondary" | "muted") => {
  switch (color) {
    case "primary":
      return "shadow-[0_0_15px_rgba(129,236,255,0.4)]";
    case "secondary":
      return "shadow-[0_0_15px_rgba(254,183,0,0.4)]";
    default:
      return "";
  }
};

export default function OccupancyChart({
  title = "Ocupação do Atelier",
  bars = defaultBars,
  peakHour = "16:00 - 17:00",
  averageWaitTime = "12 min",
}: OccupancyChartProps): ReactNode {
  return (
    <div className="glass-card rounded-2xl p-6 lg:p-8 border border-outline-variant/20">
      <h2 className="text-lg lg:text-xl font-bold font-headline mb-6 lg:mb-8">{title}</h2>

      {/* Bar Chart */}
      <div className="space-y-6 lg:space-y-8">
        <div className="flex items-end justify-between h-40 lg:h-48 gap-2 px-2 lg:px-4">
          {bars.map((bar, index) => (
            <div key={index} className="flex-1 flex flex-col items-center">
              <div
                className={`w-full rounded-t-sm group relative transition-all duration-200 ${getBarColor(
                  bar.color
                )} ${getBarGlow(bar.color)} hover:opacity-80`}
                style={{ height: `${bar.percentage}%` }}
              >
                <div className="absolute -top-8 left-1/2 -translate-x-1/2 text-[9px] font-bold opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                  {bar.hour}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Info Footer */}
        <div className="pt-6 lg:pt-8 border-t border-outline-variant/10">
          <div className="flex justify-between items-center gap-4 mb-4">
            <span className="font-label text-xs text-on-surface-variant">
              Horário de Pico
            </span>
            <span className="text-xs font-bold text-secondary">{peakHour}</span>
          </div>
          <div className="flex justify-between items-center gap-4">
            <span className="font-label text-xs text-on-surface-variant">
              Tempo Médio de Espera
            </span>
            <span className="text-xs font-bold text-on-surface">
              {averageWaitTime}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
