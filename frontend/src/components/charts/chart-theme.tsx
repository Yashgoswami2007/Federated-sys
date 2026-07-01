"use client";

import { cn } from "@/lib/utils";

export const CHART_COLORS = {
  cyan: "#22d3ee",
  cyanDim: "#06b6d4",
  violet: "#a78bfa",
  violetDim: "#8b5cf6",
  emerald: "#34d399",
  grid: "rgba(255,255,255,0.04)",
  axis: "#52525b",
  tooltipBg: "rgba(8, 8, 12, 0.96)",
  tooltipBorder: "rgba(255,255,255,0.12)",
};

export const axisTickStyle = {
  fill: CHART_COLORS.axis,
  fontSize: 11,
  fontFamily: "var(--font-geist-mono), monospace",
};

export const gridProps = {
  strokeDasharray: "4 8",
  stroke: CHART_COLORS.grid,
  vertical: false,
};

export interface ChartTooltipProps {
  active?: boolean;
  payload?: ReadonlyArray<{ name?: string | number; value?: string | number; color?: string }>;
  label?: string | number;
  valueFormatter?: (value: number) => string;
  labelFormatter?: (label: string) => string;
}

export function ChartTooltip({
  active,
  payload,
  label,
  valueFormatter = (v) => String(v),
  labelFormatter = (l) => l,
}: ChartTooltipProps) {
  if (!active || !payload?.length) return null;

  return (
    <div className="rounded-xl border border-white/10 bg-[#08080c]/95 px-4 py-3 shadow-2xl shadow-black/40 backdrop-blur-xl">
      <p className="mb-2 text-[10px] font-medium uppercase tracking-widest text-zinc-500">
        {labelFormatter(String(label ?? ""))}
      </p>
      <div className="space-y-1.5">
        {payload.map((entry, i) => {
          const numValue = Number(entry.value ?? 0);
          return (
            <div key={i} className="flex items-center justify-between gap-6">
              <div className="flex items-center gap-2">
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ backgroundColor: entry.color, boxShadow: `0 0 8px ${entry.color}` }}
                />
                <span className="text-xs text-zinc-400">{String(entry.name ?? "")}</span>
              </div>
              <span className="font-mono text-sm font-semibold text-zinc-100">
                {valueFormatter(numValue)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

interface ChartContainerProps {
  children: React.ReactNode;
  title: string;
  subtitle?: string;
  className?: string;
  height?: number;
}

export function ChartContainer({
  children,
  title,
  subtitle,
  className,
  height = 200,
}: ChartContainerProps) {
  return (
    <div
      className={cn(
        "group relative overflow-hidden rounded-2xl border border-white/[0.06] bg-gradient-to-b from-white/[0.04] to-transparent p-5",
        "before:pointer-events-none before:absolute before:inset-x-0 before:top-0 before:h-px before:bg-gradient-to-r before:from-transparent before:via-cyan-500/30 before:to-transparent",
        className
      )}
    >
      <div className="mb-4 flex items-start justify-between">
        <div>
          <h4 className="text-sm font-semibold tracking-tight text-zinc-100">
            {title}
          </h4>
          {subtitle && (
            <p className="mt-0.5 text-xs text-zinc-500">{subtitle}</p>
          )}
        </div>
        <div className="flex h-2 w-2 items-center justify-center">
          <span className="absolute h-2 w-2 animate-pulse rounded-full bg-cyan-400/40" />
          <span className="relative h-1.5 w-1.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.8)]" />
        </div>
      </div>
      <div style={{ height }}>{children}</div>
    </div>
  );
}
