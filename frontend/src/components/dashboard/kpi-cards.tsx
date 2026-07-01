"use client";

import { motion } from "framer-motion";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
} from "recharts";
import {
  Cpu,
  Layers,
  Target,
  Shield,
  TrendingUp,
  TrendingDown,
  Minus,
} from "lucide-react";
import { kpiMetrics } from "@/lib/mock-data";
import { GlassCard } from "@/components/shared/glass-card";
import { cn } from "@/lib/utils";
import { CHART_COLORS } from "@/components/charts/chart-theme";

const iconMap = {
  cpu: Cpu,
  layers: Layers,
  target: Target,
  shield: Shield,
};

const trendIcons = {
  up: TrendingUp,
  down: TrendingDown,
  neutral: Minus,
};

const sparkData = [
  { v: 210 }, { v: 225 }, { v: 218 }, { v: 235 }, { v: 240 }, { v: 247 },
];

export function KpiCards() {
  return (
    <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
      {kpiMetrics.map((metric, index) => {
        const Icon = iconMap[metric.icon as keyof typeof iconMap];
        const TrendIcon = trendIcons[metric.trend];

        return (
          <GlassCard key={metric.label} delay={index * 0.06} glow={index % 2 === 0 ? "cyan" : "violet"}>
            <div className="p-6">
              <div className="flex items-start justify-between">
                <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-white/[0.08] bg-gradient-to-br from-cyan-500/15 to-violet-500/10 shadow-[0_0_20px_-5px_rgba(34,211,238,0.3)]">
                  <Icon className="h-5 w-5 text-cyan-400" />
                </div>
                <div
                  className={cn(
                    "flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold",
                    metric.trend === "up" && "bg-emerald-500/10 text-emerald-400",
                    metric.trend === "down" && "bg-red-500/10 text-red-400",
                    metric.trend === "neutral" && "bg-zinc-500/10 text-zinc-500"
                  )}
                >
                  <TrendIcon className="h-3 w-3" />
                  {metric.change.split(" ")[0]}
                </div>
              </div>

              <motion.p
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.3 + index * 0.08 }}
                className="mt-5 font-mono text-3xl font-bold tracking-tight text-zinc-50"
              >
                {metric.value}
              </motion.p>

              <p className="mt-1 text-sm font-medium text-zinc-400">
                {metric.label}
              </p>
              <p className="mt-1 text-xs text-zinc-600">{metric.change}</p>

              <div className="mt-4 h-10 opacity-60">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={sparkData}>
                    <defs>
                      <linearGradient id={`spark-${index}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={CHART_COLORS.cyan} stopOpacity={0.3} />
                        <stop offset="100%" stopColor={CHART_COLORS.cyan} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <Area
                      type="monotone"
                      dataKey="v"
                      stroke={CHART_COLORS.cyan}
                      strokeWidth={1.5}
                      fill={`url(#spark-${index})`}
                      dot={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </GlassCard>
        );
      })}
    </div>
  );
}
