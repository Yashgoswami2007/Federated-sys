"use client";

import { motion } from "framer-motion";
import { Activity, ArrowUpRight } from "lucide-react";
import { StatusIndicator } from "@/components/shared/status-indicator";

const stats = [
  { value: "247", label: "Active Devices", color: "text-cyan-400", glow: "shadow-[0_0_30px_-5px_rgba(34,211,238,0.4)]" },
  { value: "1,842", label: "Rounds Complete", color: "text-violet-400", glow: "shadow-[0_0_30px_-5px_rgba(167,139,250,0.4)]" },
  { value: "94.7%", label: "Model Accuracy", color: "text-emerald-400", glow: "shadow-[0_0_30px_-5px_rgba(52,211,153,0.4)]" },
];

export function HeroSection() {
  return (
    <motion.section
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      className="relative overflow-hidden rounded-3xl border border-white/[0.08]"
    >
      {/* Layered background */}
      <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/[0.08] via-[#0a0a12] to-violet-500/[0.08]" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_60%_at_50%_-20%,rgba(34,211,238,0.15),transparent)]" />
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGRlZnM+PHBhdHRlcm4gaWQ9ImdyaWQiIHdpZHRoPSI2MCIgaGVpZ2h0PSI2MCIgcGF0dGVyblVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHBhdGggZD0iTSA2MCAwIEwgMCAwIDAgNjAiIGZpbGw9Im5vbmUiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjA0KSIgc3Ryb2tlLXdpZHRoPSIxIi8+PC9wYXR0ZXJuPjwvZGVmcz48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI2dyaWQpIi8+PC9zdmc+')]" />

      {/* Animated orb */}
      <motion.div
        animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.5, 0.3] }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-cyan-500/10 blur-3xl"
      />

      <div className="relative px-8 py-10 lg:px-10 lg:py-12">
        <div className="flex flex-col gap-10 lg:flex-row lg:items-center lg:justify-between">
          <div className="max-w-xl">
            <div className="mb-4 flex items-center gap-3">
              <span className="inline-flex items-center gap-2 rounded-full border border-cyan-500/20 bg-cyan-500/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.15em] text-cyan-400">
                <Activity className="h-3 w-3" />
                Live Network
              </span>
              <StatusIndicator status="online" variant="pill" size="sm" />
            </div>

            <h1 className="text-3xl font-bold tracking-tight lg:text-4xl xl:text-[2.75rem] xl:leading-tight">
              <span className="text-gradient">FusionNet</span>
              <br />
              <span className="text-zinc-100">Control Center</span>
            </h1>

            <p className="mt-4 text-base leading-relaxed text-zinc-400 lg:text-lg">
              Distributed federated learning network — orchestrating{" "}
              <span className="font-medium text-zinc-300">247 edge devices</span>{" "}
              across 5 global regions with enterprise-grade privacy.
            </p>

            <div className="mt-6 flex items-center gap-2 text-sm text-zinc-500">
              <ArrowUpRight className="h-4 w-4 text-emerald-400" />
              <span>
                Network uptime{" "}
                <span className="font-semibold text-emerald-400">99.97%</span>{" "}
                · Last sync 12s ago
              </span>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 lg:gap-4">
            {stats.map((stat, i) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 + i * 0.1, duration: 0.5 }}
                className={`relative overflow-hidden rounded-2xl border border-white/[0.08] bg-white/[0.03] px-5 py-6 text-center backdrop-blur-sm ${stat.glow}`}
              >
                <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
                <p className={`font-mono text-2xl font-bold tracking-tight lg:text-3xl ${stat.color}`}>
                  {stat.value}
                </p>
                <p className="mt-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                  {stat.label}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </motion.section>
  );
}
