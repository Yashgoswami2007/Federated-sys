"use client";

import { motion } from "framer-motion";
import { Globe2, Radio } from "lucide-react";
import { useWebSocket } from "@/lib/websocket-context";
import { GlassCard } from "@/components/shared/glass-card";
import { SectionHeader } from "@/components/shared/section-header";
import type { RegionData } from "@/types";

const hubX = 50;
const hubY = 44;

// Simplified continent silhouettes for visual depth
const continents = [
  "M18,32 Q22,28 28,30 Q32,26 38,28 Q42,32 40,38 Q36,42 30,40 Q24,42 18,38 Z",
  "M44,26 Q52,22 58,24 Q64,28 62,34 Q58,38 52,36 Q46,38 44,32 Z",
  "M68,30 Q76,28 82,32 Q86,38 82,44 Q76,48 70,44 Q66,38 68,30 Z",
  "M26,52 Q32,48 38,50 Q42,56 38,62 Q32,66 26,62 Q22,56 26,52 Z",
  "M48,50 Q54,48 58,52 Q60,58 56,62 Q50,64 46,58 Q44,52 48,50 Z",
];

export function NetworkMap() {
  const { devices, regions } = useWebSocket();
  const onlineCount = devices.filter((d) => d.status === "online" || d.status === "training").length;

  return (
    <GlassCard delay={0.1} glow="cyan">
      <div className="p-7 lg:p-8">
        <SectionHeader
          eyebrow="Global Infrastructure"
          title="Network Overview"
          description="Real-time federated node topology across active regions"
          action={
            <div className="flex items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1.5">
              <Radio className="h-3.5 w-3.5 text-emerald-400 animate-pulse" />
              <span className="text-xs font-semibold text-emerald-400">
                {onlineCount} devices live
              </span>
            </div>
          }
        />

        <div className="relative mt-6 aspect-[2/1] w-full overflow-hidden rounded-2xl border border-white/[0.06] bg-[#040408]">
          {/* Scan line effect */}
          <div className="pointer-events-none absolute inset-0 z-10 overflow-hidden">
            <div className="animate-scan-line absolute inset-x-0 h-px bg-gradient-to-r from-transparent via-cyan-400/40 to-transparent" />
          </div>

          {/* Corner glows */}
          <div className="pointer-events-none absolute left-1/2 top-1/2 h-32 w-32 -translate-x-1/2 -translate-y-1/2 rounded-full bg-cyan-500/10 blur-3xl" />

          <svg
            viewBox="0 0 100 80"
            className="h-full w-full"
            preserveAspectRatio="xMidYMid meet"
          >
            <defs>
              <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="0.8" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <filter id="glowStrong" x="-100%" y="-100%" width="300%" height="300%">
                <feGaussianBlur stdDeviation="1.5" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <radialGradient id="hubGlow" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.5" />
                <stop offset="50%" stopColor="#22d3ee" stopOpacity="0.15" />
                <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
              </radialGradient>
              <linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.2" />
                <stop offset="50%" stopColor="#22d3ee" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#a78bfa" stopOpacity="0.2" />
              </linearGradient>
              <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#22d3ee" />
                <stop offset="100%" stopColor="#a78bfa" />
              </linearGradient>
            </defs>

            {/* Grid */}
            {Array.from({ length: 25 }).map((_, i) =>
              Array.from({ length: 12 }).map((_, j) => (
                <circle
                  key={`g-${i}-${j}`}
                  cx={i * 4 + 2}
                  cy={j * 6.5 + 3}
                  r="0.12"
                  fill="rgba(255,255,255,0.04)"
                />
              ))
            )}

            {/* Continent outlines */}
            {continents.map((d, i) => (
              <path
                key={`c-${i}`}
                d={d}
                fill="rgba(255,255,255,0.015)"
                stroke="rgba(255,255,255,0.04)"
                strokeWidth="0.2"
              />
            ))}

            {/* Inter-region mesh connections */}
            {regions.map((region: RegionData, i: number) => {
              const midX = (hubX + region.x) / 2;
              const midY = (hubY + region.y) / 2 - 8;
              return (
                <g key={`arc-${region.name}`}>
                  <motion.path
                    d={`M ${hubX} ${hubY} Q ${midX} ${midY} ${region.x} ${region.y}`}
                    fill="none"
                    stroke="url(#lineGrad)"
                    strokeWidth="0.25"
                    filter="url(#glow)"
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 0.7 }}
                    transition={{ duration: 1.2, delay: i * 0.15 }}
                  />
                  {/* Secondary faint line */}
                  <motion.path
                    d={`M ${hubX} ${hubY} Q ${midX} ${midY} ${region.x} ${region.y}`}
                    fill="none"
                    stroke="rgba(34,211,238,0.1)"
                    strokeWidth="0.8"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 1.5, delay: i * 0.15 }}
                  />
                </g>
              );
            })}

            {/* Data packets traveling along arcs */}
            {regions.map((region: RegionData, i: number) => {
              const midX = (hubX + region.x) / 2;
              const midY = (hubY + region.y) / 2 - 8;
              const pathD = `M ${hubX} ${hubY} Q ${midX} ${midY} ${region.x} ${region.y}`;
              return (
                <g key={`packet-${region.name}`}>
                  {[0, 1, 2].map((p) => (
                    <motion.circle
                      key={p}
                      r="0.35"
                      fill="#22d3ee"
                      filter="url(#glowStrong)"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: [0, 1, 1, 0] }}
                      transition={{
                        duration: 2.5,
                        delay: i * 0.4 + p * 0.8,
                        repeat: Infinity,
                        ease: "linear",
                      }}
                    >
                      <animateMotion
                        dur={`${2.5 + p * 0.3}s`}
                        repeatCount="indefinite"
                        begin={`${i * 0.4 + p * 0.8}s`}
                        path={pathD}
                      />
                    </motion.circle>
                  ))}
                </g>
              );
            })}

            {/* Central hub rings */}
            <circle cx={hubX} cy={hubY} r="6" fill="url(#hubGlow)" />
            <motion.circle
              cx={hubX}
              cy={hubY}
              r="3"
              fill="none"
              stroke="#22d3ee"
              strokeWidth="0.15"
              strokeOpacity="0.4"
              animate={{ r: [3, 5, 3], strokeOpacity: [0.4, 0, 0.4] }}
              transition={{ duration: 3, repeat: Infinity }}
            />
            <motion.circle
              cx={hubX}
              cy={hubY}
              r="4.5"
              fill="none"
              stroke="#22d3ee"
              strokeWidth="0.1"
              strokeOpacity="0.3"
              animate={{ r: [4.5, 7, 4.5], strokeOpacity: [0.3, 0, 0.3] }}
              transition={{ duration: 3, delay: 1, repeat: Infinity }}
            />
            <motion.circle
              cx={hubX}
              cy={hubY}
              r="1.2"
              fill="#22d3ee"
              filter="url(#glowStrong)"
              animate={{ r: [1.2, 1.5, 1.2] }}
              transition={{ duration: 2, repeat: Infinity }}
            />

            {/* Region nodes */}
            {regions.map((region: RegionData, i: number) => (
              <g key={region.name}>
                <motion.circle
                  cx={region.x}
                  cy={region.y}
                  r="2.5"
                  fill="rgba(167, 139, 250, 0.08)"
                  stroke="rgba(167, 139, 250, 0.3)"
                  strokeWidth="0.2"
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: 0.6 + i * 0.12, type: "spring" }}
                />
                <motion.circle
                  cx={region.x}
                  cy={region.y}
                  r="0.9"
                  fill="#a78bfa"
                  filter="url(#glow)"
                  animate={{ r: [0.9, 1.1, 0.9] }}
                  transition={{ duration: 2.5, delay: i * 0.3, repeat: Infinity }}
                />
                <text
                  x={region.x}
                  y={region.y + 4.5}
                  textAnchor="middle"
                  fill="rgba(255,255,255,0.45)"
                  fontSize="1.8"
                  fontFamily="monospace"
                >
                  {region.deviceCount}
                </text>
              </g>
            ))}
          </svg>

          {/* Floating overlay badge */}
          <div className="absolute bottom-4 left-4 flex items-center gap-2 rounded-xl border border-white/10 bg-black/60 px-3 py-2 backdrop-blur-md">
            <Globe2 className="h-4 w-4 text-cyan-400" />
            <span className="text-xs font-medium text-zinc-300">
              Aggregation Hub · US-East
            </span>
          </div>
        </div>

        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-5">
          {regions.map((region: RegionData, i: number) => (
            <motion.div
              key={region.name}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 + i * 0.05 }}
              className="group relative overflow-hidden rounded-xl border border-white/[0.06] bg-white/[0.02] px-4 py-3.5 text-center transition-colors hover:border-cyan-500/20 hover:bg-cyan-500/[0.03]"
            >
              <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-500/0 to-transparent transition-all group-hover:via-cyan-500/40" />
              <p className="font-mono text-xl font-bold text-zinc-100">
                {region.deviceCount}
              </p>
              <p className="mt-1 truncate text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                {region.name}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </GlassCard>
  );
}
