"use client";

import { motion } from "framer-motion";
import {
  UserPlus,
  CheckCircle,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import { recentActivity } from "@/lib/mock-data";
import { formatRelativeTime } from "@/lib/utils";
import { GlassCard } from "@/components/shared/glass-card";
import { SectionHeader } from "@/components/shared/section-header";
import type { ActivityItem } from "@/types";

const activityIcons: Record<
  ActivityItem["type"],
  { icon: React.ElementType; color: string; border: string }
> = {
  device_joined: {
    icon: UserPlus,
    color: "text-cyan-400 bg-cyan-500/10",
    border: "border-cyan-500/10 hover:border-cyan-500/20",
  },
  round_completed: {
    icon: CheckCircle,
    color: "text-emerald-400 bg-emerald-500/10",
    border: "border-emerald-500/10 hover:border-emerald-500/20",
  },
  model_updated: {
    icon: RefreshCw,
    color: "text-violet-400 bg-violet-500/10",
    border: "border-violet-500/10 hover:border-violet-500/20",
  },
  security_verified: {
    icon: ShieldCheck,
    color: "text-amber-400 bg-amber-500/10",
    border: "border-amber-500/10 hover:border-amber-500/20",
  },
};

export function ActivityFeed() {
  return (
    <GlassCard delay={0.4} glow="cyan">
      <div className="p-7 lg:p-8">
        <SectionHeader
          eyebrow="Event Stream"
          title="Recent Activity"
          description="Live network events and system notifications"
        />

        <div className="relative mt-6 space-y-0">
          <div className="absolute left-[19px] top-4 bottom-4 w-px bg-gradient-to-b from-cyan-500/30 via-white/5 to-transparent" />

          {recentActivity.map((item, index) => {
            const config = activityIcons[item.type];
            const Icon = config.icon;

            return (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 + index * 0.06, ease: [0.22, 1, 0.36, 1] }}
                className={`relative flex items-start gap-4 rounded-xl border bg-white/[0.02] p-4 mb-3 transition-all duration-300 hover:bg-white/[0.04] ${config.border}`}
              >
                <div
                  className={`relative z-10 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-white/[0.06] ${config.color}`}
                >
                  <Icon className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1 pt-0.5">
                  <p className="text-sm leading-relaxed text-zinc-300">{item.message}</p>
                  <p className="mt-1.5 font-mono text-[11px] text-zinc-600">
                    {formatRelativeTime(item.timestamp)}
                  </p>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </GlassCard>
  );
}
