"use client";

import { motion } from "framer-motion";
import { Shield, Lock, Eye, CheckCircle2 } from "lucide-react";
import { privacyMetrics } from "@/lib/mock-data";
import { GlassCard } from "@/components/shared/glass-card";
import { SectionHeader } from "@/components/shared/section-header";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";

export function PrivacySecurityPanel() {
  const { differentialPrivacy, epsilonBudget, secureAggregation, securityScore } =
    privacyMetrics;

  return (
    <GlassCard delay={0.3} glow="emerald">
      <div className="flex h-full flex-col p-7 lg:p-8">
        <SectionHeader
          eyebrow="Compliance"
          title="Privacy & Security"
          description="Enterprise-grade protection layer"
        />

        <div className="mt-6 flex flex-1 flex-col space-y-4">
          <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-cyan-500/10">
                  <Lock className="h-4 w-4 text-cyan-400" />
                </div>
                <span className="text-sm font-medium text-zinc-300">
                  Differential Privacy
                </span>
              </div>
              <Badge variant="success">Active</Badge>
            </div>
            <p className="mt-3 font-mono text-xs text-zinc-500">
              ε = {differentialPrivacy.epsilon} · δ = {differentialPrivacy.delta}
            </p>
          </div>

          <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-sm font-medium text-zinc-300">Epsilon Budget</span>
              <span className="font-mono text-sm font-semibold text-cyan-400">
                {epsilonBudget}%
              </span>
            </div>
            <Progress
              value={epsilonBudget}
              className="h-2.5"
              indicatorClassName="from-emerald-400 to-cyan-400 shadow-[0_0_12px_rgba(52,211,153,0.4)]"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-xl border border-emerald-500/10 bg-emerald-500/[0.03] p-3.5">
              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
              <p className="mt-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                SecAgg
              </p>
              <p className="mt-1 text-sm font-semibold text-zinc-200">
                {secureAggregation.protocol}
              </p>
            </div>
            <div className="rounded-xl border border-violet-500/10 bg-violet-500/[0.03] p-3.5">
              <Eye className="h-4 w-4 text-violet-400" />
              <p className="mt-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                ZK Proof
              </p>
              <p className="mt-1 text-sm font-semibold text-emerald-400">Verified</p>
            </div>
          </div>

          <div className="relative mt-auto overflow-hidden rounded-2xl border border-emerald-500/20 bg-gradient-to-br from-emerald-500/10 via-transparent to-cyan-500/5 p-5">
            <div className="absolute -right-4 -top-4 h-24 w-24 rounded-full bg-emerald-500/10 blur-2xl" />
            <div className="relative flex items-center justify-between">
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
                  Security Score
                </p>
                <p className="mt-1 font-mono text-4xl font-bold text-emerald-400">
                  {securityScore}
                  <span className="text-lg font-normal text-zinc-600">/100</span>
                </p>
              </div>
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                className="flex h-16 w-16 items-center justify-center rounded-full border border-emerald-500/20"
              >
                <Shield className="h-7 w-7 text-emerald-400" />
              </motion.div>
            </div>
          </div>
        </div>
      </div>
    </GlassCard>
  );
}
