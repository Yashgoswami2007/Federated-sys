"use client";

import { motion } from "framer-motion";
import { BarChart3, Download, Calendar } from "lucide-react";
import { AnalyticsSection } from "@/components/dashboard/analytics-section";
import { GlassCard } from "@/components/shared/glass-card";
import { Button } from "@/components/ui/button";
import {
  analyticsAccuracy,
  deviceParticipation,
  trainingThroughput,
} from "@/lib/mock-data";

export default function AnalyticsPage() {
  const latestAccuracy =
    analyticsAccuracy[analyticsAccuracy.length - 1]?.value ?? 0;
  const peakParticipation = Math.max(
    ...deviceParticipation.map((d) => d.value)
  );
  const avgThroughput =
    trainingThroughput.reduce((a, b) => a + b.value, 0) /
    trainingThroughput.length;

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between"
      >
        <div>
          <div className="flex items-center gap-2">
            <BarChart3 className="h-6 w-6 text-cyan-400" />
            <h1 className="text-2xl font-bold text-zinc-100">Analytics</h1>
          </div>
          <p className="mt-1 text-sm text-zinc-500">
            Network performance, training metrics, and resource utilization
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" size="sm">
            <Calendar className="h-4 w-4" />
            Last 30 Days
          </Button>
          <Button variant="secondary" size="sm">
            <Download className="h-4 w-4" />
            Export Report
          </Button>
        </div>
      </motion.div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <GlassCard>
          <div className="p-4">
            <p className="text-2xl font-bold text-emerald-400">
              {latestAccuracy}%
            </p>
            <p className="text-sm text-zinc-500">Current Accuracy</p>
          </div>
        </GlassCard>
        <GlassCard delay={0.05}>
          <div className="p-4">
            <p className="text-2xl font-bold text-violet-400">
              {peakParticipation}
            </p>
            <p className="text-sm text-zinc-500">Peak Participation</p>
          </div>
        </GlassCard>
        <GlassCard delay={0.1}>
          <div className="p-4">
            <p className="text-2xl font-bold text-cyan-400">
              {avgThroughput.toFixed(0)}
            </p>
            <p className="text-sm text-zinc-500">Avg. Throughput (GB/h)</p>
          </div>
        </GlassCard>
        <GlassCard delay={0.15}>
          <div className="p-4">
            <p className="text-2xl font-bold text-zinc-100">99.7%</p>
            <p className="text-sm text-zinc-500">Network Uptime</p>
          </div>
        </GlassCard>
      </div>

      <AnalyticsSection />
    </div>
  );
}
