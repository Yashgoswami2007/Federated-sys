"use client";

import { motion } from "framer-motion";
import { Globe, Download, Rocket, History } from "lucide-react";
import { globalModel, accuracyTrend } from "@/lib/mock-data";
import { GlobalModelSection } from "@/components/dashboard/global-model-section";
import { GlassCard } from "@/components/shared/glass-card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

const modelHistory = [
  { version: "v3.2.1", accuracy: 94.7, date: "Jun 17, 2026", status: "active" },
  { version: "v3.2.0", accuracy: 94.3, date: "Jun 15, 2026", status: "archived" },
  { version: "v3.1.9", accuracy: 93.8, date: "Jun 12, 2026", status: "archived" },
  { version: "v3.1.8", accuracy: 93.2, date: "Jun 10, 2026", status: "archived" },
];

export default function GlobalModelsPage() {
  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between"
      >
        <div>
          <div className="flex items-center gap-2">
            <Globe className="h-6 w-6 text-cyan-400" />
            <h1 className="text-2xl font-bold text-zinc-100">Global Models</h1>
          </div>
          <p className="mt-1 text-sm text-zinc-500">
            Aggregated federated model versions and deployment
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" size="sm">
            <Download className="h-4 w-4" />
            Export All
          </Button>
          <Button size="sm">
            <Rocket className="h-4 w-4" />
            Deploy Latest
          </Button>
        </div>
      </motion.div>

      <div className="grid gap-4 sm:grid-cols-3">
        <GlassCard>
          <div className="p-4">
            <p className="font-mono text-lg font-bold text-cyan-400">
              {globalModel.name}
            </p>
            <p className="text-sm text-zinc-500">Current Model</p>
          </div>
        </GlassCard>
        <GlassCard delay={0.05}>
          <div className="p-4">
            <p className="text-2xl font-bold text-emerald-400">
              {globalModel.accuracy}%
            </p>
            <p className="text-sm text-zinc-500">Latest Accuracy</p>
          </div>
        </GlassCard>
        <GlassCard delay={0.1}>
          <div className="p-4">
            <p className="text-2xl font-bold text-zinc-100">
              {accuracyTrend.length}
            </p>
            <p className="text-sm text-zinc-500">Recent Data Points</p>
          </div>
        </GlassCard>
      </div>

      <GlobalModelSection />

      <GlassCard delay={0.15}>
        <div className="p-6">
          <div className="mb-4 flex items-center gap-2">
            <History className="h-5 w-5 text-zinc-400" />
            <h3 className="text-lg font-semibold text-zinc-100">
              Version History
            </h3>
          </div>
          <div className="space-y-2">
            {modelHistory.map((model) => (
              <div
                key={model.version}
                className="flex items-center justify-between rounded-lg border border-white/[0.04] bg-white/[0.02] px-4 py-3 transition-colors hover:bg-white/[0.04]"
              >
                <div className="flex items-center gap-3">
                  <span className="font-mono text-sm text-zinc-200">
                    {model.version}
                  </span>
                  <Badge
                    variant={model.status === "active" ? "success" : "secondary"}
                  >
                    {model.status}
                  </Badge>
                </div>
                <div className="flex items-center gap-4 text-sm">
                  <span className="text-emerald-400">{model.accuracy}%</span>
                  <span className="text-zinc-500">{model.date}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </GlassCard>
    </div>
  );
}
