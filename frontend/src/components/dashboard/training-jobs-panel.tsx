"use client";

import { Clock, Users } from "lucide-react";
import { trainingJobs } from "@/lib/mock-data";
import { GlassCard } from "@/components/shared/glass-card";
import { SectionHeader } from "@/components/shared/section-header";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";

export function TrainingJobsPanel() {
  const activeJob = trainingJobs.find((j) => j.status === "running");

  return (
    <GlassCard delay={0.2} glow="violet">
      <div className="flex h-full flex-col p-7 lg:p-8">
        <SectionHeader
          eyebrow="Orchestration"
          title="Training Jobs"
          description="Active federated learning round"
        />

        {activeJob && (
          <div className="mt-6 flex flex-1 flex-col space-y-5">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-widest text-zinc-500">
                  Current Round
                </p>
                <p className="mt-1 font-mono text-3xl font-bold tracking-tight text-zinc-50">
                  {activeJob.round}
                  <span className="text-base font-normal text-zinc-600">
                    {" "}/ {activeJob.totalRounds}
                  </span>
                </p>
              </div>
              <Badge variant="purple" className="gap-1.5">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-violet-400 opacity-75" />
                  <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-violet-400 glow-dot-violet" />
                </span>
                Running
              </Badge>
            </div>

            <div className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-4">
              <div className="mb-3 flex items-center justify-between">
                <span className="text-xs font-medium text-zinc-500">Round Progress</span>
                <span className="font-mono text-sm font-semibold text-cyan-400">
                  {activeJob.progress}%
                </span>
              </div>
              <Progress value={activeJob.progress} className="h-2.5" />
              <div className="mt-2 h-px bg-gradient-to-r from-cyan-500/50 via-violet-500/50 to-transparent animate-pulse-glow" />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4">
                <div className="flex items-center gap-2 text-zinc-500">
                  <Clock className="h-3.5 w-3.5 text-cyan-400" />
                  <span className="text-[10px] font-semibold uppercase tracking-wider">ETA</span>
                </div>
                <p className="mt-2 font-mono text-lg font-semibold text-zinc-100">~2h 15m</p>
              </div>
              <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4">
                <div className="flex items-center gap-2 text-zinc-500">
                  <Users className="h-3.5 w-3.5 text-violet-400" />
                  <span className="text-[10px] font-semibold uppercase tracking-wider">Nodes</span>
                </div>
                <p className="mt-2 font-mono text-lg font-semibold text-zinc-100">
                  {activeJob.participatingDevices}
                </p>
              </div>
            </div>

            <div className="mt-auto flex items-center justify-between rounded-xl border border-cyan-500/10 bg-cyan-500/[0.04] px-4 py-3.5">
              <span className="text-xs font-medium text-zinc-500">Model Version</span>
              <span className="font-mono text-sm font-semibold text-cyan-400">
                {activeJob.modelVersion}
              </span>
            </div>
          </div>
        )}

        <div className="mt-5 space-y-2">
          {trainingJobs
            .filter((j) => j.status !== "running")
            .map((job) => (
              <div
                key={job.id}
                className="flex items-center justify-between rounded-xl border border-white/[0.04] bg-white/[0.02] px-4 py-3 transition-colors hover:border-white/[0.08]"
              >
                <div>
                  <p className="text-sm font-medium text-zinc-300">
                    Round {job.round}
                  </p>
                  <p className="font-mono text-xs text-zinc-600">{job.modelVersion}</p>
                </div>
                <Badge variant="secondary">Queued</Badge>
              </div>
            ))}
        </div>
      </div>
    </GlassCard>
  );
}
