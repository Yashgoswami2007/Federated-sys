"use client";

import { motion } from "framer-motion";
import { Layers, Play, Pause, Plus } from "lucide-react";
import { trainingJobs } from "@/lib/mock-data";
import { GlassCard } from "@/components/shared/glass-card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

const statusVariant = {
  running: "purple" as const,
  queued: "secondary" as const,
  completed: "success" as const,
};

export default function TrainingJobsPage() {
  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between"
      >
        <div>
          <div className="flex items-center gap-2">
            <Layers className="h-6 w-6 text-violet-400" />
            <h1 className="text-2xl font-bold text-zinc-100">Training Jobs</h1>
          </div>
          <p className="mt-1 text-sm text-zinc-500">
            Federated learning rounds and job orchestration
          </p>
        </div>
        <Button size="sm">
          <Plus className="h-4 w-4" />
          New Training Round
        </Button>
      </motion.div>

      <div className="grid gap-4 sm:grid-cols-3">
        <GlassCard>
          <div className="p-4">
            <p className="text-2xl font-bold text-violet-400">1</p>
            <p className="text-sm text-zinc-500">Active Jobs</p>
          </div>
        </GlassCard>
        <GlassCard delay={0.05}>
          <div className="p-4">
            <p className="text-2xl font-bold text-zinc-100">1,842</p>
            <p className="text-sm text-zinc-500">Completed Rounds</p>
          </div>
        </GlassCard>
        <GlassCard delay={0.1}>
          <div className="p-4">
            <p className="text-2xl font-bold text-cyan-400">198</p>
            <p className="text-sm text-zinc-500">Avg. Participants</p>
          </div>
        </GlassCard>
      </div>

      <div className="space-y-4">
        {trainingJobs.map((job, i) => (
          <GlassCard key={job.id} delay={i * 0.05}>
            <div className="p-6">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <div className="flex items-center gap-3">
                    <h3 className="text-lg font-semibold text-zinc-100">
                      Round {job.round}
                    </h3>
                    <Badge variant={statusVariant[job.status]}>
                      {job.status}
                    </Badge>
                    <span className="font-mono text-sm text-cyan-400">
                      {job.modelVersion}
                    </span>
                  </div>
                  <p className="mt-1 text-sm text-zinc-500">
                    {job.participatingDevices} devices participating
                  </p>
                </div>
                <div className="flex gap-2">
                  {job.status === "running" ? (
                    <Button variant="secondary" size="sm">
                      <Pause className="h-4 w-4" />
                      Pause
                    </Button>
                  ) : (
                    <Button size="sm">
                      <Play className="h-4 w-4" />
                      Start
                    </Button>
                  )}
                </div>
              </div>

              {job.status === "running" && (
                <div className="mt-4">
                  <div className="mb-2 flex justify-between text-sm">
                    <span className="text-zinc-500">Progress</span>
                    <span className="text-cyan-400">{job.progress}%</span>
                  </div>
                  <Progress value={job.progress} className="h-2" />
                  <p className="mt-2 text-xs text-zinc-600">
                    Estimated completion: ~2h 15m remaining
                  </p>
                </div>
              )}
            </div>
          </GlassCard>
        ))}
      </div>
    </div>
  );
}
