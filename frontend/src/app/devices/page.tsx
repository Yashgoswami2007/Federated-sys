"use client";

import { motion } from "framer-motion";
import { Cpu, Plus, Filter } from "lucide-react";
import { useWebSocket } from "@/lib/websocket-context";
import { DeviceTable } from "@/components/dashboard/device-table";
import { GlassCard } from "@/components/shared/glass-card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { StatusIndicator } from "@/components/shared/status-indicator";
import type { EdgeDevice } from "@/types";

export default function DevicesPage() {
  const { devices, regions } = useWebSocket();
  const onlineCount = devices.filter(
    (d) => d.status === "online" || d.status === "training"
  ).length;

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between"
      >
        <div>
          <div className="flex items-center gap-2">
            <Cpu className="h-6 w-6 text-cyan-400" />
            <h1 className="text-2xl font-bold text-zinc-100">Edge Devices</h1>
          </div>
          <p className="mt-1 text-sm text-zinc-500">
            Monitor and manage federated learning edge nodes
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" size="sm">
            <Filter className="h-4 w-4" />
            Filter
          </Button>
          <Button size="sm">
            <Plus className="h-4 w-4" />
            Register Device
          </Button>
        </div>
      </motion.div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <GlassCard>
          <div className="p-4">
            <p className="text-2xl font-bold text-zinc-100">{devices.length}</p>
            <p className="text-sm text-zinc-500">Total Devices</p>
          </div>
        </GlassCard>
        <GlassCard delay={0.05}>
          <div className="p-4">
            <p className="text-2xl font-bold text-emerald-400">{onlineCount}</p>
            <p className="text-sm text-zinc-500">Online / Training</p>
          </div>
        </GlassCard>
        <GlassCard delay={0.1}>
          <div className="p-4">
            <p className="text-2xl font-bold text-cyan-400">
              {regions.length}
            </p>
            <p className="text-sm text-zinc-500">Active Regions</p>
          </div>
        </GlassCard>
        <GlassCard delay={0.15}>
          <div className="p-4">
            <p className="text-2xl font-bold text-violet-400">89</p>
            <p className="text-sm text-zinc-500">Avg. Contribution</p>
          </div>
        </GlassCard>
      </div>

      <DeviceTable />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {devices.slice(0, 3).map((device: EdgeDevice, i: number) => (
          <GlassCard key={device.id} delay={i * 0.05}>
            <div className="p-5">
              <div className="flex items-center justify-between">
                <span className="font-mono text-sm text-zinc-200">
                  {device.name}
                </span>
                <StatusIndicator status={device.status} />
              </div>
              <p className="mt-2 text-xs text-zinc-500">{device.hardwareType}</p>
              <div className="mt-3 flex items-center gap-2">
                <Badge variant="secondary">{device.region}</Badge>
                <Badge
                  variant={
                    device.contributionScore >= 90 ? "success" : "default"
                  }
                >
                  Score: {device.contributionScore}
                </Badge>
              </div>
            </div>
          </GlassCard>
        ))}
      </div>
    </div>
  );
}
