"use client";

import { Monitor, Server, Gamepad2, HardDrive } from "lucide-react";
import { useWebSocket } from "@/lib/websocket-context";
import { formatRelativeTime } from "@/lib/utils";
import { GlassCard } from "@/components/shared/glass-card";
import { StatusIndicator } from "@/components/shared/status-indicator";
import { SectionHeader } from "@/components/shared/section-header";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";

const hardwareIcons: Record<string, React.ElementType> = {
  "MI300X": Server,
  "RX_7900_XTX": Server,
  "Steam_Deck": Gamepad2,
  "CPU_only": Monitor,
  "Steam Deck": Gamepad2,
  "Ryzen AI Laptop": Monitor,
  "Radeon Workstation": Server,
  "Edge Server": HardDrive,
};

export function DeviceTable() {
  const { devices } = useWebSocket();

  return (
    <GlassCard delay={0.15} glow="violet">
      <div className="p-7 lg:p-8">
        <SectionHeader
          eyebrow="Edge Fleet"
          title="Device Monitoring"
          description="Real-time telemetry from federated learning nodes"
          action={<Badge variant="secondary">{devices.length} devices</Badge>}
        />

        <div className="mt-6 overflow-x-auto -mx-2 px-2">
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead>Device Name</TableHead>
                <TableHead>Hardware Type</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>CPU Usage</TableHead>
                <TableHead>Memory Usage</TableHead>
                <TableHead>Last Sync</TableHead>
                <TableHead>Score</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {devices.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="h-24 text-center text-zinc-500">
                    No active devices. Launch a client to register.
                  </TableCell>
                </TableRow>
              ) : (
                devices.map((device) => {
                  const HwIcon = hardwareIcons[device.hardwareType] || Monitor;
                  return (
                    <TableRow key={device.id} className="group">
                      <TableCell>
                        <span className="font-mono text-sm font-medium text-zinc-200 group-hover:text-cyan-400 transition-colors">
                          {device.name}
                        </span>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2.5">
                          <div className="flex h-7 w-7 items-center justify-center rounded-lg border border-white/[0.06] bg-white/[0.03]">
                            <HwIcon className="h-3.5 w-3.5 text-zinc-500" />
                          </div>
                          <span className="text-sm text-zinc-400">{device.hardwareType}</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <StatusIndicator status={device.status} variant="pill" />
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-3 min-w-[120px]">
                          <Progress
                            value={device.cpuUsage}
                            className="h-1.5 w-20"
                            indicatorClassName={
                              device.cpuUsage > 80
                                ? "from-amber-400 to-orange-500 shadow-[0_0_8px_rgba(251,191,36,0.4)]"
                                : "shadow-[0_0_8px_rgba(34,211,238,0.3)]"
                            }
                          />
                          <span className="font-mono text-xs text-zinc-500">
                            {Math.round(device.cpuUsage)}%
                          </span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-3 min-w-[120px]">
                          <Progress value={device.memoryUsage} className="h-1.5 w-20" />
                          <span className="font-mono text-xs text-zinc-500">
                            {Math.round(device.memoryUsage)}%
                          </span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <span className="text-xs text-zinc-500">
                          {formatRelativeTime(device.lastSync)}
                        </span>
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant={
                            device.contributionScore >= 90
                              ? "success"
                              : device.contributionScore >= 75
                                ? "default"
                                : "secondary"
                          }
                        >
                          {device.contributionScore}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  );
                })
              )}
            </TableBody>
          </Table>
        </div>
      </div>
    </GlassCard>
  );
}
