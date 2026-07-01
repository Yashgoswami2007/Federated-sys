"use client";

import { cn } from "@/lib/utils";
import type { DeviceStatus } from "@/types";

const statusConfig: Record<
  DeviceStatus,
  {
    label: string;
    dotClass: string;
    glowClass: string;
    textClass: string;
    pulse: boolean;
  }
> = {
  online: {
    label: "Online",
    dotClass: "bg-emerald-400",
    glowClass: "glow-dot-emerald",
    textClass: "text-emerald-400/90",
    pulse: true,
  },
  training: {
    label: "Training",
    dotClass: "bg-cyan-400",
    glowClass: "glow-dot-cyan",
    textClass: "text-cyan-400/90",
    pulse: true,
  },
  syncing: {
    label: "Syncing",
    dotClass: "bg-amber-400",
    glowClass: "glow-dot-amber",
    textClass: "text-amber-400/90",
    pulse: true,
  },
  offline: {
    label: "Offline",
    dotClass: "bg-zinc-600",
    glowClass: "",
    textClass: "text-zinc-500",
    pulse: false,
  },
};

interface StatusIndicatorProps {
  status: DeviceStatus;
  showLabel?: boolean;
  size?: "sm" | "md" | "lg";
  variant?: "default" | "pill";
}

export function StatusIndicator({
  status,
  showLabel = true,
  size = "sm",
  variant = "default",
}: StatusIndicatorProps) {
  const config = statusConfig[status];
  const dotSize =
    size === "sm" ? "h-2 w-2" : size === "md" ? "h-2.5 w-2.5" : "h-3 w-3";

  if (variant === "pill") {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-xs font-medium",
          status === "online" && "border-emerald-500/20 bg-emerald-500/10",
          status === "training" && "border-cyan-500/20 bg-cyan-500/10",
          status === "syncing" && "border-amber-500/20 bg-amber-500/10",
          status === "offline" && "border-zinc-700/50 bg-zinc-800/30",
          config.textClass
        )}
      >
        <span className="relative flex">
          {config.pulse && (
            <span
              className={cn(
                "absolute inline-flex h-full w-full animate-ping rounded-full opacity-60",
                config.dotClass
              )}
            />
          )}
          <span
            className={cn(
              "relative inline-flex rounded-full",
              dotSize,
              config.dotClass,
              config.glowClass
            )}
          />
        </span>
        {showLabel && config.label}
      </span>
    );
  }

  return (
    <div className="flex items-center gap-2.5">
      <span className="relative flex">
        {config.pulse && (
          <span
            className={cn(
              "absolute inline-flex h-full w-full animate-ping rounded-full opacity-50",
              config.dotClass
            )}
          />
        )}
        <span
          className={cn(
            "relative inline-flex rounded-full ring-2 ring-white/5",
            dotSize,
            config.dotClass,
            config.glowClass
          )}
        />
      </span>
      {showLabel && (
        <span className={cn("text-xs font-medium", config.textClass)}>
          {config.label}
        </span>
      )}
    </div>
  );
}
