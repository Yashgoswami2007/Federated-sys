"use client";

import { motion } from "framer-motion";
import {
  Settings,
  Bell,
  Shield,
  Network,
  Database,
  Palette,
} from "lucide-react";
import { GlassCard } from "@/components/shared/glass-card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";

const settingsSections = [
  {
    icon: Network,
    title: "Network Configuration",
    description: "Aggregation server, sync intervals, and node discovery",
  },
  {
    icon: Shield,
    title: "Security Policies",
    description: "Privacy budgets, encryption, and access control",
  },
  {
    icon: Bell,
    title: "Notifications",
    description: "Alerts for training events, device status, and security",
  },
  {
    icon: Database,
    title: "Data Retention",
    description: "Model checkpoints, logs, and audit trail settings",
  },
  {
    icon: Palette,
    title: "Appearance",
    description: "Theme, density, and dashboard layout preferences",
  },
];

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center gap-2">
          <Settings className="h-6 w-6 text-zinc-400" />
          <h1 className="text-2xl font-bold text-zinc-100">Settings</h1>
        </div>
        <p className="mt-1 text-sm text-zinc-500">
          Configure your FusionNet network and preferences
        </p>
      </motion.div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-4">
          {settingsSections.map((section, i) => {
            const Icon = section.icon;
            return (
              <GlassCard key={section.title} delay={i * 0.05}>
                <div className="p-6">
                  <div className="flex items-start gap-4">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-white/[0.04] border border-white/[0.06]">
                      <Icon className="h-5 w-5 text-cyan-400" />
                    </div>
                    <div className="flex-1">
                      <h3 className="font-semibold text-zinc-100">
                        {section.title}
                      </h3>
                      <p className="mt-1 text-sm text-zinc-500">
                        {section.description}
                      </p>
                      <Separator className="my-4" />
                      <div className="grid gap-3 sm:grid-cols-2">
                        <div>
                          <label className="text-xs text-zinc-500">
                            Setting
                          </label>
                          <Input
                            placeholder="Configure..."
                            className="mt-1"
                            defaultValue={
                              section.title === "Network Configuration"
                                ? "agg.fusionnet.io:8443"
                                : section.title === "Security Policies"
                                  ? "ε = 4.0, δ = 1e-5"
                                  : ""
                            }
                          />
                        </div>
                        <div>
                          <label className="text-xs text-zinc-500">
                            Status
                          </label>
                          <div className="mt-2">
                            <Badge variant="success">Configured</Badge>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </GlassCard>
            );
          })}
        </div>

        <div className="space-y-4">
          <GlassCard>
            <div className="p-6">
              <h3 className="font-semibold text-zinc-100">Account</h3>
              <Separator className="my-4" />
              <div className="space-y-3">
                <div>
                  <label className="text-xs text-zinc-500">Name</label>
                  <Input defaultValue="Dr. Sarah Chen" className="mt-1" />
                </div>
                <div>
                  <label className="text-xs text-zinc-500">Email</label>
                  <Input
                    defaultValue="sarah.chen@fusionnet.io"
                    className="mt-1"
                  />
                </div>
                <div>
                  <label className="text-xs text-zinc-500">Role</label>
                  <div className="mt-2">
                    <Badge variant="purple">Network Admin</Badge>
                  </div>
                </div>
              </div>
              <Button className="mt-4 w-full" size="sm">
                Save Changes
              </Button>
            </div>
          </GlassCard>

          <GlassCard delay={0.1}>
            <div className="p-6">
              <h3 className="font-semibold text-zinc-100">System Info</h3>
              <Separator className="my-4" />
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-zinc-500">Version</span>
                  <span className="font-mono text-zinc-300">v2.4.1</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Environment</span>
                  <Badge variant="success">Production</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Region</span>
                  <span className="text-zinc-300">us-east-1</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Uptime</span>
                  <span className="text-zinc-300">47d 12h</span>
                </div>
              </div>
            </div>
          </GlassCard>
        </div>
      </div>
    </div>
  );
}
