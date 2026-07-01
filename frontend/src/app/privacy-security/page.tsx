"use client";

import { motion } from "framer-motion";
import { Shield, AlertTriangle, FileCheck, Key } from "lucide-react";
import { privacyMetrics } from "@/lib/mock-data";
import { PrivacySecurityPanel } from "@/components/dashboard/privacy-security-panel";
import { GlassCard } from "@/components/shared/glass-card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

const securityChecks = [
  { name: "TLS 1.3 Encryption", status: "pass", detail: "All channels encrypted" },
  { name: "Certificate Pinning", status: "pass", detail: "Valid until Dec 2026" },
  { name: "Model Integrity Hash", status: "pass", detail: "SHA-256 verified" },
  { name: "Audit Log Retention", status: "warning", detail: "87% capacity used" },
  { name: "Access Control (RBAC)", status: "pass", detail: "12 roles configured" },
  { name: "Anomaly Detection", status: "pass", detail: "0 alerts in 24h" },
];

export default function PrivacySecurityPage() {
  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center gap-2">
          <Shield className="h-6 w-6 text-emerald-400" />
          <h1 className="text-2xl font-bold text-zinc-100">
            Privacy & Security
          </h1>
        </div>
        <p className="mt-1 text-sm text-zinc-500">
          Differential privacy, secure aggregation, and compliance monitoring
        </p>
      </motion.div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <GlassCard>
          <div className="p-4">
            <div className="flex items-center gap-2">
              <Key className="h-4 w-4 text-cyan-400" />
              <span className="text-sm text-zinc-500">Privacy Budget</span>
            </div>
            <p className="mt-2 text-2xl font-bold text-cyan-400">
              {privacyMetrics.epsilonBudget}%
            </p>
            <Progress
              value={privacyMetrics.epsilonBudget}
              className="mt-2 h-1.5"
            />
          </div>
        </GlassCard>
        <GlassCard delay={0.05}>
          <div className="p-4">
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4 text-emerald-400" />
              <span className="text-sm text-zinc-500">Security Score</span>
            </div>
            <p className="mt-2 text-2xl font-bold text-emerald-400">
              {privacyMetrics.securityScore}/100
            </p>
          </div>
        </GlassCard>
        <GlassCard delay={0.1}>
          <div className="p-4">
            <div className="flex items-center gap-2">
              <FileCheck className="h-4 w-4 text-violet-400" />
              <span className="text-sm text-zinc-500">SecAgg Protocol</span>
            </div>
            <p className="mt-2 text-lg font-bold text-zinc-200">
              {privacyMetrics.secureAggregation.protocol}
            </p>
            <Badge variant="success" className="mt-1">
              Active
            </Badge>
          </div>
        </GlassCard>
        <GlassCard delay={0.15}>
          <div className="p-4">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-amber-400" />
              <span className="text-sm text-zinc-500">Active Alerts</span>
            </div>
            <p className="mt-2 text-2xl font-bold text-zinc-100">0</p>
            <p className="text-xs text-zinc-500">All systems nominal</p>
          </div>
        </GlassCard>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <PrivacySecurityPanel />

        <GlassCard delay={0.2}>
          <div className="p-6">
            <h3 className="mb-4 text-lg font-semibold text-zinc-100">
              Security Audit
            </h3>
            <div className="space-y-2">
              {securityChecks.map((check) => (
                <div
                  key={check.name}
                  className="flex items-center justify-between rounded-lg border border-white/[0.04] bg-white/[0.02] px-4 py-3"
                >
                  <div>
                    <p className="text-sm text-zinc-300">{check.name}</p>
                    <p className="text-xs text-zinc-600">{check.detail}</p>
                  </div>
                  <Badge
                    variant={
                      check.status === "pass"
                        ? "success"
                        : check.status === "warning"
                          ? "warning"
                          : "destructive"
                    }
                  >
                    {check.status}
                  </Badge>
                </div>
              ))}
            </div>
          </div>
        </GlassCard>
      </div>
    </div>
  );
}
