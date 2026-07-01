"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
} from "recharts";
import { Download, Rocket } from "lucide-react";
import { globalModel, accuracyTrend, lossCurve } from "@/lib/mock-data";
import { GlassCard } from "@/components/shared/glass-card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { SectionHeader } from "@/components/shared/section-header";
import {
  CHART_COLORS,
  axisTickStyle,
  gridProps,
  ChartTooltip,
  ChartContainer,
  type ChartTooltipProps,
} from "@/components/charts/chart-theme";

export function GlobalModelSection() {
  return (
    <GlassCard delay={0.25} glow="violet">
      <div className="p-7 lg:p-8">
        <SectionHeader
          eyebrow="Model Registry"
          title="Global Model"
          description={`${globalModel.name} · Production deployment`}
          action={
            <div className="flex gap-2">
              <Button variant="secondary" size="sm">
                <Download className="h-4 w-4" />
                Download
              </Button>
              <Button size="sm">
                <Rocket className="h-4 w-4" />
                Deploy
              </Button>
            </div>
          }
        />

        <div className="mt-2 flex items-center gap-3">
          <span className="font-mono text-sm text-cyan-400">{globalModel.name}</span>
          <Badge variant="default">{globalModel.version}</Badge>
          <Badge variant="success">{globalModel.accuracy}% accuracy</Badge>
        </div>

        <div className="mt-6 grid gap-5 lg:grid-cols-2">
          <ChartContainer title="Accuracy Trend" subtitle="Last 8 training rounds" height={220}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={accuracyTrend} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                <defs>
                  <linearGradient id="accGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={CHART_COLORS.cyan} stopOpacity={0.25} />
                    <stop offset="100%" stopColor={CHART_COLORS.cyan} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid {...gridProps} />
                <XAxis
                  dataKey="label"
                  tick={axisTickStyle}
                  tickFormatter={(v) => v.replace("Round ", "R")}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  domain={[92.5, 95.5]}
                  tick={axisTickStyle}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v) => `${v}%`}
                  width={42}
                />
                <ReferenceLine y={94} stroke="rgba(255,255,255,0.06)" strokeDasharray="4 4" />
                <Tooltip
                  content={(props) => (
                    <ChartTooltip
                      active={props.active}
                      payload={props.payload as ChartTooltipProps["payload"]}
                      label={props.label}
                      valueFormatter={(v) => `${v}%`}
                      labelFormatter={(l) => l}
                    />
                  )}
                />
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke="none"
                  fill="url(#accGrad)"
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  name="Accuracy"
                  stroke={CHART_COLORS.cyan}
                  strokeWidth={2.5}
                  dot={false}
                  activeDot={{ r: 5, fill: CHART_COLORS.cyan, stroke: "#060608", strokeWidth: 2 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </ChartContainer>

          <ChartContainer title="Loss Curve" subtitle="Convergence trajectory" height={220}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lossCurve} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                <defs>
                  <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={CHART_COLORS.violet} stopOpacity={0.2} />
                    <stop offset="100%" stopColor={CHART_COLORS.violet} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid {...gridProps} />
                <XAxis
                  dataKey="label"
                  tick={axisTickStyle}
                  tickFormatter={(v) => v.replace("Round ", "R")}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={axisTickStyle}
                  axisLine={false}
                  tickLine={false}
                  domain={[0.04, 0.16]}
                  width={42}
                />
                <Tooltip
                  content={(props) => (
                    <ChartTooltip
                      active={props.active}
                      payload={props.payload as ChartTooltipProps["payload"]}
                      label={props.label}
                      valueFormatter={(v) => v.toFixed(3)}
                    />
                  )}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  name="Loss"
                  stroke={CHART_COLORS.violet}
                  strokeWidth={2.5}
                  dot={false}
                  activeDot={{ r: 5, fill: CHART_COLORS.violet, stroke: "#060608", strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>
        </div>
      </div>
    </GlassCard>
  );
}
