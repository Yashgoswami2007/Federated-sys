"use client";

import {
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";
import {
  analyticsAccuracy,
  deviceParticipation,
  trainingThroughput,
  resourceUtilization,
} from "@/lib/mock-data";
import { GlassCard } from "@/components/shared/glass-card";
import { SectionHeader } from "@/components/shared/section-header";
import {
  CHART_COLORS,
  axisTickStyle,
  gridProps,
  ChartTooltip,
  ChartContainer,
  type ChartTooltipProps,
} from "@/components/charts/chart-theme";

const charts = [
  {
    title: "Model Accuracy",
    subtitle: "6-month trend",
    type: "line" as const,
    data: analyticsAccuracy,
    dataKey: "value",
    color: CHART_COLORS.cyan,
    yFormatter: (v: number) => `${v}%`,
    domain: [85, 100] as [number, number],
    refLine: 90,
  },
  {
    title: "Device Participation",
    subtitle: "Weekly active nodes",
    type: "bar" as const,
    data: deviceParticipation,
    dataKey: "value",
    color: CHART_COLORS.violet,
  },
  {
    title: "Training Throughput",
    subtitle: "GB/h processed",
    type: "area" as const,
    data: trainingThroughput,
    dataKey: "value",
    color: CHART_COLORS.cyan,
  },
  {
    title: "Resource Utilization",
    subtitle: "Current vs average",
    type: "bar" as const,
    data: resourceUtilization,
    dataKey: "value",
    color: CHART_COLORS.cyan,
    dataKey2: "value2",
    color2: CHART_COLORS.violet,
  },
];

export function AnalyticsSection() {
  return (
    <GlassCard delay={0.35} glow="cyan">
      <div className="p-7 lg:p-8">
        <SectionHeader
          eyebrow="Performance Intelligence"
          title="Analytics"
          description="Network-wide metrics and resource utilization trends"
        />

        <div className="mt-6 grid gap-5 lg:grid-cols-2">
          {charts.map((chart) => (
            <ChartContainer
              key={chart.title}
              title={chart.title}
              subtitle={chart.subtitle}
              height={180}
            >
              <ResponsiveContainer width="100%" height="100%">
                {chart.type === "line" ? (
                  <LineChart data={chart.data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="label" tick={axisTickStyle} axisLine={false} tickLine={false} />
                    <YAxis
                      tick={axisTickStyle}
                      axisLine={false}
                      tickLine={false}
                      domain={chart.domain}
                      tickFormatter={chart.yFormatter}
                      width={36}
                    />
                    {chart.refLine && (
                      <ReferenceLine y={chart.refLine} stroke="rgba(255,255,255,0.06)" strokeDasharray="4 4" />
                    )}
                    <Tooltip
                      content={(props) => (
                        <ChartTooltip
                          active={props.active}
                          payload={props.payload as ChartTooltipProps["payload"]}
                          label={props.label}
                          valueFormatter={(v) => `${v}%`}
                        />
                      )}
                    />
                    <Line
                      type="monotone"
                      dataKey={chart.dataKey}
                      name="Accuracy"
                      stroke={chart.color}
                      strokeWidth={2.5}
                      dot={false}
                      activeDot={{ r: 4, fill: chart.color, stroke: "#060608", strokeWidth: 2 }}
                    />
                  </LineChart>
                ) : chart.type === "area" ? (
                  <AreaChart data={chart.data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id="throughputGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={chart.color} stopOpacity={0.3} />
                        <stop offset="100%" stopColor={chart.color} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="label" tick={axisTickStyle} axisLine={false} tickLine={false} />
                    <YAxis tick={axisTickStyle} axisLine={false} tickLine={false} width={36} />
                    <Tooltip
                      content={(props) => (
                        <ChartTooltip
                          active={props.active}
                          payload={props.payload as ChartTooltipProps["payload"]}
                          label={props.label}
                        />
                      )}
                    />
                    <Area
                      type="monotone"
                      dataKey={chart.dataKey}
                      name="Throughput"
                      stroke={chart.color}
                      fill="url(#throughputGrad)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                ) : (
                  <BarChart data={chart.data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }} barGap={4}>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="label" tick={axisTickStyle} axisLine={false} tickLine={false} />
                    <YAxis tick={axisTickStyle} axisLine={false} tickLine={false} width={36} />
                    <Tooltip
                      content={(props) => (
                        <ChartTooltip
                          active={props.active}
                          payload={props.payload as ChartTooltipProps["payload"]}
                          label={props.label}
                        />
                      )}
                    />
                    {chart.dataKey2 && (
                      <Legend
                        wrapperStyle={{ fontSize: 10, paddingTop: 8 }}
                        formatter={(v) => <span className="text-zinc-500">{v}</span>}
                      />
                    )}
                    <Bar
                      dataKey={chart.dataKey}
                      name="Current"
                      fill={chart.color}
                      radius={[6, 6, 0, 0]}
                      maxBarSize={32}
                      fillOpacity={0.9}
                    />
                    {chart.dataKey2 && (
                      <Bar
                        dataKey={chart.dataKey2}
                        name="Average"
                        fill={chart.color2}
                        radius={[6, 6, 0, 0]}
                        maxBarSize={32}
                        fillOpacity={0.5}
                      />
                    )}
                  </BarChart>
                )}
              </ResponsiveContainer>
            </ChartContainer>
          ))}
        </div>
      </div>
    </GlassCard>
  );
}
