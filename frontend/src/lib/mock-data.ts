import { EdgeDevice, TrainingJob, GlobalModel, ActivityItem, RegionData, ChartDataPoint, KpiMetric } from "@/types";

// Empty placeholders for real-time integrated data (will be replaced by context/hooks)
export const edgeDevices: EdgeDevice[] = [];
export const regionData: RegionData[] = [];

// Static mock data for other views and dashboard components
export const kpiMetrics: KpiMetric[] = [
  { label: "Active Nodes", value: "247", change: "+12.3% vs last week", trend: "up", icon: "layers" },
  { label: "Training Accuracy", value: "85.2%", change: "+0.4% improvement", trend: "up", icon: "target" },
  { label: "SecAgg Coverage", value: "100%", change: "Fully verified", trend: "neutral", icon: "shield" },
  { label: "Avg Round Time", value: "4.2 mins", change: "-1.5% speedup", trend: "up", icon: "cpu" }
];

export const trainingJobs: TrainingJob[] = [
  {
    id: "job_1",
    round: 4,
    totalRounds: 10,
    progress: 75,
    estimatedCompletion: "2h 15m",
    participatingDevices: 184,
    modelVersion: "v1.2.0",
    status: "running",
  },
  {
    id: "job_2",
    round: 5,
    totalRounds: 10,
    progress: 0,
    estimatedCompletion: "Unknown",
    participatingDevices: 0,
    modelVersion: "v1.2.0",
    status: "queued",
  }
];

export const privacyMetrics = {
  differentialPrivacy: {
    epsilon: 1.0,
    delta: 0.00001,
  },
  epsilonBudget: 78,
  secureAggregation: {
    protocol: "Shamirs Secret Sharing",
  },
  securityScore: 98,
};

export const globalModel: GlobalModel = {
  name: "TinyLlama-1.1B",
  version: "v1.2.0",
  accuracy: 85.2,
  lastUpdated: new Date().toISOString()
};

export const accuracyTrend: ChartDataPoint[] = [
  { label: "Round 1", value: 50.5 },
  { label: "Round 2", value: 65.1 },
  { label: "Round 3", value: 78.8 },
  { label: "Round 4", value: 85.2 }
];

export const lossCurve: ChartDataPoint[] = [
  { label: "Round 1", value: 0.15 },
  { label: "Round 2", value: 0.12 },
  { label: "Round 3", value: 0.09 },
  { label: "Round 4", value: 0.07 }
];

export const analyticsAccuracy: ChartDataPoint[] = [
  { label: "Round 1", value: 50.5 },
  { label: "Round 2", value: 65.1 },
  { label: "Round 3", value: 78.8 },
  { label: "Round 4", value: 85.2 }
];

export const deviceParticipation: ChartDataPoint[] = [
  { label: "W1", value: 180 },
  { label: "W2", value: 210 },
  { label: "W3", value: 247 }
];

export const trainingThroughput: ChartDataPoint[] = [
  { label: "Mon", value: 12.4 },
  { label: "Tue", value: 14.2 },
  { label: "Wed", value: 15.8 },
  { label: "Thu", value: 13.5 },
  { label: "Fri", value: 18.2 }
];

export const resourceUtilization: ChartDataPoint[] = [
  { label: "CPU", value: 45.0, value2: 40.0 },
  { label: "Memory", value: 60.0, value2: 55.0 },
  { label: "Network", value: 75.0, value2: 65.0 }
];

export const recentActivity: ActivityItem[] = [
  {
    id: "act_1",
    type: "device_joined",
    message: "Node client_1 joined the federation",
    timestamp: new Date(Date.now() - 5000).toISOString(),
  },
  {
    id: "act_2",
    type: "round_completed",
    message: "Round 3 aggregation finished successfully",
    timestamp: new Date(Date.now() - 60000).toISOString(),
  },
  {
    id: "act_3",
    type: "model_updated",
    message: "Global model weights synchronized to v1.2.0",
    timestamp: new Date(Date.now() - 120000).toISOString(),
  },
  {
    id: "act_4",
    type: "security_verified",
    message: "SecAgg protocol successfully completed for all active clients",
    timestamp: new Date(Date.now() - 180000).toISOString(),
  }
];
