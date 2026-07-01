export type DeviceStatus = "online" | "training" | "offline" | "syncing";

export interface EdgeDevice {
  id: string;
  name: string;
  hardwareType: string;
  status: DeviceStatus;
  cpuUsage: number;
  memoryUsage: number;
  lastSync: string;
  contributionScore: number;
  region: string;
}

export interface TrainingJob {
  id: string;
  round: number;
  totalRounds: number;
  progress: number;
  estimatedCompletion: string;
  participatingDevices: number;
  modelVersion: string;
  status: "running" | "queued" | "completed";
}

export interface GlobalModel {
  name: string;
  version: string;
  accuracy: number;
  lastUpdated: string;
}

export interface ActivityItem {
  id: string;
  type: "device_joined" | "round_completed" | "model_updated" | "security_verified";
  message: string;
  timestamp: string;
}

export interface RegionData {
  name: string;
  deviceCount: number;
  x: number;
  y: number;
}

export interface ChartDataPoint {
  label: string;
  value: number;
  value2?: number;
}

export interface KpiMetric {
  label: string;
  value: string;
  change: string;
  trend: "up" | "down" | "neutral";
  icon: string;
}
