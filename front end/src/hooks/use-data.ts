import { useState, useEffect, useCallback } from "react";
import { TrainingJob, GlobalModel, ActivityItem, ChartDataPoint, KpiMetric } from "@/types";

const BACKEND_HTTP_URL = "http://localhost:8000";

interface UseDataResult<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

function useApi<T>(endpoint: string, initialData: T | null = null): UseDataResult<T> {
  const [data, setData] = useState<T | null>(initialData);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(`${BACKEND_HTTP_URL}${endpoint}`);
      if (!res.ok) {
        throw new Error(`Error: ${res.status} ${res.statusText}`);
      }
      const json = await res.json();
      setData(json);
    } catch (err) {
      console.error(`Failed to fetch ${endpoint}:`, err);
      setError(err instanceof Error ? err : new Error("Unknown error"));
    } finally {
      setLoading(false);
    }
  }, [endpoint]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
}

export function useKpiMetrics() {
  return useApi<KpiMetric[]>("/api/dashboard/kpi", []);
}

export function useGlobalModel() {
  return useApi<GlobalModel>("/api/models/global", null);
}

export function useTrainingJobs() {
  return useApi<TrainingJob[]>("/api/rounds/jobs", []);
}

// We'll define a type for privacy status here since it might not be in "@/types"
export interface PrivacyStatus {
  differentialPrivacy: { epsilon: number; delta: number };
  epsilonBudget: number;
  secureAggregation: { protocol: string };
  securityScore: number;
}

export function usePrivacyStatus() {
  return useApi<PrivacyStatus>("/api/privacy/status", null);
}

export function useActivityFeed() {
  return useApi<ActivityItem[]>("/api/events/activity", []);
}

export function useAccuracyTrend() {
  return useApi<ChartDataPoint[]>("/api/metrics/accuracy-trend", []);
}

export function useLossCurve() {
  return useApi<ChartDataPoint[]>("/api/metrics/loss-curve", []);
}

export function useAnalyticsAccuracy() {
  return useApi<ChartDataPoint[]>("/api/metrics/analytics-accuracy", []);
}

export function useDeviceParticipation() {
  return useApi<ChartDataPoint[]>("/api/metrics/device-participation", []);
}

export function useTrainingThroughput() {
  return useApi<ChartDataPoint[]>("/api/metrics/training-throughput", []);
}

export function useResourceUtilization() {
  return useApi<ChartDataPoint[]>("/api/metrics/resource-utilization", []);
}
