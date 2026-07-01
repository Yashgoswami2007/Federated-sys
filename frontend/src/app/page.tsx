"use client";

import { HeroSection } from "@/components/dashboard/hero-section";
import { KpiCards } from "@/components/dashboard/kpi-cards";
import { NetworkMap } from "@/components/dashboard/network-map";
import { DeviceTable } from "@/components/dashboard/device-table";
import { TrainingJobsPanel } from "@/components/dashboard/training-jobs-panel";
import { GlobalModelSection } from "@/components/dashboard/global-model-section";
import { PrivacySecurityPanel } from "@/components/dashboard/privacy-security-panel";
import { AnalyticsSection } from "@/components/dashboard/analytics-section";
import { ActivityFeed } from "@/components/dashboard/activity-feed";
import { DashboardSkeleton } from "@/components/dashboard/dashboard-skeleton";

export default function DashboardPage() {
  return (
    <DashboardSkeleton>
      <div className="space-y-10">
        <HeroSection />
        <KpiCards />

        <div className="grid gap-8 lg:grid-cols-5">
          <div className="lg:col-span-3">
            <NetworkMap />
          </div>
          <div className="lg:col-span-2">
            <TrainingJobsPanel />
          </div>
        </div>

        <DeviceTable />

        <div className="grid gap-8 lg:grid-cols-5">
          <div className="lg:col-span-3">
            <GlobalModelSection />
          </div>
          <div className="lg:col-span-2">
            <PrivacySecurityPanel />
          </div>
        </div>

        <AnalyticsSection />
        <ActivityFeed />
      </div>
    </DashboardSkeleton>
  );
}
