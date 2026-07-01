"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import {
  LayoutDashboard,
  Cpu,
  Layers,
  Globe,
  Shield,
  BarChart3,
  Settings,
  Zap,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/devices", label: "Devices", icon: Cpu },
  { href: "/training-jobs", label: "Training Jobs", icon: Layers },
  { href: "/global-models", label: "Global Models", icon: Globe },
  { href: "/privacy-security", label: "Privacy & Security", icon: Shield },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 z-40 hidden h-screen w-[17rem] flex-col border-r border-white/[0.06] bg-[#060608]/90 backdrop-blur-2xl lg:flex">
      <div className="flex h-[4.25rem] items-center gap-3 border-b border-white/[0.06] px-6">
        <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400 to-violet-600 shadow-lg shadow-cyan-500/25">
          <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-white/20 to-transparent" />
          <Zap className="relative h-5 w-5 text-white" />
        </div>
        <div>
          <h1 className="text-base font-bold tracking-tight text-zinc-50">
            FusionNet
          </h1>
          <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-zinc-600">
            AI Infrastructure
          </p>
        </div>
      </div>

      <nav className="flex-1 space-y-0.5 p-4">
        <p className="mb-3 px-3 text-[10px] font-semibold uppercase tracking-[0.2em] text-zinc-600">
          Platform
        </p>
        {navItems.map((item) => {
          const isActive =
            pathname === item.href ||
            (item.href !== "/" && pathname.startsWith(item.href));
          const Icon = item.icon;

          return (
            <Link key={item.href} href={item.href}>
              <motion.div
                whileHover={{ x: 3 }}
                transition={{ type: "spring", stiffness: 400, damping: 25 }}
                className={cn(
                  "relative flex items-center gap-3 rounded-xl px-3.5 py-2.5 text-sm font-medium transition-colors",
                  isActive
                    ? "text-cyan-400"
                    : "text-zinc-500 hover:bg-white/[0.03] hover:text-zinc-300"
                )}
              >
                {isActive && (
                  <motion.div
                    layoutId="sidebar-active"
                    className="absolute inset-0 rounded-xl border border-cyan-500/20 bg-gradient-to-r from-cyan-500/10 to-transparent shadow-[0_0_20px_-5px_rgba(34,211,238,0.2)]"
                    transition={{ type: "spring", bounce: 0.15, duration: 0.5 }}
                  />
                )}
                <Icon className={cn("relative h-4 w-4 shrink-0", isActive && "drop-shadow-[0_0_6px_rgba(34,211,238,0.6)]")} />
                <span className="relative">{item.label}</span>
              </motion.div>
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-white/[0.06] p-4">
        <div className="relative overflow-hidden rounded-2xl border border-emerald-500/15 bg-gradient-to-br from-emerald-500/[0.06] to-cyan-500/[0.04] p-4">
          <div className="absolute -right-6 -top-6 h-16 w-16 rounded-full bg-emerald-500/10 blur-2xl" />
          <div className="relative flex items-center gap-2.5">
            <span className="relative flex h-2.5 w-2.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
              <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-emerald-400 glow-dot-emerald" />
            </span>
            <span className="text-xs font-semibold text-emerald-400">
              Network Online
            </span>
          </div>
          <p className="relative mt-2 text-xs leading-relaxed text-zinc-500">
            247 devices · 5 regions · 99.97% uptime
          </p>
        </div>
      </div>
    </aside>
  );
}
