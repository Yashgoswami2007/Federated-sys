"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Zap } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Sidebar } from "./sidebar";
import { Navbar } from "./navbar";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  Cpu,
  Layers,
  Globe,
  Shield,
  BarChart3,
  Settings,
} from "lucide-react";

const mobileNavItems = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/devices", label: "Devices", icon: Cpu },
  { href: "/training-jobs", label: "Training Jobs", icon: Layers },
  { href: "/global-models", label: "Global Models", icon: Globe },
  { href: "/privacy-security", label: "Privacy & Security", icon: Shield },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/settings", label: "Settings", icon: Settings },
];

interface AppShellProps {
  children: React.ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const pathname = usePathname();

  return (
    <div className="relative min-h-screen bg-[#060608]">
      {/* Ambient mesh */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -left-32 -top-32 h-[28rem] w-[28rem] rounded-full bg-cyan-500/[0.07] blur-[140px]" />
        <div className="absolute -right-32 top-1/4 h-[28rem] w-[28rem] rounded-full bg-violet-500/[0.06] blur-[140px]" />
        <div className="absolute bottom-0 left-1/2 h-64 w-[40rem] -translate-x-1/2 rounded-full bg-blue-500/[0.04] blur-[120px]" />
      </div>

      <Sidebar />

      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 bg-black/70 backdrop-blur-md lg:hidden"
              onClick={() => setMobileOpen(false)}
            />
            <motion.aside
              initial={{ x: -300 }}
              animate={{ x: 0 }}
              exit={{ x: -300 }}
              transition={{ type: "spring", damping: 28, stiffness: 220 }}
              className="fixed left-0 top-0 z-50 flex h-full w-72 flex-col border-r border-white/[0.06] bg-[#060608] lg:hidden"
            >
              <div className="flex h-[4.25rem] items-center justify-between border-b border-white/[0.06] px-5">
                <div className="flex items-center gap-2.5">
                  <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400 to-violet-600">
                    <Zap className="h-4 w-4 text-white" />
                  </div>
                  <span className="font-bold text-zinc-50">FusionNet</span>
                </div>
                <button
                  onClick={() => setMobileOpen(false)}
                  className="rounded-xl p-2 text-zinc-500 hover:bg-white/5"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
              <nav className="flex-1 space-y-0.5 p-4">
                {mobileNavItems.map((item) => {
                  const isActive =
                    pathname === item.href ||
                    (item.href !== "/" && pathname.startsWith(item.href));
                  const Icon = item.icon;
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      onClick={() => setMobileOpen(false)}
                      className={cn(
                        "flex items-center gap-3 rounded-xl px-3.5 py-2.5 text-sm font-medium",
                        isActive
                          ? "border border-cyan-500/20 bg-cyan-500/10 text-cyan-400"
                          : "text-zinc-500 hover:bg-white/[0.03]"
                      )}
                    >
                      <Icon className="h-4 w-4" />
                      {item.label}
                    </Link>
                  );
                })}
              </nav>
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      <div className="relative lg:pl-[17rem]">
        <Navbar onMenuClick={() => setMobileOpen(true)} />
        <main className="mx-auto max-w-[1600px] px-5 py-8 lg:px-8 lg:py-10">
          {children}
        </main>
      </div>
    </div>
  );
}
