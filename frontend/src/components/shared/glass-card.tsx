"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  delay?: number;
  glow?: "cyan" | "violet" | "emerald" | "none";
  noHover?: boolean;
}

const glowMap = {
  cyan: "hover:shadow-[0_0_60px_-12px_rgba(34,211,238,0.15)]",
  violet: "hover:shadow-[0_0_60px_-12px_rgba(167,139,250,0.15)]",
  emerald: "hover:shadow-[0_0_60px_-12px_rgba(52,211,153,0.15)]",
  none: "",
};

export function GlassCard({
  children,
  className,
  delay = 0,
  glow = "cyan",
  noHover = false,
}: GlassCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay, ease: [0.22, 1, 0.36, 1] }}
      whileHover={noHover ? undefined : { y: -3 }}
      className={cn(
        "glass-premium relative overflow-hidden rounded-2xl transition-all duration-500",
        !noHover && glowMap[glow],
        className
      )}
    >
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
      <div className="pointer-events-none absolute -right-20 -top-20 h-40 w-40 rounded-full bg-cyan-500/5 blur-3xl" />
      {children}
    </motion.div>
  );
}
