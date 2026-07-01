"use client";

import { Bell, Menu, Search } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { StatusIndicator } from "@/components/shared/status-indicator";

interface NavbarProps {
  onMenuClick?: () => void;
}

export function Navbar({ onMenuClick }: NavbarProps) {
  return (
    <header className="sticky top-0 z-30 flex h-[4.25rem] items-center justify-between gap-4 border-b border-white/[0.06] bg-[#060608]/70 px-5 backdrop-blur-2xl lg:px-8">
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          size="icon"
          className="lg:hidden"
          onClick={onMenuClick}
        >
          <Menu className="h-5 w-5" />
        </Button>
        <div className="relative hidden sm:block">
          <Search className="absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-600" />
          <Input
            placeholder="Search devices, jobs, models..."
            className="h-10 w-72 border-white/[0.06] bg-white/[0.03] pl-10 lg:w-96"
          />
        </div>
      </div>

      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" className="relative h-10 w-10">
          <Bell className="h-4 w-4" />
          <span className="absolute right-2.5 top-2.5 h-2 w-2 rounded-full bg-cyan-400 glow-dot-cyan" />
        </Button>

        <div className="hidden h-8 w-px bg-white/[0.06] sm:block" />

        <div className="hidden items-center gap-3 sm:flex">
          <div className="text-right">
            <p className="text-sm font-semibold text-zinc-200">Dr. Sarah Chen</p>
            <p className="text-[11px] text-zinc-600">Network Admin</p>
          </div>
          <Avatar className="h-10 w-10 border-cyan-500/20">
            <AvatarFallback className="text-sm">SC</AvatarFallback>
          </Avatar>
        </div>

        <StatusIndicator status="online" variant="pill" size="sm" />
      </div>
    </header>
  );
}
