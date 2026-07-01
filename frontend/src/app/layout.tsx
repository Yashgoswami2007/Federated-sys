import type { Metadata } from "next";
import { Inter, Fira_Code } from "next/font/google";
import { AppShell } from "@/components/layout/app-shell";
import { WebSocketProvider } from "@/lib/websocket-context";
import "./globals.css";

const geistSans = Inter({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Fira_Code({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "FusionNet — Distributed Federated Learning Network",
  description:
    "Enterprise-grade federated learning infrastructure dashboard for managing edge devices, training jobs, and global models.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} dark h-full antialiased`}
    >
      <body className="min-h-full bg-[#060608] font-sans text-zinc-100">
        <WebSocketProvider>
          <AppShell>{children}</AppShell>
        </WebSocketProvider>
      </body>
    </html>
  );
}
