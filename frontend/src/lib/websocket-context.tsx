"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import { EdgeDevice, RegionData } from "@/types";

interface WebSocketContextType {
  devices: EdgeDevice[];
  regions: RegionData[];
  isConnected: boolean;
  refetchDevices: () => Promise<void>;
  refetchRegions: () => Promise<void>;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

const BACKEND_HTTP_URL = "http://localhost:8000";
const BACKEND_WS_URL = "ws://localhost:8000/ws/all";

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const [devices, setDevices] = useState<EdgeDevice[]>([]);
  const [regions, setRegions] = useState<RegionData[]>([]);
  const [isConnected, setIsConnected] = useState(false);

  const refetchDevices = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND_HTTP_URL}/api/devices`);
      if (res.ok) {
        const data = await res.json();
        setDevices(data);
      }
    } catch (err) {
      console.error("Failed to fetch devices:", err);
    }
  }, []);

  const refetchRegions = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND_HTTP_URL}/api/devices/regions`);
      if (res.ok) {
        const data = await res.json();
        setRegions(data);
      }
    } catch (err) {
      console.error("Failed to fetch regions:", err);
    }
  }, []);

  // Fetch initial data
  useEffect(() => {
    refetchDevices();
    refetchRegions();
  }, [refetchDevices, refetchRegions]);

  // WebSocket connection management with automatic reconnects
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimeout: NodeJS.Timeout | null = null;
    let keepAliveInterval: NodeJS.Timeout | null = null;

    const connect = () => {
      console.log("Connecting to WebSocket:", BACKEND_WS_URL);
      ws = new WebSocket(BACKEND_WS_URL);

      ws.onopen = () => {
        console.log("WebSocket connected");
        setIsConnected(true);
        // Start sending a ping every 20 seconds to keep the connection alive
        keepAliveInterval = setInterval(() => {
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send("ping");
          }
        }, 20000);
      };

      ws.onmessage = (event) => {
        if (event.data === "pong") return;
        
        try {
          const message = JSON.parse(event.data);
          const { event: eventType, data } = message;

          if (eventType === "device.registered") {
            const registeredDevice = data as EdgeDevice;
            setDevices((prev) => {
              const exists = prev.some((d) => d.id === registeredDevice.id);
              if (exists) {
                return prev.map((d) => (d.id === registeredDevice.id ? registeredDevice : d));
              }
              return [...prev, registeredDevice];
            });
            // Refetch regions when a new device registers to update coordinates/counts
            refetchRegions();
          } else if (eventType === "device.heartbeat") {
            const heartbeatData = data as { id: string; status: any; cpuUsage: number; memoryUsage: number };
            setDevices((prev) =>
              prev.map((d) =>
                d.id === heartbeatData.id
                  ? {
                      ...d,
                      status: heartbeatData.status,
                      cpuUsage: heartbeatData.cpuUsage,
                      memoryUsage: heartbeatData.memoryUsage,
                      lastSync: new Date().toISOString(),
                    }
                  : d
              )
            );
          }
        } catch (err) {
          console.error("Error parsing WebSocket message:", err);
        }
      };

      ws.onclose = (e) => {
        console.log("WebSocket disconnected:", e.reason);
        setIsConnected(false);
        if (keepAliveInterval) clearInterval(keepAliveInterval);
        
        // Reconnect after 3 seconds
        reconnectTimeout = setTimeout(() => {
          connect();
        }, 3000);
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        if (ws) ws.close();
      };
    };

    connect();

    return () => {
      if (ws) ws.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (keepAliveInterval) clearInterval(keepAliveInterval);
    };
  }, [refetchRegions]);

  return (
    <WebSocketContext.Provider value={{ devices, regions, isConnected, refetchDevices, refetchRegions }}>
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocket() {
  const context = useContext(WebSocketContext);
  if (context === undefined) {
    throw new Error("useWebSocket must be used within a WebSocketProvider");
  }
  return context;
}
