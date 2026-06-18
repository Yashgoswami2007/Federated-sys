export const globalModel = {
  name: "FusionNet-Llama3-8B",
  version: "v3.2.1",
  accuracy: 94.7,
};

export const kpiMetrics = [
  { label: "Active Devices", value: "247", icon: "cpu", trend: "up", change: "+12 this week" },
  { label: "Rounds Complete", value: "1,842", icon: "layers", trend: "up", change: "+54 last week" },
  { label: "Model Accuracy", value: "94.7%", icon: "target", trend: "up", change: "+0.3% last round" },
  { label: "Privacy Budget", value: "68%", icon: "shield", trend: "neutral", change: "32% remaining" },
];

export const privacyMetrics = {
  differentialPrivacy: { epsilon: 0.1, delta: "1e-5" },
  epsilonBudget: 68,
  secureAggregation: { protocol: "SecAgg+" },
  securityScore: 97,
};

export const trainingJobs = [
  { id: '1', round: 1843, totalRounds: 2000, status: 'running', progress: 72, participatingDevices: 198, modelVersion: 'v3.2.1' },
  { id: '2', round: 1844, totalRounds: 2000, status: 'queued', progress: 0, participatingDevices: 0, modelVersion: 'v3.2.2' },
  { id: '3', round: 1845, totalRounds: 2000, status: 'queued', progress: 0, participatingDevices: 0, modelVersion: 'v3.2.2' },
];

export const recentActivity = [
  { id: '1', type: 'round_completed', message: 'Training round 1842 completed successfully', timestamp: new Date(Date.now() - 1000 * 60 * 5), status: 'success' },
  { id: '2', type: 'device_joined', message: 'New edge device connected: Node-247', timestamp: new Date(Date.now() - 1000 * 60 * 15), status: 'info' },
  { id: '3', type: 'security_verified', message: 'Privacy budget updated for region Asia-Pacific', timestamp: new Date(Date.now() - 1000 * 60 * 30), status: 'warning' },
  { id: '4', type: 'model_updated', message: 'Global model accuracy reached 94.7%', timestamp: new Date(Date.now() - 1000 * 60 * 60), status: 'success' },
];

export const edgeDevices = [
  { id: 'node-001', name: 'Node-001', region: 'Asia-Pacific', status: 'online', accuracy: 94.2, rounds: 1842, lastSeen: '2 min ago' },
  { id: 'node-002', name: 'Node-002', region: 'Europe', status: 'online', accuracy: 91.8, rounds: 1839, lastSeen: '5 min ago' },
  { id: 'node-003', name: 'Node-003', region: 'Americas', status: 'training', accuracy: 93.1, rounds: 1841, lastSeen: '1 min ago' },
  { id: 'node-004', name: 'Node-004', region: 'Africa', status: 'offline', accuracy: 88.5, rounds: 1820, lastSeen: '1 hr ago' },
  { id: 'node-005', name: 'Node-005', region: 'Middle East', status: 'online', accuracy: 92.3, rounds: 1838, lastSeen: '3 min ago' },
];
export const accuracyTrend = [
  { label: 'Round 1', value: 78.2 },
  { label: 'Round 2', value: 81.5 },
  { label: 'Round 3', value: 84.1 },
  { label: 'Round 4', value: 87.3 },
  { label: 'Round 5', value: 89.8 },
  { label: 'Round 6', value: 91.2 },
  { label: 'Round 7', value: 93.0 },
  { label: 'Round 8', value: 94.7 },
];

export const lossCurve = [
  { label: 'Round 1', value: 0.085 },
  { label: 'Round 2', value: 0.072 },
  { label: 'Round 3', value: 0.061 },
  { label: 'Round 4', value: 0.050 },
  { label: 'Round 5', value: 0.041 },
  { label: 'Round 6', value: 0.033 },
  { label: 'Round 7', value: 0.026 },
  { label: 'Round 8', value: 0.019 },
];

export const analyticsAccuracy = [
  { label: 'Round 1', value: 78.2 },
  { label: 'Round 2', value: 81.5 },
  { label: 'Round 3', value: 84.1 },
  { label: 'Round 4', value: 87.3 },
  { label: 'Round 5', value: 89.8 },
  { label: 'Round 6', value: 91.2 },
  { label: 'Round 7', value: 93.0 },
  { label: 'Round 8', value: 94.7 },
];

export const deviceParticipation = [
  { label: 'Asia-Pacific', value: 89, value2: 72 },
  { label: 'Europe', value: 76, value2: 68 },
  { label: 'Americas', value: 92, value2: 85 },
  { label: 'Africa', value: 61, value2: 55 },
  { label: 'Middle East', value: 74, value2: 69 },
];

export const trainingThroughput = [
  { label: 'Mon', value: 120 },
  { label: 'Tue', value: 145 },
  { label: 'Wed', value: 132 },
  { label: 'Thu', value: 167 },
  { label: 'Fri', value: 189 },
  { label: 'Sat', value: 201 },
  { label: 'Sun', value: 178 },
];

export const resourceUtilization = [
  { label: 'CPU', value: 67, value2: 55 },
  { label: 'Memory', value: 54, value2: 48 },
  { label: 'Network', value: 78, value2: 65 },
  { label: 'Storage', value: 43, value2: 38 },
];
export const regionData = [
  { id: 'asia', name: 'Asia-Pacific', lat: 35, lng: 105, x: 75, y: 35, devices: 89, status: 'online' },
  { id: 'europe', name: 'Europe', lat: 51, lng: 10, x: 48, y: 25, devices: 76, status: 'online' },
  { id: 'americas', name: 'Americas', lat: 40, lng: -95, x: 20, y: 30, devices: 92, status: 'online' },
  { id: 'africa', name: 'Africa', lat: 0, lng: 20, x: 48, y: 55, devices: 61, status: 'online' },
  { id: 'mideast', name: 'Middle East', lat: 25, lng: 45, x: 60, y: 40, devices: 74, status: 'online' },
];