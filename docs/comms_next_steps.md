# FusionNet Comms Next Steps

This document tracks what should happen after the local comms MVP.

## Current Status

The first comms milestone is complete:

```text
simulated clients
  -> LocalCommunicationBackend
  -> FedAvg
  -> HttpEventSink
  -> FastAPI backend route contract
```

The flow was verified with the backend running in in-memory mode. The backend
received round status, events, metrics, and dashboard KPI data.

## What This Proves

- clients can submit updates
- the coordinator can aggregate updates with FedAvg
- comms can report round lifecycle events to backend routes
- backend can expose the reported data for frontend/dashboard consumption
- the local demo still works when backend reporting is disabled

## What This Does Not Prove Yet

- real model training
- real VM-to-VM communication
- persistent PostgreSQL storage
- HF Hub weight transport in this new comms abstraction
- RCCL or AMD-specific transport

## Recommended Next Order

1. **frontend integration layer**
   - create or expand the centralized frontend API client
   - consume `/api/rounds/jobs`, `/api/events/activity`,
     `/api/metrics/loss-curve`, and `/api/dashboard/kpi`
   - replace mock data gradually

2. **multi-process comms simulation**
   - run coordinator and clients as separate Python processes
   - keep the same `ClientUpdate` and `GlobalUpdate` contract
   - use this as the stepping stone toward separate VMs

3. **real ML adapter bridge**
   - ML side exposes real adapter updates from LoRA/AFLoRA training
   - comms replaces fake tensors with real `A` matrix updates
   - local training updates `A`, `B`, and `Lambda`, but only `A` is sent to the coordinator
   - `B` and `Lambda` remain private on the client as `local_B.pt` and `local_lambda.pt`
   - coordinator aggregation logic should stay mostly unchanged

4. **persistent backend mode**
   - PostgreSQL setup
   - migrations or reliable table creation
   - verify the same event sink against the real database-backed routes

5. **future transports**
   - HF Hub transport for real `.pt` weight exchange
   - HTTP transport if backend should receive update metadata or small weights
   - RCCL only after the MVP round flow is stable

## Ownership Boundary

Comms owns:

```text
round lifecycle, update contracts, aggregation handoff, event emission
```

Backend owns:

```text
persistence, REST routes, WebSocket broadcasts, auth, database setup
```

Frontend owns:

```text
API consumption, dashboard state, loading/error UI, visualization
```

ML owns:

```text
local training, real adapter generation, privacy mechanics
```

## Next Decision Needed

The team should decide whether the next implementation task is:

```text
A. frontend-to-backend API integration
```

or:

```text
B. multi-process comms simulation
```

Both are valid, but they should not be mixed in the same task.
