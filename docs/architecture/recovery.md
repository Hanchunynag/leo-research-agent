# Durable and Recovery Strategy

Project Runtime owns durability; CrewAI owns agent execution.

```mermaid
flowchart LR
  Checkpoint[Run checkpoint projection] --> Restart[Worker restart]
  Restart --> Inspect[Read Run + Job + DraftPatch]
  Inspect --> Resume[Reconstruct bounded Flow input]
  Resume --> Idempotent[Idempotent domain operation]
  Idempotent --> Events[Persist RunEvent before publish]
```

- Research, Writer, and Reviewer are replayable cognition steps. Their
  bounded domain contracts are reconstructed from the Run and domain stores.
- DraftPatch registration is idempotent and guarded by project/base hash.
- Approval and Safe Apply remain exclusively in `PatchApprovalService`.
- A lost Worker heartbeat marks the Job and Run `INTERRUPTED`; retry policy
  moves it to `RETRY_PENDING` or terminal `FAILED`.
- Approval after restart is reconciled from the persisted Patch status. The
  caller's resume value is never interpreted as authorization.
- Worker liveness is projected separately through the durable heartbeat
  registry. The API reports `RUNNING`, `STOPPED`, or derived `STALE` status;
  this registry is observability state and cannot advance a Run by itself.
