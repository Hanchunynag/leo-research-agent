# Multi-Agent Evaluation

The fixed release dataset is
`data/evaluation/multi_agent_cases.jsonl`. It covers research-only,
support-claim, all three writing routes, revise/pass, provider/tool failure,
approval rejection, stale patch, session isolation, and restart recovery.

The deterministic evaluator reports:

- Supervisor Routing Accuracy
- Specialist Task Success Rate
- Tool Validity and Capability Violation Rate
- Human Approval Bypass Rate
- Review Loop and Trajectory Validity
- Session Isolation and Failure Recovery Accuracy
- Average Agent/Tool Calls, P50/P95 latency, token usage, and cost

Hard constraints are zero for Research manuscript writes, Writer direct web
research, Reviewer Safe Apply, approval bypass, invalid tool calls, and
cross-session contamination. The same ground truth can be used by legacy and
CrewAI parity runners.

