import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  ScholarRequest,
  ScholarDemoPayload,
  ScholarResult,
  ScholarRunEvent,
  ScholarRunSnapshot,
  scholarApi,
  subscribeScholarEvents,
} from "./scholar-api";

type TaskType = ScholarRequest["task_type"];

const TASKS: Array<{ value: TaskType; label: string; short: string }> = [
  { value: "SUPPORT_CLAIM", label: "Support Claim", short: "Claim" },
  { value: "WRITE_INTRODUCTION", label: "Write Introduction", short: "Introduction" },
  { value: "WRITE_CONCLUSION", label: "Write Conclusion", short: "Conclusion" },
  { value: "WRITE_ABSTRACT", label: "Write Abstract", short: "Abstract" },
];

function statusLabel(status: string) {
  return { COMPLETED: "Completed", RUNNING: "Running", FAILED: "Failed", WAITING_USER: "Waiting", INTERRUPTED: "Interrupted" }[status] || status;
}

function eventNodes(events: ScholarRunEvent[]) {
  const nodes = new Map<string, ScholarRunEvent>();
  events.forEach((event) => nodes.set(event.node, event));
  return [...nodes.values()];
}

function formatMetric(value: unknown) {
  if (typeof value !== "number") return "—";
  return Number.isInteger(value) ? String(value) : `${Math.round(value * 100)}%`;
}

export default function ScholarConsole() {
  const demoMode = new URLSearchParams(window.location.search).get("demo") === "1";
  const [projectId, setProjectId] = useState("PROJECT_ID");
  const [sessionId, setSessionId] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("WRITE_INTRODUCTION");
  const [instruction, setInstruction] = useState("Write an evidence-grounded introduction about LEO positioning.");
  const [result, setResult] = useState<ScholarResult | null>(null);
  const [snapshot, setSnapshot] = useState<ScholarRunSnapshot | null>(null);
  const [events, setEvents] = useState<ScholarRunEvent[]>([]);
  const [project, setProject] = useState<Record<string, any> | null>(null);
  const [evidence, setEvidence] = useState<any[]>([]);
  const [citationRequirements, setCitationRequirements] = useState<any[]>([]);
  const [evaluation, setEvaluation] = useState<Record<string, any> | null>(null);
  const [runtime, setRuntime] = useState<Record<string, any>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeEvidence, setActiveEvidence] = useState<string | null>(null);
  const [approvalMessage, setApprovalMessage] = useState<string | null>(null);
  const [resumeValue, setResumeValue] = useState('{"decisions":[{"type":"approve"}]}');

  useEffect(() => {
    if (demoMode) {
      scholarApi.demo().then((payload: ScholarDemoPayload) => {
        setSnapshot(payload.snapshot);
        setEvents(payload.events);
        setProject({ sections: payload.sections, project_hash: "demo-hash", version: 3 });
        setEvidence(payload.evidence);
        setCitationRequirements([]);
        setEvaluation(payload.evaluation);
        setRuntime({ status: "DEMO", mode: "fixture", checkpoint: "FixtureTrace", persistent: false });
      }).catch((reason) => setError(reason.message));
      return;
    }
    scholarApi.runtime().then(setRuntime).catch((reason) => setError(reason.message));
  }, [demoMode]);

  useEffect(() => {
    if (demoMode || !snapshot?.run?.run_id) return;
    const runId = String(snapshot.run.run_id);
    let alive = true;
    const cleanup = subscribeScholarEvents(runId, (event) => {
      if (alive) setEvents((items) => [...items, event]);
    }, (reason) => alive && setError(reason.message), events.length);
    return () => {
      alive = false;
      cleanup();
    };
  }, [demoMode, snapshot?.run?.run_id]);

  const refreshRun = async (runId: string, projectKey: string) => {
    const [nextSnapshot, nextProject, nextEvidence, nextEvaluation] = await Promise.all([
      scholarApi.snapshot(runId),
      scholarApi.project(projectKey),
      scholarApi.evidence(projectKey, runId),
      scholarApi.evaluation(runId),
    ]);
    setSnapshot(nextSnapshot);
    setProject(nextProject);
    setEvidence(Array.isArray(nextEvidence.verified_evidence) ? nextEvidence.verified_evidence : []);
    setCitationRequirements(Array.isArray(nextEvidence.citation_requirements) ? nextEvidence.citation_requirements : []);
    setEvaluation(nextEvaluation);
    setEvents([]);
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (demoMode || busy || !instruction.trim() || !projectId.trim()) return;
    setBusy(true);
    setError(null);
    setEvents([]);
    try {
      const next = await scholarApi.request({ instruction, project_id: projectId, task_type: taskType, session_id: sessionId || undefined });
      setResult(next);
      setSessionId(next.session_id);
      await refreshRun(next.run_id, projectId);
    } catch (reason: any) {
      setError(reason.message || String(reason));
    } finally {
      setBusy(false);
    }
  };

  const nodes = useMemo(() => eventNodes(events), [events]);
  const patch = snapshot?.result?.value?.patch;
  const usage = snapshot?.harness?.usage || {};
  const metrics = evaluation?.metrics || {};

  const decidePatch = async (decision: "accept" | "reject") => {
    if (demoMode || !patch?.patch_id || !patch?.base_hash) return;
    setApprovalMessage(null);
    try {
      const payload = { project_id: projectId, expected_base_hash: String(patch.base_hash), actor: "human:web-console" };
      const response = decision === "accept"
        ? await scholarApi.approve(String(patch.patch_id), payload)
        : await scholarApi.reject(String(patch.patch_id), payload);
      setApprovalMessage(`${decision === "accept" ? "Accepted" : "Rejected"}: ${response.status || "recorded"}`);
      if (decision === "accept") await refreshRun(String(snapshot?.run?.run_id || ""), projectId);
    } catch (reason: any) {
      setApprovalMessage(reason.message || String(reason));
    }
  };

  const resumeRun = async () => {
    if (demoMode || !snapshot?.run?.thread_id) return;
    setApprovalMessage(null);
    try {
      const parsed = JSON.parse(resumeValue);
      const next = await scholarApi.resume({
        project_id: projectId,
        thread_id: String(snapshot.run.thread_id),
        instruction: "continue",
        session_id: snapshot.run.session_id ? String(snapshot.run.session_id) : undefined,
        task_type: snapshot.routing.task_type as TaskType | undefined,
        resume_value: parsed,
      });
      setResult(next);
      await refreshRun(next.run_id, projectId);
    } catch (reason: any) {
      setApprovalMessage(reason.message || String(reason));
    }
  };

  return (
    <div className="scholar-console">
      {demoMode && <div className="demo-ribbon">DEMO / FIXTURE DATA · 不连接 Provider，不代表 Production E2E</div>}
      <header className="scholar-console-header">
        <div>
          <span className="console-kicker">SCHOLARHARNESS / CONTROL CONSOLE</span>
          <h1>Evidence-first writing, with an audit trail.</h1>
          <p>Web Console 是 Agent / Research observability surface；VS Code + LaTeX Workshop 负责真实编辑、Diff、Approval、Build 和 PDF Preview。</p>
        </div>
        <div className="console-runtime"><span className={`console-dot ${demoMode || runtime.llm_configured ? "ok" : "warn"}`} />{demoMode ? "Fixture mode" : runtime.llm_configured ? "Production configured" : "Provider status unknown"}</div>
      </header>

      <section className="console-request panel-surface">
        <div className="section-heading"><span className="console-kicker">REQUEST</span><span className="mono">{snapshot?.run?.run_id || "NO RUN"}</span></div>
        <form onSubmit={submit}>
            <div className="request-controls">
            <label>Project ID<input value={projectId} onChange={(event) => setProjectId(event.target.value)} disabled={demoMode || busy} /></label>
            <label>Session ID<input value={sessionId} onChange={(event) => setSessionId(event.target.value)} placeholder="create on request" disabled={demoMode || busy} /></label>
            <label>Skill<select value={taskType} onChange={(event) => setTaskType(event.target.value as TaskType)} disabled={demoMode || busy}>{TASKS.map((task) => <option key={task.value} value={task.value}>{task.label}</option>)}</select></label>
          </div>
          <label>Instruction<textarea value={instruction} onChange={(event) => setInstruction(event.target.value)} rows={2} disabled={demoMode || busy} /></label>
          <div className="request-footer"><span>{snapshot ? `${snapshot.routing.selected_skill || "skill"} · ${statusLabel(String(snapshot.run.status || ""))} · ${snapshot.termination_reason || "—"}` : "Submit a Scholar Request to inspect a real run."}</span><button type="submit" disabled={demoMode || busy}>{busy ? "Running…" : "Send Scholar Request"}</button></div>
        </form>
        {snapshot?.run?.status === "INTERRUPTED" && <div className="resume-callout"><label>Checkpoint resume payload <span>(framework resume only, not Patch Approval)</span><textarea rows={2} value={resumeValue} onChange={(event) => setResumeValue(event.target.value)} disabled={busy} /></label><button type="button" onClick={() => void resumeRun()} disabled={busy}>Resume from checkpoint</button></div>}
        {error && <div className="console-error" role="alert">{error}</div>}
      </section>

      <main className="console-grid">
        <section className="panel-surface workflow-panel">
          <div className="section-heading"><span className="console-kicker">LIVE WORKFLOW</span><span className="mono">{events.length} events</span></div>
          <div className="workflow-lane" aria-live="polite">
            {nodes.length ? nodes.map((node) => <div className={`workflow-node ${node.status.toLowerCase()}`} key={`${node.node}-${node.event_id}`}><div className="node-marker">{node.status === "FAILED" ? "!" : node.status === "WAITING_USER" ? "…" : "✓"}</div><div><strong>{node.node}</strong><span>{statusLabel(node.status)}</span><small>{node.summary}{node.duration_ms ? ` · ${node.duration_ms} ms` : ""}</small></div></div>) : <div className="empty-console">Run Events 会在提交请求后显示。</div>}
          </div>
          <div className="workflow-footer"><span>Context {usage.context_tokens || "—"} tokens</span><span>Steps {usage.steps || "—"}</span><span>Termination {snapshot?.termination_reason || "—"}</span></div>
        </section>

        <section className="panel-surface evidence-console-panel">
          <div className="section-heading"><span className="console-kicker">EVIDENCE / CITATION</span><span className="mono">{evidence.length} verified</span></div>
          {evidence.length ? evidence.map((item) => <button className={`console-evidence ${activeEvidence === item.evidence_id ? "selected" : ""}`} key={item.evidence_id} onClick={() => setActiveEvidence(activeEvidence === item.evidence_id ? null : item.evidence_id)}><div><strong>{item.evidence_id}</strong><span className={item.source_type === "WEB_LITERATURE" ? "web-tag" : "local-tag"}>{item.source_type === "WEB_LITERATURE" ? "WEB" : "LOCAL"}</span></div><b>{item.title || item.metadata?.title || "Verified Evidence"}</b><small>{item.locator || item.source_locator || item.validation_status || "VERIFIED"}</small>{activeEvidence === item.evidence_id && <em>Identity: {item.canonical_id || item.work_id || item.paper_id || "—"} · Date: {item.publication_date || "—"} · Provider: {item.provider || "local corpus"} · Validation: {item.validation_status || "verified"} · Citation: {item.citation || item.bibkey || "Needs user review"}</em>}</button>) : <div className="empty-console">Only Verified Evidence appears here. Discovery Candidates stay hidden.</div>}
          <div className="citation-chain">Claim → Verified Evidence → CitationIdentity → BibKey · Requirements: {citationRequirements.length ? `${citationRequirements.length} needs user review` : "0"}</div>
        </section>

        <section className="panel-surface manuscript-console-panel">
          <div className="section-heading"><span className="console-kicker">MANUSCRIPT STATE</span><span className="mono">v{project?.version || "—"}</span></div>
          <div className="section-list">{(project?.sections || []).map((section: any) => <div key={section.name} className="section-row"><span className={section.stale ? "stale-dot" : "current-dot"} /><strong>{section.name}</strong><code>{section.relative_path}</code><span className={section.stale ? "stale-label" : "current-label"}>{section.stale ? "STALE" : "CURRENT"}</span><small>v{section.version}</small></div>)}</div>
          {patch && <div className="patch-callout"><span className="console-kicker">DRAFTPATCH</span><strong>{patch.patch_id}</strong><span>Human Approval required · VS Code opens the actual diff</span><div className="patch-actions"><button type="button" onClick={() => void decidePatch("reject")} disabled={demoMode}>Reject</button><button type="button" onClick={() => void decidePatch("accept")} disabled={demoMode}>Accept in Web</button></div>{approvalMessage && <small role="status">{approvalMessage}</small>}</div>}
        </section>

        <section className="panel-surface evaluation-console-panel">
          <div className="section-heading"><span className="console-kicker">EVALUATION / RUNTIME</span><span className="mono">{demoMode ? "FIXTURE" : snapshot ? "LIVE" : "—"}</span></div>
          <div className="metric-grid">{["Task Routing Accuracy", "Forbidden Tool Call Count", "Unexpected Research Rate", "Required Research Miss Rate", "Domain Result Validity", "Context Isolation Violation Count", "Resume Success Rate", "Capability Violation Count"].map((key) => <div key={key}><span>{key}</span><strong>{formatMetric(metrics[key])}</strong></div>)}</div>
          <div className="runtime-footnote">Run {snapshot?.run?.run_id || "—"} · Session {snapshot?.run?.session_id || "—"} · Thread {snapshot?.run?.thread_id || "—"}<br />Runtime {runtime.mode || "—"} · Checkpoint {runtime.checkpoint || "—"} · {runtime.persistent ? "persistent" : "not persistent / fixture"}</div>
        </section>
      </main>
    </div>
  );
}
