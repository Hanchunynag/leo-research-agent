import { ChangeEvent, FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { api, watchJob } from "./api";
import {
  AsyncScholarRun,
  ScholarDemoPayload,
  ScholarRequest,
  ScholarRunEvent,
  ScholarRunSnapshot,
  scholarApi,
  subscribeScholarEvents,
} from "./scholar-api";

type TaskType = ScholarRequest["task_type"];
type ViewId = "new" | "history" | "detail" | "manuscript" | "evidence" | "approvals" | "evaluation" | "system";
type BatchImportSummary = {
  total: number;
  completed: number;
  succeeded: number;
  failed: string[];
};

const INTRODUCTION_MINIMUM_UNIQUE_PAPERS = 5;

const TASKS: Array<{ value: TaskType; label: string; description: string }> = [
  { value: "SUPPORT_CLAIM", label: "核验论文论点", description: "判断论点是否被可靠证据支持" },
  { value: "WRITE_INTRODUCTION", label: "撰写论文引言", description: "基于证据生成引言修改补丁" },
  { value: "WRITE_CONCLUSION", label: "撰写论文结论", description: "基于当前稿件和结果生成结论" },
  { value: "WRITE_ABSTRACT", label: "撰写论文摘要", description: "根据论文状态生成结构化摘要" },
];

const NAV_ITEMS: Array<{ id: ViewId; label: string; icon: string }> = [
  { id: "new", label: "新建任务", icon: "＋" },
  { id: "history", label: "运行历史", icon: "◷" },
  { id: "detail", label: "任务详情", icon: "⌁" },
  { id: "manuscript", label: "论文稿件", icon: "▤" },
  { id: "evidence", label: "证据与引用", icon: "◈" },
  { id: "approvals", label: "审批中心", icon: "✓" },
  { id: "evaluation", label: "运行评估", icon: "▥" },
  { id: "system", label: "系统状态", icon: "◉" },
];

const METRIC_LABELS: Record<string, string> = {
  "Task Routing Accuracy": "任务路由准确率",
  "Forbidden Tool Call Count": "违规工具调用次数",
  "Unexpected Research Rate": "意外研究触发率",
  "Required Research Miss Rate": "应研究但未研究比例",
  "Domain Result Validity": "领域结果有效率",
  "Context Isolation Violation Count": "上下文隔离违规次数",
  "Resume Success Rate": "断点恢复成功率",
  "Capability Violation Count": "能力边界违规次数",
};

const NODE_LABELS: Record<string, string> = {
  SUPERVISOR: "主管路由",
  RESEARCH: "证据研究",
  WRITER: "论文写作",
  REVIEWER: "质量审查",
  HUMAN_APPROVAL: "人工审批",
  ROUTING: "任务路由",
  RESEARCHING: "资料研究",
  WRITING: "内容写作",
  REVIEWING: "质量审查",
  REVISION: "内容修订",
  APPROVAL: "人工审批",
};

const ACTIVE_STATUSES = new Set(["PENDING", "QUEUED", "RUNNING", "CANCEL_REQUESTED", "WAITING_USER", "WAITING_HUMAN_APPROVAL"]);

function displayValue(value: unknown, fallback = "—") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function displayList(value: unknown) {
  if (!Array.isArray(value) || value.length === 0) return "—";
  return value.map((item) => displayValue(item)).join("、");
}

function uniqueEvidencePaperCount(items: Array<Record<string, any>>) {
  const identities = new Set<string>();
  items.forEach((item) => {
    const metadata = item.metadata && typeof item.metadata === "object" ? item.metadata : {};
    const identity = item.paper_id || metadata.paper_id || item.canonical_id || item.doi || item.work_id || item.document_id;
    if (identity) identities.add(String(identity));
  });
  return identities.size;
}

function introductionCoverage(snapshot: ScholarRunSnapshot | null) {
  const value = snapshot?.result?.value;
  const metadata = value && typeof value === "object" && value.metadata && typeof value.metadata === "object" ? value.metadata : {};
  const coverage = metadata.citation_coverage;
  return coverage && typeof coverage === "object" ? coverage as Record<string, any> : null;
}

function statusLabel(status: string, jobStatus?: string) {
  if (jobStatus === "CANCEL_REQUESTED") return "正在结束";
  return {
    COMPLETED: "已完成",
    SUCCESS: "编译成功",
    BUILD_TRIGGERED: "等待编译回报",
    UNAVAILABLE: "不可用",
    UNKNOWN: "未知状态",
    SUCCEEDED: "已完成",
    RUNNING: "运行中",
    FAILED: "失败",
    WAITING_USER: "等待人工审批",
    WAITING_HUMAN_APPROVAL: "等待人工审批",
    AWAITING_APPROVAL: "等待审批",
    NEEDS_USER_REVIEW: "待人工复核",
    APPLIED: "已写入",
    REJECTED: "已拒绝",
    BUILD_FAILED: "编译失败",
    INTERRUPTED: "已暂停",
    PENDING: "排队中",
    QUEUED: "排队中",
    CANCELLED: "已结束",
    CANCEL_REQUESTED: "正在结束",
    NOT_STARTED: "尚未启动",
    READY: "已就绪",
  }[status] || status || "未知状态";
}

function taskLabel(value: unknown) {
  const raw = String(value || "");
  return TASKS.find((task) => task.value === raw)?.label || raw.replaceAll("_", " ") || "学术任务";
}

function eventTypeLabel(value: string) {
  return {
    RUN_STARTED: "任务开始",
    RUN_RESUMED: "从检查点恢复",
    SKILL_SELECTED: "选择任务能力",
    CONTEXT_ASSEMBLED: "整理上下文",
    SUBAGENT_STARTED: "代理开始工作",
    SUBAGENT_COMPLETED: "代理完成工作",
    TOOL_STARTED: "工具开始执行",
    TOOL_COMPLETED: "工具执行完成",
    RESEARCH_PROGRESS: "研究阶段进度",
    RESEARCH_COMPLETED: "研究完成",
    EVIDENCE_VERIFIED: "证据验证完成",
    DOMAIN_RESULT: "业务结果",
    DRAFT_CREATED: "草稿已生成",
    REVIEW_STARTED: "开始审查",
    REVIEW_COMPLETED: "审查完成",
    PATCH_CREATED: "修改补丁已生成",
    CHECKPOINT_SAVED: "检查点已保存",
    WAITING_USER: "等待人工审批",
    RUN_INTERRUPTED: "任务已暂停",
    RUN_COMPLETED: "任务完成",
    RUN_FAILED: "任务失败",
    RUN_CANCELLED: "任务已结束",
  }[value] || value;
}

function statusClass(status: string, jobStatus?: string) {
  const value = jobStatus === "CANCEL_REQUESTED" ? "cancelling" : status.toLowerCase().replaceAll("_", "-");
  return `status-pill ${value}`;
}

function nodeLabel(node: string) {
  const normalized = node.toUpperCase().replaceAll("-", "_").replaceAll(" ", "_");
  return NODE_LABELS[normalized] || node.replaceAll("_", " ");
}

function runtimeModeLabel(mode: unknown) {
  const value = String(mode || "");
  return { production: "生产模式", fixture: "演示数据", local: "本地模式", test: "测试模式" }[value] || value || "未知";
}

function formatDate(value: unknown, fallback = "暂无时间") {
  if (!value) return fallback;
  const date = new Date(String(value));
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function formatMetric(name: string, value: unknown) {
  if (typeof value !== "number") return "—";
  if (name.includes("Accuracy") || name.includes("Rate") || name.includes("Validity")) return `${Math.round(value * 100)}%`;
  return Number.isInteger(value) ? String(value) : `${Math.round(value * 100)}%`;
}

function mergeEvents(current: ScholarRunEvent[], incoming: ScholarRunEvent[]) {
  const byId = new Map<string, ScholarRunEvent>();
  [...current, ...incoming].forEach((event) => {
    const fallbackId = `${event.run_id || "run"}:${event.cursor || event.event_id}`;
    byId.set(event.event_id || fallbackId, event);
  });
  return [...byId.values()].sort((left, right) => {
    const cursorDelta = (left.cursor || 0) - (right.cursor || 0);
    return cursorDelta || String(left.event_id).localeCompare(String(right.event_id));
  });
}

function timelineEventKey(event: ScholarRunEvent) {
  const metadata = event.metadata || {};
  const kind = String(metadata.kind || event.type).toLowerCase();
  let name = String(metadata.trace_name || event.summary || event.type).toLowerCase();
  name = name.replace(/(?:started|completed|finished|failed)(?:event)?$/, "");
  name = name.replace(/(?:_|\s|-)(started|completed|finished|failed)$/, "");
  const fieldsByKind: Record<string, string[]> = {
    crew: ["crew_name"],
    task: ["task_id", "task_name"],
    agent: ["agent_id", "agent_role"],
    tool: ["tool_name"],
    llm: ["task_id", "agent_id", "agent_role", "task_name"],
    flow: ["flow_name", "method_name"],
  };
  const fields = fieldsByKind[kind] || ["tool_name", "agent", "task"];
  const identity = fields
    .filter((field) => metadata[field] !== undefined && metadata[field] !== null)
    .map((field) => `${field}=${String(metadata[field])}`)
    .join("|");
  return `${kind}|${name}|${identity}`;
}

function timelineDurations(events: ScholarRunEvent[]) {
  const durations = new Map<string, number>();
  const pending = new Map<string, string[]>();
  const terminalStatuses = new Set(["COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"]);
  const eventKey = (event: ScholarRunEvent) => `${event.event_id}:${event.cursor || ""}`;
  const numericDuration = (event: ScholarRunEvent) => (
    typeof event.duration_ms === "number" && Number.isFinite(event.duration_ms)
      ? event.duration_ms
      : null
  );

  events.forEach((event) => {
    const key = timelineEventKey(event);
    const id = eventKey(event);
    const status = String(event.status || "").toUpperCase();
    const reported = numericDuration(event);
    if (reported !== null) durations.set(id, reported);
    if (status === "RUNNING" || status === "PENDING") {
      pending.set(key, [...(pending.get(key) || []), id]);
      return;
    }
    if (!terminalStatuses.has(status)) return;
    const starts = pending.get(key) || [];
    const startedId = starts.shift();
    if (startedId && reported !== null) {
      durations.set(startedId, reported);
    } else if (startedId) {
      const started = events.find((candidate) => eventKey(candidate) === startedId);
      const startedAt = started?.timestamp ? Date.parse(started.timestamp) : NaN;
      const finishedAt = event.timestamp ? Date.parse(event.timestamp) : NaN;
      if (Number.isFinite(startedAt) && Number.isFinite(finishedAt)) {
        durations.set(startedId, Math.max(0, finishedAt - startedAt));
        durations.set(id, Math.max(0, finishedAt - startedAt));
      }
    }
    if (starts.length) pending.set(key, starts);
    else pending.delete(key);
  });
  return durations;
}

function timelineDurationLabel(event: ScholarRunEvent, duration: number | undefined) {
  if (duration !== undefined) {
    if (duration >= 1000) return `${(duration / 1000).toFixed(duration >= 10000 ? 1 : 2)} s`;
    return `${Math.round(duration)} ms`;
  }
  const status = String(event.status || "").toUpperCase();
  if (status === "RUNNING") return "进行中";
  if (status === "PENDING") return "等待开始";
  return "未计时";
}

function isActiveStatus(status: unknown, jobStatus?: unknown) {
  return ACTIVE_STATUSES.has(String(jobStatus || status || "").toUpperCase());
}

function StatusPill({ status, jobStatus }: { status: string; jobStatus?: string }) {
  return <span className={statusClass(status, jobStatus)}>{statusLabel(status, jobStatus)}</span>;
}

function EmptyState({ title, detail, action }: { title: string; detail: string; action?: React.ReactNode }) {
  return <div className="scholar-empty"><div className="empty-mark">○</div><strong>{title}</strong><p>{detail}</p>{action}</div>;
}

function StatCard({ label, value, note, tone = "blue" }: { label: string; value: string; note?: string; tone?: string }) {
  return <div className={`scholar-stat ${tone}`}><span>{label}</span><strong>{value}</strong>{note && <small>{note}</small>}</div>;
}

function EventTimeline({ events, selectedEvent, onSelect }: { events: ScholarRunEvent[]; selectedEvent: string | null; onSelect: (event: ScholarRunEvent) => void }) {
  if (!events.length) return <EmptyState title="还没有运行事件" detail="提交任务后，主管代理、研究代理、写作代理和审查代理的真实事件会按时间顺序出现在这里。" />;
  const durations = timelineDurations(events);
  return <div className="event-timeline" aria-live="polite">
    {events.map((event) => {
      const selected = selectedEvent === event.event_id;
      return <button type="button" className={`timeline-item ${selected ? "selected" : ""}`} key={`${event.event_id}-${event.cursor}`} onClick={() => onSelect(event)}>
        <span className="timeline-index">{event.cursor || "·"}</span><span className="timeline-line" />
        <span className="timeline-main"><span className="timeline-meta"><b>{nodeLabel(event.node)}</b><time>{event.timestamp ? formatDate(event.timestamp) : `步骤 ${event.cursor || "—"}`}</time></span><strong>{event.summary}</strong><small>{eventTypeLabel(event.type)} · {timelineDurationLabel(event, durations.get(`${event.event_id}:${event.cursor || ""}`))}</small></span>
        <span className={`timeline-dot ${event.status.toLowerCase()}`} />
      </button>;
    })}
  </div>;
}

function FlowMap({ events, status }: { events: ScholarRunEvent[]; status: string }) {
  const stages = [
    { key: "supervisor", label: "主管路由", hint: "拆解任务" },
    { key: "research", label: "证据研究", hint: "检索与验证" },
    { key: "writer", label: "论文写作", hint: "形成草稿" },
    { key: "reviewer", label: "质量审查", hint: "检查引用" },
    { key: "approval", label: "人工审批", hint: "接受或拒绝" },
  ];
  const stageEvents = (key: string) => events.filter((event) => {
    const value = `${event.node} ${event.summary}`.toLowerCase();
    if (key === "supervisor") return value.includes("supervisor") || value.includes("主管") || value.includes("route") || value.includes("worker");
    if (key === "research") return value.includes("research") || value.includes("evidence") || value.includes("研究");
    if (key === "writer") return value.includes("writer") || value.includes("writing") || value.includes("draft") || value.includes("写作");
    if (key === "reviewer") return value.includes("review") || value.includes("审查");
    return value.includes("approval") || value.includes("patch") || value.includes("human") || value.includes("审批");
  });
  return <div className="flow-map" aria-label="学术 Agent 运行流程图">
    {stages.map((stage, index) => {
      const matching = stageEvents(stage.key);
      const last = matching[matching.length - 1];
      const stageStatus = last?.status || (status === "COMPLETED" ? "COMPLETED" : "PENDING");
      return <div className="flow-step-wrap" key={stage.key}><div className={`flow-step ${stageStatus.toLowerCase()}`}><span className="flow-step-number">{index + 1}</span><strong>{stage.label}</strong><small>{last ? statusLabel(stageStatus) : stage.hint}</small></div>{index < stages.length - 1 && <span className="flow-arrow">→</span>}</div>;
    })}
  </div>;
}

function EvidenceCard({ item, active, onToggle }: { item: Record<string, any>; active: boolean; onToggle: () => void }) {
  const metadata = item.metadata && typeof item.metadata === "object" ? item.metadata : {};
  const binding = item.citation_binding && typeof item.citation_binding === "object" ? item.citation_binding : null;
  const sourceType = String(item.source_type || metadata.source_type || "LOCAL_CORPUS").toUpperCase();
  return <div className={`evidence-item ${active ? "selected" : ""}`}>
    <button type="button" className="evidence-item-head" onClick={onToggle}><span className="evidence-id">{displayValue(item.evidence_id, "证据")}</span><span className={sourceType === "WEB_LITERATURE" ? "source-tag web" : "source-tag local"}>{sourceType === "WEB_LITERATURE" ? "网页来源" : "本地论文"}</span><span className="evidence-chevron">{active ? "收起" : "查看"}</span></button>
    <button type="button" className="evidence-item-title" onClick={onToggle}><strong>{displayValue(item.title || metadata.title, "已验证证据")}</strong><small>{displayValue(item.source_locator || item.locator || item.validation_status, "验证通过")}</small></button>
    {active && <div className="evidence-expanded"><div><span>文献标识</span><b>{displayValue(item.canonical_id || item.work_id || item.paper_id)}</b></div><div><span>作者 / 日期</span><b>{displayList(item.authors || metadata.authors)} · {displayValue(item.publication_date || item.year || metadata.year)}</b></div><div><span>证据原文</span><p>{displayValue(item.evidence_span || item.content || item.text || metadata.content)}</p></div><div><span>引用状态</span><b>{binding ? `${displayValue(binding.status)} · ${displayValue(binding.bibkey)}` : displayValue(item.bibkey, "需要人工检查")}</b></div></div>}
  </div>;
}

function DiffPanel({ patch }: { patch: Record<string, any> | null }) {
  if (!patch) return <EmptyState title="当前没有待审批补丁" detail="写作代理生成并通过审查后，修改建议会出现在这里。" />;
  const before = String(patch.original_content || "").split("\n");
  const after = String(patch.proposed_content || "").split("\n");
  const rows = Math.max(before.length, after.length);
  return <div className="diff-panel"><div className="diff-head"><span>原文</span><span>AI 提议内容</span></div><div className="diff-body">{Array.from({ length: rows }, (_, index) => <div className="diff-row" key={index}><code>{index + 1}</code><pre className={before[index] !== after[index] ? "changed" : ""}>{before[index] ?? ""}</pre><pre className={before[index] !== after[index] ? "changed" : ""}>{after[index] ?? ""}</pre></div>)}</div></div>;
}

function escapeHtml(value: string) {
  return value.replace(/[&<>\"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character] || character);
}

function highlightLatexLine(value: string) {
  const tokenPattern = /(%.*$)|(\\[A-Za-z@]+)|([{}[\]])|(\$+)/g;
  let cursor = 0;
  let output = "";
  value.replace(tokenPattern, (token, comment, command, bracket, math, offset: number) => {
    output += escapeHtml(value.slice(cursor, offset));
    const className = comment ? "latex-token-comment" : command ? "latex-token-command" : bracket ? "latex-token-bracket" : "latex-token-math";
    output += `<span class="${className}">${escapeHtml(token)}</span>`;
    cursor = offset + token.length;
    return token;
  });
  output += escapeHtml(value.slice(cursor));
  return output || "&nbsp;";
}

function LatexEditor({ section }: { section: Record<string, any> | null }) {
  if (!section) {
    return <div className="latex-editor-empty"><EmptyState title="请选择一个 Section" detail="左侧文件树中的每个文件都可以在这里查看 LaTeX 源码。" /></div>;
  }
  const content = String(section.content || "");
  const lines = content.split("\n");
  return <div className="latex-editor">
    <div className="editor-toolbar">
      <div className="editor-file"><span className="editor-file-icon">T</span><strong>{displayValue(section.relative_path, "section.tex")}</strong><span className="editor-readonly">只读审阅</span></div>
      <div className="editor-toolbar-meta"><span>UTF-8</span><span>LaTeX</span><span>{lines.length} 行</span></div>
    </div>
    <div className="editor-heading"><div><span className="view-eyebrow">SOURCE VIEW</span><h3>{displayValue(section.name, "LaTeX Section")}</h3></div><span className={section.stale ? "stale-label" : "current-label"}>{section.stale ? "需要复核" : "当前版本"}</span></div>
    <div className="editor-meta">内容 Hash：<code>{displayValue(section.content_hash)}</code> · {displayValue(section.character_count, "0")} 字符 · 只读展示</div>
    <div className="editor-surface" role="region" aria-label="LaTeX 源码查看器">
      <div className="editor-gutter" aria-hidden="true">{lines.map((_, index) => <span key={index}>{index + 1}</span>)}</div>
      <pre className="editor-code">{lines.map((line, index) => <span className="editor-code-line" key={index} dangerouslySetInnerHTML={{ __html: highlightLatexLine(line) }} />)}</pre>
    </div>
    <div className="editor-footer"><span>项目文件是唯一事实来源</span><span>选择左侧文件查看其他 Section</span></div>
  </div>;
}

export default function ScholarConsole() {
  const demoMode = new URLSearchParams(window.location.search).get("demo") === "1";
  const [view, setView] = useState<ViewId>("new");
  const [projectId, setProjectId] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("WRITE_INTRODUCTION");
  const [instruction, setInstruction] = useState("请基于已验证证据，撰写一段关于低轨卫星定位的论文引言。");
  const [snapshot, setSnapshot] = useState<ScholarRunSnapshot | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [events, setEvents] = useState<ScholarRunEvent[]>([]);
  const [project, setProject] = useState<Record<string, any> | null>(null);
  const [manuscript, setManuscript] = useState<Record<string, any> | null>(null);
  const [evidence, setEvidence] = useState<Array<Record<string, any>>>([]);
  const [citationRequirements, setCitationRequirements] = useState<any[]>([]);
  const [evaluation, setEvaluation] = useState<Record<string, any> | null>(null);
  const [runtime, setRuntime] = useState<Record<string, any>>({});
  const [controlRuns, setControlRuns] = useState<AsyncScholarRun[]>([]);
  const [projects, setProjects] = useState<Array<Record<string, any>>>([]);
  const [approvals, setApprovals] = useState<Array<Record<string, any>>>([]);
  const [controlMetrics, setControlMetrics] = useState<Record<string, any>>({});
  const [workers, setWorkers] = useState<Record<string, any>>({});
  const [selectedSection, setSelectedSection] = useState<string | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<string | null>(null);
  const [activeEvidence, setActiveEvidence] = useState<string | null>(null);
  const [selectedApproval, setSelectedApproval] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [resumeValue, setResumeValue] = useState('{"decisions":[{"type":"approve"}]}');
  const batchFileInput = useRef<HTMLInputElement>(null);
  const [batchImporting, setBatchImporting] = useState(false);
  const [batchStopping, setBatchStopping] = useState(false);
  const [batchProgress, setBatchProgress] = useState(0);
  const [batchCurrentFile, setBatchCurrentFile] = useState("");
  const [batchSummary, setBatchSummary] = useState<BatchImportSummary | null>(null);
  const batchJobId = useRef<string | null>(null);
  const batchUploadController = useRef<AbortController | null>(null);
  const batchStopRequested = useRef(false);
  const refreshToken = useRef(0);
  const streamEpoch = useRef(0);
  const [streamVersion, setStreamVersion] = useState(0);

  const refreshControls = useCallback(async () => {
    const [runPayload, projectPayload, approvalPayload, metricPayload, workerPayload, runtimePayload, systemPayload] = await Promise.all([
      scholarApi.runs(), scholarApi.projects(), scholarApi.approvals(), scholarApi.metrics(), scholarApi.workers(), scholarApi.runtime(), scholarApi.systemStatus(),
    ]);
    const availableProjects = projectPayload.projects || [];
    const projectIds = new Set(availableProjects.map((item) => String(item.project_id || "")));
    const scopedRuns = projectIds.size ? (runPayload.runs || []).filter((run) => !run.project_id || projectIds.has(String(run.project_id))) : (runPayload.runs || []);
    setControlRuns(scopedRuns);
    setProjects(availableProjects);
    const nextApprovals = approvalPayload.approvals || [];
    setApprovals(nextApprovals);
    setSelectedApproval((current) => current && nextApprovals.some((item) => String(item.patch_id) === current) ? current : nextApprovals[0] ? String(nextApprovals[0].patch_id) : null);
    setControlMetrics(metricPayload.metrics || {});
    setWorkers(workerPayload || {});
    setRuntime({ ...(runtimePayload || {}), ...(systemPayload || {}) });
    return { runs: scopedRuns, projects: availableProjects };
  }, []);

  const importPapers = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    event.target.value = "";
    if (!files.length || demoMode || busy) return;

    const pdfFiles = files.filter((file) => file.name.toLowerCase().endsWith(".pdf"));
    const failed = files
      .filter((file) => !file.name.toLowerCase().endsWith(".pdf"))
      .map((file) => `${file.name}：只支持 PDF 文件`);
    let succeeded = 0;
    let completed = 0;
    let stopped = false;

    setError(null);
    setNotice(null);
    batchStopRequested.current = false;
    batchJobId.current = null;
    batchUploadController.current = null;
    setBatchStopping(false);
    setBatchProgress(0);
    setBatchCurrentFile("");
    setBatchSummary({ total: files.length, completed: 0, succeeded: 0, failed: [...failed] });
    if (!pdfFiles.length) {
      setError("未找到可导入的 PDF 文件。");
      return;
    }

    setBatchImporting(true);
    setBusy(true);
    try {
      for (const [index, file] of files.entries()) {
        if (batchStopRequested.current) {
          stopped = true;
          break;
        }
        const isPdf = file.name.toLowerCase().endsWith(".pdf");
        if (!isPdf) {
          completed = index + 1;
          setBatchSummary({ total: files.length, completed: index + 1, succeeded, failed: [...failed] });
          continue;
        }

        setBatchCurrentFile(`${index + 1}/${files.length} · ${file.name}`);
        setNotice(`正在导入第 ${index + 1}/${files.length} 篇：${file.name}`);
        setBatchProgress(index / files.length);
        try {
          const controller = new AbortController();
          batchUploadController.current = controller;
          const created = await api.upload(file, controller.signal);
          batchUploadController.current = null;
          batchJobId.current = created.job_id;
          if (batchStopRequested.current) await api.cancelJob(created.job_id);
          const snapshot = await watchJob(created.job_id, (jobEvent) => {
            setBatchProgress((index + jobEvent.progress) / files.length);
            setNotice(`正在导入第 ${index + 1}/${files.length} 篇：${file.name} · ${jobEvent.message}`);
          });
          batchJobId.current = null;
          if (snapshot.status === "cancelled") {
            stopped = true;
            completed = index + 1;
            break;
          }
          succeeded += 1;
        } catch (reason: any) {
          batchUploadController.current = null;
          batchJobId.current = null;
          if (batchStopRequested.current) {
            stopped = true;
            completed = index + 1;
            break;
          }
          failed.push(`${file.name}：${reason.message || String(reason)}`);
        }
        completed = index + 1;
        setBatchSummary({ total: files.length, completed, succeeded, failed: [...failed] });
      }

      await refreshControls();
      setBatchProgress(stopped ? completed / files.length : 1);
      setBatchSummary({ total: files.length, completed: stopped ? completed : files.length, succeeded, failed: [...failed] });
      setNotice(stopped
        ? `批量导入已停止：${succeeded} 个文件成功，后续文件未提交。`
        : `批量导入完成：${succeeded}/${files.length} 个文件成功。${failed.length ? `失败 ${failed.length} 个，请查看左侧明细。` : ""}`);
    } catch (reason: any) {
      setError(reason.message || String(reason));
    } finally {
      batchUploadController.current = null;
      batchJobId.current = null;
      setBatchImporting(false);
      setBatchStopping(false);
      setBusy(false);
    }
  };

  const stopBatchImport = async () => {
    if (!batchImporting || batchStopping) return;
    batchStopRequested.current = true;
    setBatchStopping(true);
    batchUploadController.current?.abort();
    const jobId = batchJobId.current;
    if (!jobId) {
      setNotice("已请求停止，当前文件上传结束后不会再提交后续文献。 ");
      return;
    }
    try {
      const snapshot = await api.cancelJob(jobId);
      setNotice(snapshot.status === "cancelled"
        ? "当前文献解析已取消，后续文献不会继续提交。 "
        : "已发出停止请求，当前文献将在安全检查点结束，后续文献不会继续提交。 ");
    } catch (reason: any) {
      setError(reason.message || String(reason));
    }
  };

  const refreshRun = useCallback(async (runId: string, projectKey: string) => {
    const token = ++refreshToken.current;
    ++streamEpoch.current;
    const [nextSnapshot, nextProject, nextManuscript, nextEvidence, nextEvaluation] = await Promise.all([
      scholarApi.snapshot(runId),
      scholarApi.project(projectKey),
      scholarApi.manuscript(projectKey),
      scholarApi.evidence(projectKey, runId),
      // Evaluation is an optional read model. An unsupported task or a
      // temporary evaluator failure must not hide the actual run, event log,
      // evidence, or manuscript from the user.
      scholarApi.evaluation(runId).catch((reason: any) => ({
        run_id: runId,
        supported: false,
        reason_code: "EVALUATION_UNAVAILABLE",
        reason: reason?.message || "评估服务暂时不可用。",
        metrics: {},
        record: null,
      })),
    ]);
    if (token !== refreshToken.current) return;
    const asyncRun = (nextSnapshot as any).async_run || {};
    setSnapshot({ ...nextSnapshot, run: { ...nextSnapshot.run, ...asyncRun } });
    setProject(nextProject);
    setManuscript(nextManuscript);
    const availableSections = Array.isArray(nextManuscript.sections) ? nextManuscript.sections : [];
    setSelectedSection((current) => current && availableSections.some((section: any) => section.name === current) ? current : availableSections[0]?.name || null);
    setEvidence(Array.isArray(nextEvidence.verified_evidence) ? nextEvidence.verified_evidence : []);
    setCitationRequirements(Array.isArray(nextEvidence.citation_requirements) ? nextEvidence.citation_requirements : []);
    setEvaluation(nextEvaluation);
    setEvents([]);
    setSelectedEvent(null);
    setStreamVersion((value) => value + 1);
  }, []);

  const openRun = useCallback(async (run: AsyncScholarRun | Record<string, any>) => {
    if (run.detail_available === false) {
      setNotice("这条运行记录属于历史遗留数据，详情投影已不可用。系统保留它用于审计，但不会尝试打开。 ");
      return;
    }
    const runId = String(run.run_id || "");
    const projectKey = String(run.project_id || projectId || "");
    if (!runId || !projectKey) return;
    setError(null);
    setNotice(null);
    setActiveRunId(runId);
    setProjectId(projectKey);
    setView("detail");
    try {
      await refreshRun(runId, projectKey);
    } catch (reason: any) {
      setError(reason.message || String(reason));
    }
  }, [projectId, refreshRun]);

  useEffect(() => {
    if (demoMode) {
      scholarApi.demo().then((payload: ScholarDemoPayload) => {
        const demoRun = payload.snapshot.run || {};
        setSnapshot(payload.snapshot);
        setActiveRunId(String(demoRun.run_id || "DEMO_RUN"));
        setProjectId(String(demoRun.project_id || "DEMO_PROJECT"));
        setEvents(payload.events);
        const demoProject = { project_id: demoRun.project_id || "DEMO_PROJECT", sections: payload.sections, project_hash: "demo-hash", version: 3, patches: [] };
        setProject(demoProject);
        setManuscript({ ...demoProject, pdf_available: false, latest_build: null });
        setEvidence(payload.evidence);
        setEvaluation(payload.evaluation);
        setRuntime({ status: "DEMO", mode: "fixture", checkpoint: "演示检查点", persistent: false, orchestration_backend: "crewai" });
        setView("detail");
      }).catch((reason) => setError(reason.message || String(reason)));
      return;
    }
    refreshControls().then(({ runs, projects: availableProjects }) => {
      const projectKey = availableProjects[0]?.project_id ? String(availableProjects[0].project_id) : "";
      if (projectKey) setProjectId((current) => current || projectKey);
      const latest = [...runs].sort((left, right) => String(right.created_at || "").localeCompare(String(left.created_at || "")))[0];
      if (latest && projectKey) void openRun(latest);
    }).catch((reason) => setError(reason.message || String(reason)));
  }, [demoMode, openRun, refreshControls]);

  useEffect(() => {
    if (demoMode || !activeRunId) return;
    let alive = true;
    const epoch = streamEpoch.current;
    const cleanup = subscribeScholarEvents(activeRunId, (event) => {
      if (alive && epoch === streamEpoch.current) setEvents((items) => mergeEvents(items, [event]));
    }, (reason) => {
      if (alive && epoch === streamEpoch.current && !reason.message.includes("重连")) setError(reason.message);
    }, 0);
    return () => { alive = false; cleanup(); };
  }, [activeRunId, demoMode, streamVersion]);

  useEffect(() => {
    if (demoMode || !activeRunId || !snapshot || !isActiveStatus(snapshot.run.status, snapshot.run.job_status)) return;
    let alive = true;
    const timer = window.setInterval(async () => {
      try {
        const nextSnapshot = await scholarApi.snapshot(activeRunId);
        if (!alive) return;
        const asyncRun = (nextSnapshot as any).async_run || {};
        const merged = { ...nextSnapshot, run: { ...nextSnapshot.run, ...asyncRun } };
        setSnapshot(merged);
        setControlRuns((items) => items.map((item) => item.run_id === activeRunId ? { ...item, ...merged.run } : item));
        if (!isActiveStatus(merged.run.status, merged.run.job_status)) {
          await refreshRun(activeRunId, projectId);
          await refreshControls();
        }
      } catch (reason: any) {
        if (alive) setError(reason.message || String(reason));
      }
    }, 1400);
    return () => { alive = false; window.clearInterval(timer); };
  }, [activeRunId, demoMode, projectId, refreshControls, refreshRun, snapshot?.run?.status, snapshot?.run?.job_status]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (demoMode || busy || !instruction.trim() || !projectId.trim()) return;
    setBusy(true); setError(null); setNotice(null); ++streamEpoch.current;
    try {
      const next = await scholarApi.createRun({ instruction, project_id: projectId, task_type: taskType, session_id: sessionId || undefined });
      setSessionId(next.session_id);
      setControlRuns((items) => [next, ...items.filter((item) => item.run_id !== next.run_id)]);
      setActiveRunId(next.run_id); setView("detail");
      await refreshRun(next.run_id, projectId);
      setNotice("任务已提交，页面正在接收 Worker 的实时事件。 ");
    } catch (reason: any) {
      setError(reason.message || String(reason));
    } finally { setBusy(false); }
  };

  const stopRun = async () => {
    if (demoMode || busy || !activeRunId) return;
    setBusy(true); setError(null); setNotice(null);
    try {
      const stopped = await scholarApi.stopRun(activeRunId);
      setSnapshot((current) => current ? { ...current, run: { ...current.run, ...stopped } } : current);
      setNotice(stopped.job_status === "CANCEL_REQUESTED" ? "已发出结束请求，Worker 会在下一个安全检查点停止。" : "任务已结束。 ");
      await refreshControls();
    } catch (reason: any) { setError(reason.message || String(reason)); }
    finally { setBusy(false); }
  };

  const resumeRun = async () => {
    if (demoMode || busy || !snapshot?.run?.thread_id) return;
    setBusy(true); setError(null);
    try {
      const parsed = JSON.parse(resumeValue);
      const next = await scholarApi.resume({ project_id: projectId, thread_id: String(snapshot.run.thread_id), instruction: "continue", session_id: snapshot.run.session_id ? String(snapshot.run.session_id) : undefined, task_type: snapshot.routing.task_type as TaskType | undefined, resume_value: parsed });
      setActiveRunId(next.run_id); setSessionId(next.session_id); setView("detail");
      await refreshRun(next.run_id, projectId);
    } catch (reason: any) { setError(reason.message || String(reason)); }
    finally { setBusy(false); }
  };

  const patch = snapshot?.result?.value?.patch as Record<string, any> | undefined;
  const storedPatchRecord = [...(project?.patches || [])].reverse().find((item: any) => item?.patch?.patch_id);
  const selectedApprovalRecord = approvals.find((item) => String(item.patch_id) === selectedApproval);
  const displayPatch = patch || storedPatchRecord?.patch || selectedApprovalRecord?.patch || null;
  const storedPatchStatus = storedPatchRecord?.status || selectedApprovalRecord?.status;
  const usage = snapshot?.harness?.usage || {};
  const metrics = evaluation?.metrics || {};
  const selectedSectionData = (manuscript?.sections || []).find((item: any) => item.name === selectedSection) || null;
  const currentStatus = String(snapshot?.run?.status || "");
  const activeRun = controlRuns.find((item) => item.run_id === activeRunId) || snapshot?.run;
  const introCoverage = introductionCoverage(snapshot);
  const citedPaperCount = Number(introCoverage?.cited_paper_count ?? (taskType === "WRITE_INTRODUCTION" ? uniqueEvidencePaperCount(evidence) : 0));
  const availablePaperCount = Number(introCoverage?.available_paper_count ?? uniqueEvidencePaperCount(evidence));

  const refreshManuscript = async () => {
    if (!projectId) return;
    try {
      const [nextManuscript, nextProject] = await Promise.all([scholarApi.manuscript(projectId), scholarApi.project(projectId)]);
      setManuscript(nextManuscript); setProject(nextProject);
      setNotice("稿件内容已刷新。 ");
    } catch (reason: any) { setError(reason.message || String(reason)); }
  };

  const chooseProject = async (nextProjectId: string) => {
    setProjectId(nextProjectId);
    setActiveRunId(null);
    setSnapshot(null);
    setManuscript(null);
    setProject(null);
    if (!nextProjectId) return;
    try {
      const [nextManuscript, nextProject] = await Promise.all([scholarApi.manuscript(nextProjectId), scholarApi.project(nextProjectId)]);
      setManuscript(nextManuscript);
      setProject(nextProject);
      const firstSection = Array.isArray(nextManuscript.sections) ? nextManuscript.sections[0] : null;
      setSelectedSection(firstSection?.name || null);
      setNotice("已切换项目，稿件内容已加载。 ");
    } catch (reason: any) {
      setError(reason.message || String(reason));
    }
  };

  const requestManuscriptBuild = async () => {
    if (demoMode || busy || !projectId) return;
    setBusy(true); setError(null); setNotice(null);
    try {
      const nextBuild = await scholarApi.requestBuild(projectId, displayPatch?.patch_id ? String(displayPatch.patch_id) : undefined);
      setManuscript((current) => current ? { ...current, latest_build: nextBuild } : current);
      setNotice(nextBuild.status === "BUILD_TRIGGERED" ? "已请求 LaTeX Workshop 编译，请等待 VS Code 回报结果后刷新稿件。" : "编译请求已提交。");
    } catch (reason: any) {
      setError(reason.message || String(reason));
    } finally {
      setBusy(false);
    }
  };

  const decidePatch = async (decision: "accept" | "reject", selectedPatch = displayPatch) => {
    if (demoMode || busy || !selectedPatch?.patch_id || !selectedPatch?.base_hash) return;
    setBusy(true); setError(null); setNotice(null);
    try {
      const payload = { project_id: projectId, expected_base_hash: String(selectedPatch.base_hash), actor: "human:web-console" };
      const response = decision === "accept" ? await scholarApi.approve(String(selectedPatch.patch_id), payload) : await scholarApi.reject(String(selectedPatch.patch_id), payload);
      setNotice(`${decision === "accept" ? "修改已接受" : "修改已拒绝"}：${response.status || "已记录"}`);
      await refreshControls();
      if (activeRunId) await refreshRun(activeRunId, projectId); else await refreshManuscript();
    } catch (reason: any) { setError(reason.message || String(reason)); }
    finally { setBusy(false); }
  };

  const showView = (next: ViewId) => { setView(next); setError(null); setNotice(null); };
  const selectRun = (run: AsyncScholarRun) => {
    if (run.detail_available === false) {
      setNotice(run.availability_message || "这条历史运行记录的详情数据不可用，无法打开。 ");
      return;
    }
    void openRun(run);
  };

  const initializeManuscript = async () => {
    if (demoMode || busy || !projectId) return;
    setBusy(true); setError(null); setNotice(null);
    try {
      const next = await scholarApi.initializeManuscript(projectId);
      setManuscript(next);
      setProject((current) => current ? { ...current, ...next } : current);
      const firstSection = Array.isArray(next.sections) ? next.sections[0] : null;
      setSelectedSection(firstSection?.name || null);
      setNotice("已创建空白 LaTeX 项目骨架。AI 正文仍需经过 DraftPatch 和人工审批后才会写入。 ");
    } catch (reason: any) { setError(reason.message || String(reason)); }
    finally { setBusy(false); }
  };

  const renderNewTask = () => <section className="view-card new-task-view">
    <div className="view-intro"><div><span className="view-eyebrow">WORKSPACE / NEW TASK</span><h2>让 Agent 帮你处理一项论文工作</h2><p>选择当前 LaTeX 项目，说明你要完成的工作。运行过程、证据、稿件修改和审批都会被记录。</p></div><div className="intro-decoration"><span>01</span><span>证据</span><span>→</span><span>稿件</span></div></div>
    <form className="task-form" onSubmit={submit}><div className="form-grid"><label>论文项目<select value={projectId} onChange={(event) => setProjectId(event.target.value)} disabled={demoMode || busy}><option value="">请选择项目</option>{projects.map((item) => <option key={String(item.project_id)} value={String(item.project_id)}>{String(item.name || item.title || item.project_id)}</option>)}</select><small>{projectId ? `项目 ID：${projectId}` : "选择后才可以提交任务"}</small></label><label>会话 ID（可选）<input value={sessionId} onChange={(event) => setSessionId(event.target.value)} placeholder="留空则自动创建" disabled={demoMode || busy} /><small>同一会话可继续查看相关任务</small></label></div><label>你希望 Agent 做什么<textarea value={instruction} onChange={(event) => setInstruction(event.target.value)} rows={5} disabled={demoMode || busy} placeholder="例如：请比较三篇论文的 Doppler 定位方法，并写出引言中的研究动机。" /><small>任务说明会保存到运行历史，建议写清楚目标、范围和输出形式。</small></label><div className="task-type-grid">{TASKS.map((task) => <button type="button" key={task.value} className={taskType === task.value ? "selected" : ""} onClick={() => setTaskType(task.value)} disabled={demoMode || busy}><span>{task.label}</span><small>{task.description}</small></button>)}</div>{taskType === "WRITE_INTRODUCTION" && <div className="citation-policy-note"><strong>引言引用门槛：至少 {INTRODUCTION_MINIMUM_UNIQUE_PAPERS} 篇不同论文</strong><span>系统会按 paper_id 去重，同一篇论文的多个 Chunk 不会重复计数；文献不足时不会生成可审批的稿件补丁。</span></div>}<div className="form-actions"><span>{demoMode ? "演示模式不能创建真实任务" : runtime.llm_configured ? "本地 LLM 已配置 · CrewAI Worker 可用" : "请先检查模型服务配置"}</span><button className="primary-button" type="submit" disabled={demoMode || busy || !instruction.trim() || !projectId.trim()}>{busy ? "正在提交…" : "开始运行任务 →"}</button></div></form>
  </section>;

  const renderHistory = () => <section className="view-card history-view"><div className="view-header"><div><span className="view-eyebrow">RUNS / HISTORY</span><h2>运行历史</h2><p>每一次任务都保留完整的状态、事件、证据和稿件结果。详情不可恢复的遗留记录会明确标记，且不会误导你打开。</p></div><button className="secondary-button" type="button" onClick={() => void refreshControls()}>↻ 刷新列表</button></div><div className="history-summary"><StatCard label="全部运行" value={String(controlRuns.length)} note="持久化记录" /><StatCard label="已完成" value={String(controlRuns.filter((run) => ["COMPLETED", "SUCCEEDED"].includes(run.status)).length)} note="可查看结果" tone="green" /><StatCard label="进行中" value={String(controlRuns.filter((run) => isActiveStatus(run.status, run.job_status)).length)} note="自动刷新" tone="orange" /><StatCard label="待审批" value={String(approvals.filter((item) => ["AWAITING_APPROVAL", "PENDING", "NEEDS_USER_REVIEW"].includes(String(item.status))).length)} note="需要你的决定" tone="purple" /></div>{controlRuns.length ? <div className="run-table"><div className="run-table-head"><span>任务</span><span>状态</span><span>时间</span><span>运行 ID</span><span /></div>{[...controlRuns].sort((left, right) => String(right.created_at || "").localeCompare(String(left.created_at || ""))).map((run) => { const unavailable = run.detail_available === false; return <button type="button" className={`run-row ${run.run_id === activeRunId ? "selected" : ""} ${unavailable ? "unavailable" : ""}`} key={run.run_id} onClick={() => selectRun(run)} disabled={unavailable} aria-label={unavailable ? "历史数据不可用" : `打开运行 ${run.run_id}`}><span><strong>{displayValue(run.title, taskLabel(run.task_type))}</strong><small>{unavailable ? displayValue(run.availability_message, "历史数据不可用") : `${displayValue(run.project_id)} · ${displayValue(run.session_id)}`}</small></span><StatusPill status={String(run.status)} jobStatus={run.job_status} /><span>{formatDate(run.created_at)}</span><code>{run.run_id}</code><b>{unavailable ? "不可打开" : "查看 →"}</b></button>; })}</div> : <EmptyState title="还没有运行记录" detail="从“新建任务”开始，完成的任务会自动出现在这里。" action={<button className="primary-button" type="button" onClick={() => showView("new")}>创建第一项任务</button>} />}</section>;

  const renderDetail = () => <>{!snapshot ? <section className="view-card"><EmptyState title="尚未选择运行" detail="从左侧历史记录选择一项任务，或创建一项新的学术任务。" action={<button className="primary-button" type="button" onClick={() => showView("new")}>新建任务</button>} /></section> : <><section className="run-hero view-card"><div><span className="view-eyebrow">任务详情 / {taskLabel(snapshot.routing.task_type)}</span><h2>{displayValue(activeRun?.title, taskLabel(snapshot.routing.task_type))}</h2><div className="run-identity"><code>{displayValue(snapshot.run.run_id)}</code><span>创建于 {formatDate(snapshot.run.created_at)}</span><StatusPill status={currentStatus} jobStatus={String(snapshot.run.job_status || "")} /></div></div><div className="run-hero-actions">{isActiveStatus(currentStatus, snapshot.run.job_status) && <button className="danger-button" type="button" onClick={() => void stopRun()} disabled={busy}>{snapshot.run.job_status === "CANCEL_REQUESTED" ? "正在结束…" : "结束运行"}</button>}{currentStatus === "INTERRUPTED" && <button className="primary-button" type="button" onClick={() => void resumeRun()} disabled={busy}>从检查点继续</button>}<button className="secondary-button" type="button" onClick={() => void refreshRun(String(snapshot.run.run_id), projectId)}>↻ 刷新</button></div></section>{snapshot.routing.task_type === "WRITE_INTRODUCTION" && <section className={`citation-coverage-banner ${citedPaperCount >= INTRODUCTION_MINIMUM_UNIQUE_PAPERS ? "sufficient" : "insufficient"}`}><strong>引言引用覆盖：{citedPaperCount} / {INTRODUCTION_MINIMUM_UNIQUE_PAPERS} 篇不同论文</strong><span>{citedPaperCount >= INTRODUCTION_MINIMUM_UNIQUE_PAPERS ? "已达到最低要求" : `还缺 ${Math.max(0, INTRODUCTION_MINIMUM_UNIQUE_PAPERS - citedPaperCount)} 篇；未达到门槛不会生成可审批补丁。`}</span></section>}<section className="flow-card view-card"><div className="card-title"><div><span className="view-eyebrow">LIVE WORKFLOW</span><h3>运行流程图</h3></div><span className="event-count">{events.length} 个事件</span></div><FlowMap events={events} status={currentStatus} /><div className="flow-footer"><span>当前步骤：{taskLabel(snapshot.routing.task_type)}</span><span>上下文令牌：{displayValue(usage.context_tokens)}</span><span>步骤数：{displayValue(usage.steps)}</span><span>结束原因：{displayValue(snapshot.termination_reason, "尚未结束")}</span></div></section><div className="detail-columns"><section className="view-card timeline-card"><div className="card-title"><div><span className="view-eyebrow">EVENT LOG</span><h3>完整运行时间线</h3></div><span className="muted-label">按 cursor 顺序</span></div><EventTimeline events={events} selectedEvent={selectedEvent} onSelect={(event) => setSelectedEvent(event.event_id)} />{selectedEvent && <div className="event-detail"><strong>事件详细信息</strong><pre>{JSON.stringify(events.find((event) => event.event_id === selectedEvent), null, 2)}</pre></div>}</section><section className="view-card detail-side-card"><div className="card-title"><div><span className="view-eyebrow">RESULT SNAPSHOT</span><h3>任务结果</h3></div></div><div className="result-summary"><div><span>任务类型</span><strong>{taskLabel(snapshot.routing.task_type)}</strong></div><div><span>编排框架</span><strong>{displayValue(snapshot.orchestration_backend, "CrewAI")}</strong></div><div><span>已验证论文</span><strong>{snapshot.routing.task_type === "WRITE_INTRODUCTION" ? `${citedPaperCount} / ${INTRODUCTION_MINIMUM_UNIQUE_PAPERS} 篇` : `${availablePaperCount} 篇`}</strong></div><div><span>修改补丁</span><strong>{displayPatch ? "1 个待处理" : "无"}</strong></div></div><button className="wide-link" type="button" onClick={() => showView("evidence")}>查看证据与引用 →</button>{displayPatch && <button className="wide-link accent" type="button" onClick={() => showView("manuscript")}>查看稿件修改 →</button>}{snapshot?.run?.status === "INTERRUPTED" && <div className="resume-box"><strong>任务已暂停</strong><p>可以从持久化检查点恢复运行。</p><textarea value={resumeValue} onChange={(event) => setResumeValue(event.target.value)} rows={3} /><button className="secondary-button" type="button" onClick={() => void resumeRun()} disabled={busy}>继续运行</button></div>}</section></div></>}</>;

  const renderManuscript = () => {
    if (!manuscript) {
      return <section className="manuscript-view view-card"><div className="view-header"><div><span className="view-eyebrow">LATEX / MANUSCRIPT</span><h2>论文稿件</h2><p>在类似 Overleaf 的工作区中查看项目文件、LaTeX 源码、PDF 编译结果和 AI 修改差异。</p></div><button className="secondary-button" type="button" onClick={() => void refreshManuscript()}>↻ 刷新稿件</button></div><EmptyState title="尚未加载论文稿件" detail="选择一个运行或项目后，这里会读取安全范围内的 root.tex 和所有被引用的 section。" /></section>;
    }
    if (manuscript.manuscript_available === false) {
      return <section className="manuscript-view view-card"><div className="view-header"><div><span className="view-eyebrow">LATEX / MANUSCRIPT</span><h2>论文稿件</h2><p>当前项目还没有 root.tex。先创建空白 LaTeX 项目骨架，之后 Agent 才能生成待审批的 DraftPatch。</p></div><button className="secondary-button" type="button" onClick={() => void refreshManuscript()}>↻ 刷新稿件</button></div><EmptyState title="尚未初始化 LaTeX 项目" detail="初始化只会创建 main.tex、sections/*.tex 和 references.bib，不会写入 AI 正文。" action={<button className="primary-button" type="button" onClick={() => void initializeManuscript()} disabled={busy || demoMode}>创建 LaTeX 项目骨架</button>} /></section>;
    }
    const sections = Array.isArray(manuscript.sections) ? manuscript.sections as Array<Record<string, any>> : [];
    const orderedSections = [...sections].sort((left, right) => {
      if (left.name === "root") return -1;
      if (right.name === "root") return 1;
      return String(left.relative_path || left.name || "").localeCompare(String(right.relative_path || right.name || ""));
    });
    const build = manuscript.latest_build && typeof manuscript.latest_build === "object" ? manuscript.latest_build as Record<string, any> : null;
    const buildStatus = String(build?.status || (manuscript.pdf_available ? "SUCCESS" : "UNAVAILABLE")).toUpperCase();
    const buildTone = buildStatus === "SUCCESS" ? "ready" : buildStatus === "FAILED" ? "failed" : "waiting";
    const diagnostics = Array.isArray(build?.diagnostics) ? build.diagnostics as Array<Record<string, any>> : [];
    const errorCount = diagnostics.filter((item) => String(item.severity).toUpperCase() === "ERROR").length;
    const warningCount = diagnostics.filter((item) => String(item.severity).toUpperCase() === "WARNING").length;
    return <section className="manuscript-view view-card">
      <div className="view-header">
        <div><span className="view-eyebrow">LATEX / MANUSCRIPT</span><h2>论文稿件</h2><p>左侧选择文件，中间阅读带行号和语法高亮的 LaTeX 源码，右侧查看 PDF。AI 修改仍必须经过人工审批。</p></div>
        <div className="header-actions"><button className="primary-button" type="button" onClick={() => void requestManuscriptBuild()} disabled={busy || demoMode}>⌘ 请求编译</button><button className="secondary-button" type="button" onClick={() => void refreshManuscript()}>↻ 刷新稿件</button>{manuscript.pdf_available && <a className="secondary-button link-button" href={manuscript.pdf_url} target="_blank" rel="noreferrer">新窗口打开 PDF ↗</a>}</div>
      </div>
      <div className="manuscript-meta">
        <StatCard label="项目版本" value={"v" + displayValue(manuscript.version)} note={"root：" + displayValue(manuscript.root_tex)} />
        <StatCard label="LaTeX 文件" value={String(orderedSections.length)} note={String(manuscript.stale_sections?.length || 0) + " 个需要复核"} tone="orange" />
        <StatCard label="项目 Hash" value={String(manuscript.project_hash || "—").slice(0, 12)} note="当前内容指纹" tone="purple" />
        <div className={"build-status " + buildTone}><span>最近一次编译</span><strong>{buildStatus === "SUCCESS" ? "编译成功" : buildStatus === "FAILED" ? "编译失败" : buildStatus === "BUILD_TRIGGERED" ? "等待回报" : "尚未编译"}</strong><small>{manuscript.pdf_available ? "PDF 可在右侧预览" : build?.message || "等待 VS Code LaTeX Workshop 构建"}</small></div>
      </div>
      <div className="overleaf-workspace">
        <aside className="section-tree overleaf-tree">
          <div className="tree-title">
            <span className="file-tree-caption">PROJECT FILES</span>
            <strong>论文项目</strong>
            <small>{displayValue(manuscript.root_tex, "尚未找到 root.tex")}</small>
          </div>
          <div className="file-tree-folder"><span>⌄</span><strong>LaTeX 源文件</strong><em>{orderedSections.length}</em></div>
          {orderedSections.length ? orderedSections.map((section: any) => <button type="button" key={section.name} className={selectedSection === section.name ? "selected" : ""} onClick={() => setSelectedSection(section.name)} title={section.relative_path}>
            <span className={section.stale ? "tree-dot stale" : "tree-dot"} />
            <span><strong>{section.name === "root" ? displayValue(manuscript.root_tex, "root.tex") : displayValue(section.name)}</strong><small>{displayValue(section.relative_path)}</small></span>
            <em>{section.line_count || 0} 行</em>
          </button>) : <div className="tree-empty">当前项目没有可读取的 LaTeX 文件。</div>}
          <div className="file-tree-legend"><span><i className="tree-dot" />当前版本</span><span><i className="tree-dot stale" />上游变化</span></div>
        </aside>
        <LatexEditor section={selectedSectionData} />
        <aside className="pdf-panel overleaf-preview">
          <div className="pdf-panel-head"><div><span className="view-eyebrow">PDF PREVIEW</span><h3>编译结果</h3></div><StatusPill status={buildStatus} /></div>
          <div className="pdf-preview-toolbar"><span>{displayValue(manuscript.root_tex, "main.tex").replace(/\.tex$/, ".pdf")}</span>{manuscript.pdf_available && <a href={manuscript.pdf_url} target="_blank" rel="noreferrer">弹出预览 ↗</a>}</div>
          {manuscript.pdf_available ? <iframe title="论文 PDF 预览" src={manuscript.pdf_url} /> : <div className="pdf-placeholder"><div>⌁</div><strong>当前项目尚未生成 PDF</strong><p>在 VS Code 中执行 LaTeX Workshop Build，完成后点击“刷新稿件”。</p>{build?.message && <small>{build.message}</small>}</div>}
        </aside>
      </div>
      <section className={"compile-console " + buildTone}>
        <div className="compile-console-head"><div><span className="view-eyebrow">BUILD OUTPUT</span><h3>编译状态与日志</h3></div><div className="compile-state"><i className={"compile-state-dot " + buildTone} />{buildStatus === "SUCCESS" ? "编译成功" : buildStatus === "FAILED" ? "编译失败" : buildStatus === "BUILD_TRIGGERED" ? "等待 VS Code 回报" : "暂无编译记录"}</div></div>
        <div className="compile-console-meta"><span>Build ID：<code>{displayValue(build?.build_id, "—")}</code></span><span>开始：{formatDate(build?.started_at, "—")}</span><span>结束：{formatDate(build?.completed_at, "尚未结束")}</span><span className={errorCount ? "diagnostic-count error" : "diagnostic-count"}>{errorCount} 个错误</span><span className={warningCount ? "diagnostic-count warning" : "diagnostic-count"}>{warningCount} 个警告</span></div>
        {build?.message && <pre className="compile-message">{String(build.message)}</pre>}
        {diagnostics.length ? <div className="diagnostic-list">{diagnostics.map((item, index) => <div className={"diagnostic-item " + String(item.severity || "INFO").toLowerCase()} key={String(item.file || "project") + "-" + String(item.line || 0) + "-" + index}><span>{String(item.severity || "INFO")}</span><code>{displayValue(item.file, "项目")}{item.line ? ":" + item.line : ""}{item.column ? ":" + item.column : ""}</code><p>{displayValue(item.message)}</p></div>)}</div> : <div className="compile-empty">{build ? "本次编译没有返回错误或警告诊断。" : "尚未收到 LaTeX Workshop 的构建回报；PDF 文件和编译日志由项目文件/VS Code 构建链提供。"}</div>}
      </section>
      <section className="patch-review">
        <div className="card-title"><div><span className="view-eyebrow">PATCH REVIEW</span><h3>AI 修改差异</h3></div>{displayPatch && <StatusPill status={String(storedPatchStatus || "WAITING_USER")} />}</div>
        <DiffPanel patch={displayPatch} />
        {displayPatch && <div className="patch-review-actions"><span>补丁：<code>{displayPatch.patch_id}</code> · 目标：{displayPatch.target_section}</span><div><button className="secondary-button" type="button" onClick={() => void decidePatch("reject")} disabled={busy || demoMode}>拒绝修改</button><button className="primary-button" type="button" onClick={() => void decidePatch("accept")} disabled={busy || demoMode}>接受并写入稿件</button></div></div>}
      </section>
    </section>;
  };

  const renderEvidence = () => <section className="evidence-view view-card"><div className="view-header"><div><span className="view-eyebrow">EVIDENCE / CITATIONS</span><h2>证据与引用</h2><p>只有已验证证据会进入回答和论文修改。点击证据可查看原文、来源、定位和引用绑定。</p></div><div className="evidence-total"><strong>{evidence.length}</strong><span>条已验证证据</span></div></div>{snapshot?.routing.task_type === "WRITE_INTRODUCTION" && <div className={`citation-coverage-banner ${citedPaperCount >= INTRODUCTION_MINIMUM_UNIQUE_PAPERS ? "sufficient" : "insufficient"}`}><strong>引言至少需要 {INTRODUCTION_MINIMUM_UNIQUE_PAPERS} 篇不同论文</strong><span>当前可用于引用的论文：{citedPaperCount} / {INTRODUCTION_MINIMUM_UNIQUE_PAPERS}；同一论文的多个 Chunk 只计 1 篇。</span></div>}<div className="evidence-flow"><span>任务问题</span><b>→</b><span>Paper-Level 召回</span><b>→</b><span>Per-Paper Content 证据</span><b>→</b><span>验证 / 引用</span></div>{evidence.length ? <div className="evidence-grid">{evidence.map((item) => <EvidenceCard key={String(item.evidence_id)} item={item} active={activeEvidence === item.evidence_id} onToggle={() => setActiveEvidence(activeEvidence === item.evidence_id ? null : String(item.evidence_id))} />)}</div> : <EmptyState title="当前运行没有已验证证据" detail="研究代理完成检索和验证后，证据会在这里按论文来源展示。" />}<div className="citation-requirements"><div className="card-title"><div><span className="view-eyebrow">CITATION CHECK</span><h3>引用要求</h3></div><span>{citationRequirements.length} 条待处理</span></div>{citationRequirements.length ? citationRequirements.map((item, index) => <div className="requirement-row" key={`${item.evidence_id || item.identity_key || "requirement"}-${index}`}><span>需要检查</span><strong>{displayValue(item.evidence_id || item.identity_key)}</strong><p>{displayValue(item.reason, "尚未解析 BibKey")}</p></div>) : <div className="success-empty">✓ 当前没有待人工检查的引用要求</div>}</div></section>;

  const renderApprovals = () => <section className="approval-view view-card"><div className="view-header"><div><span className="view-eyebrow">HUMAN IN THE LOOP</span><h2>审批中心</h2><p>AI 只能提出修改，不能绕过人工审批直接改写论文。每次决定都会回写运行状态和审计记录。</p></div><div className="approval-total"><strong>{approvals.length}</strong><span>条补丁记录</span></div></div>{approvals.length ? <div className="approval-layout"><div className="approval-list">{approvals.map((item) => <button type="button" className={`approval-row ${selectedApproval === item.patch_id ? "selected" : ""}`} key={item.patch_id} onClick={() => setSelectedApproval(String(item.patch_id))}><span className="approval-icon">{item.status === "APPLIED" ? "✓" : item.status === "REJECTED" ? "×" : "!"}</span><span><strong>{displayValue(item.patch?.target_section, "论文 Section")}</strong><small>{item.patch_id} · {statusLabel(String(item.status))}</small></span><time>{formatDate(item.patch?.created_at || item.created_at)}</time></button>)}</div><div className="approval-detail">{selectedApprovalRecord ? <><div className="card-title"><div><span className="view-eyebrow">SELECTED PATCH</span><h3>{selectedApprovalRecord.patch?.target_section || "论文修改"}</h3></div><StatusPill status={String(selectedApprovalRecord.status)} /></div><p className="approval-explain">补丁 ID：<code>{selectedApprovalRecord.patch_id}</code> · 基线 Hash：<code>{selectedApprovalRecord.patch?.base_hash}</code></p><DiffPanel patch={selectedApprovalRecord.patch || null} />{["AWAITING_APPROVAL", "PENDING", "NEEDS_USER_REVIEW"].includes(String(selectedApprovalRecord.status)) && <div className="patch-review-actions"><span>审批后会触发项目状态刷新。</span><div><button className="secondary-button" type="button" onClick={() => void decidePatch("reject", selectedApprovalRecord.patch)} disabled={busy}>拒绝</button><button className="primary-button" type="button" onClick={() => void decidePatch("accept", selectedApprovalRecord.patch)} disabled={busy}>接受修改</button></div></div>}</> : <EmptyState title="请选择一条补丁" detail="左侧列表显示所有待审批、已接受和已拒绝的修改。" />}</div></div> : <EmptyState title="暂无审批记录" detail="当写作代理生成通过审查的 DraftPatch 后，审批任务会出现在这里。" />}</section>;

  const renderEvaluation = () => <section className="evaluation-view view-card"><div className="view-header"><div><span className="view-eyebrow">EVALUATION / QUALITY</span><h2>运行评估</h2><p>这里展示当前任务真实返回的评估指标和运行消耗，不再使用无来源的占位统计。</p></div><StatusPill status={currentStatus || "NOT_STARTED"} /></div>{snapshot ? evaluation?.supported === false ? <div className="evaluation-unavailable"><div className="evaluation-unavailable-icon">i</div><div><strong>当前任务暂无对应评估</strong><p>{displayValue(evaluation.reason, "该任务类型当前没有可用的质量评估合同。")}</p><small>状态码：{displayValue(evaluation.reason_code, "EVALUATION_UNAVAILABLE")}</small></div></div> : <><div className="evaluation-top"><StatCard label="运行状态" value={statusLabel(currentStatus)} note={displayValue(snapshot.termination_reason, "未结束")} tone="green" /><StatCard label="事件数量" value={String(events.length)} note="完整事件时间线" /><StatCard label="上下文令牌" value={displayValue(usage.context_tokens)} note="Harness 返回" tone="purple" /><StatCard label="证据数量" value={String(evidence.length)} note="已验证证据" tone="orange" /></div><div className="metric-grid-large">{Object.keys(METRIC_LABELS).map((key) => <div key={key}><span>{METRIC_LABELS[key]}</span><strong>{formatMetric(key, metrics[key])}</strong><small>{metrics[key] === undefined ? "当前运行未返回该指标" : "来自评估接口"}</small></div>)}</div><div className="runtime-details"><div><span>运行 ID</span><code>{snapshot.run.run_id}</code></div><div><span>会话 ID</span><code>{snapshot.run.session_id}</code></div><div><span>线程 ID</span><code>{snapshot.run.thread_id}</code></div><div><span>编排后端</span><strong>{snapshot.orchestration_backend || "CrewAI"}</strong></div></div></> : <EmptyState title="还没有评估对象" detail="选择一项运行记录后，页面会加载该运行的真实指标。" />}</section>;

  const renderSystem = () => <section className="system-view view-card"><div className="view-header"><div><span className="view-eyebrow">RUNTIME / OBSERVABILITY</span><h2>系统状态</h2><p>学术 Agent 独立运行在 8001 端口，前端只通过 API 读取状态和提交任务。</p></div><button className="secondary-button" type="button" onClick={() => void refreshControls()}>↻ 刷新状态</button></div><div className="system-grid"><div className="system-card"><span className="view-eyebrow">MODEL SERVICE</span><h3><i className={`health-dot ${runtime.llm_configured ? "ok" : "warn"}`} />{runtime.llm_configured ? "本地 LLM 已配置" : "LLM 配置待检查"}</h3><p>当前学术任务使用用户服务器上的 OpenAI-compatible API。</p><div className="system-kv"><span>编排框架</span><strong>{runtime.orchestration_backend || "CrewAI"}</strong></div><div className="system-kv"><span>运行模式</span><strong>{runtimeModeLabel(runtime.mode)}</strong></div></div><div className="system-card"><span className="view-eyebrow">WORKER</span><h3><i className="health-dot ok" />任务执行器</h3><p>API 负责入队，Worker 负责执行 Agent，状态持久化后由页面实时读取。</p><div className="system-kv"><span>运行中任务</span><strong>{workers.active_jobs?.length || 0}</strong></div><div className="system-kv"><span>队列深度</span><strong>{displayValue(controlMetrics.queue_depth, "0")}</strong></div></div><div className="system-card"><span className="view-eyebrow">RAG ARCHITECTURE</span><h3>两阶段层级检索</h3><p>第一阶段在 Paper-Level 召回论文，第二阶段只在 Top-K 论文的内容 Chunk 中检索。</p><div className="rag-pipeline"><span>Paper-Level BM25 + BGE-M3</span><b>→</b><span>Top-K paper_id</span><b>→</b><span>Per-Paper Content</span></div></div></div><div className="system-runtime-foot"><span>持久化：{runtime.persistent ? "已启用" : demoMode ? "演示模式关闭" : "状态未知"}</span><span>运行历史：{controlRuns.length} 条</span><span>审批记录：{approvals.length} 条</span><span>最后刷新：{formatDate(new Date().toISOString())}</span></div></section>;

  const title = NAV_ITEMS.find((item) => item.id === view)?.label || "学术 Agent";
  return <div className="scholar-workbench">{demoMode && <div className="demo-ribbon">演示数据 · 不连接模型服务，不代表生产环境真实运行</div>}<aside className="scholar-sidebar"><div className="scholar-brand"><div className="brand-symbol">LR</div><div><strong>LEO Research Agent</strong><span>学术研究工作台</span></div></div><button className="sidebar-new-button" type="button" onClick={() => showView("new")}>＋ <span>新建学术任务</span></button><div className="sidebar-import"><div className="sidebar-import-head"><span className="view-eyebrow">PAPER LIBRARY</span><small>PDF</small></div><button className="sidebar-import-button" type="button" onClick={() => batchFileInput.current?.click()} disabled={demoMode || busy}>批量导入文献</button><input ref={batchFileInput} type="file" accept="application/pdf,.pdf" multiple hidden onChange={importPapers} />{batchImporting && <div className="sidebar-import-progress" aria-live="polite"><div><span>{batchCurrentFile || "准备导入…"}</span><b>{Math.round(batchProgress * 100)}%</b></div><i><em style={{ width: `${batchProgress * 100}%` }} /></i><button className="sidebar-import-stop" type="button" onClick={() => void stopBatchImport()} disabled={batchStopping} aria-label="停止批量导入" title="停止批量导入">{batchStopping ? "正在停止…" : "停止导入"}</button></div>}{batchSummary && <div className={`sidebar-import-summary ${batchSummary.failed.length ? "has-failures" : ""}`}><strong>最近批量导入</strong><span>{batchSummary.completed}/{batchSummary.total} 完成 · {batchSummary.succeeded} 成功</span>{batchSummary.failed.length > 0 && <ul>{batchSummary.failed.map((item) => <li key={item}>{item}</li>)}</ul>}</div>}</div><nav className="scholar-sidebar-nav" aria-label="学术 Agent 导航">{NAV_ITEMS.map((item) => <button type="button" key={item.id} className={view === item.id ? "active" : ""} onClick={() => showView(item.id)}><span>{item.icon}</span>{item.label}{item.id === "approvals" && approvals.length > 0 && <em>{approvals.length}</em>}</button>)}</nav><div className="sidebar-project"><span className="view-eyebrow">CURRENT PROJECT</span><select value={projectId} onChange={(event) => void chooseProject(event.target.value)}><option value="">请选择项目</option>{projects.map((item) => <option key={String(item.project_id)} value={String(item.project_id)}>{String(item.name || item.title || item.project_id)}</option>)}</select><small>{projectId || "未选择论文项目"}</small></div><div className="sidebar-history"><div><span className="view-eyebrow">RECENT RUNS</span><button type="button" onClick={() => showView("history")}>全部</button></div>{controlRuns.slice(0, 5).map((run) => <button type="button" className={run.run_id === activeRunId ? "selected" : ""} key={run.run_id} onClick={() => selectRun(run)}><span className="mini-run-dot" data-status={run.status} /><span><strong>{displayValue(run.title, taskLabel(run.task_type))}</strong><small>{formatDate(run.created_at)}</small></span></button>)}{!controlRuns.length && <small className="sidebar-muted">暂无运行记录</small>}</div><div className="sidebar-footer"><span className={`health-dot ${runtime.llm_configured || demoMode ? "ok" : "warn"}`} />{demoMode ? "演示模式" : runtime.llm_configured ? "本地模型服务在线" : "模型状态未知"}<small>Scholar Agent · CrewAI</small></div></aside><main className="scholar-main"><header className="scholar-topbar"><div><span className="topbar-path">SCHOLAR WORKSPACE / {title.toUpperCase()}</span><h1>{title}</h1></div><div className="topbar-run"><span className="run-context">{activeRunId ? `当前运行：${activeRunId}` : "未选择运行"}</span>{snapshot && <StatusPill status={currentStatus} jobStatus={String(snapshot.run.job_status || "")} />}<button className="icon-refresh" type="button" onClick={() => { if (activeRunId && projectId) void refreshRun(activeRunId, projectId); else void refreshControls(); }} aria-label="刷新当前页面">↻</button></div></header>{(error || notice) && <div className={`workbench-notice ${error ? "error" : "success"}`} role={error ? "alert" : "status"}><span>{error ? "!" : "✓"}</span><p>{error || notice}</p><button type="button" onClick={() => { setError(null); setNotice(null); }}>×</button></div>}<div className="scholar-content">{view === "new" && renderNewTask()}{view === "history" && renderHistory()}{view === "detail" && renderDetail()}{view === "manuscript" && renderManuscript()}{view === "evidence" && renderEvidence()}{view === "approvals" && renderApprovals()}{view === "evaluation" && renderEvaluation()}{view === "system" && renderSystem()}</div></main></div>;
}
