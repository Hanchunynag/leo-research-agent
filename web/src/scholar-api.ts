export type ScholarRequest = {
  instruction: string;
  project_id: string;
  task_type: "RESEARCH" | "WRITE_INTRODUCTION" | "SUPPORT_CLAIM" | "WRITE_CONCLUSION" | "WRITE_ABSTRACT" | "REVIEW";
  session_id?: string;
  thread_id?: string;
};

export type AsyncScholarRunRequest = ScholarRequest & {
  metadata?: Record<string, unknown>;
  idempotency_key?: string;
};

export type AsyncScholarRun = {
  run_id: string;
  session_id: string;
  project_id: string;
  thread_id: string;
  trace_id: string;
  status: string;
  job_id?: string;
  job_status?: string;
  event_cursor?: number;
  error?: string | null;
  created_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  title?: string | null;
  task_type?: string | null;
  detail_available?: boolean;
  availability_message?: string | null;
};

export type ScholarRunEvent = {
  event_id: string;
  cursor?: number;
  run_id?: string;
  session_id?: string;
  timestamp?: string;
  type: string;
  node: string;
  status: string;
  summary: string;
  duration_ms?: number;
  metadata?: Record<string, unknown>;
};

export type ScholarRunSnapshot = {
  run: Record<string, any>;
  session: Record<string, any>;
  routing: Record<string, any>;
  result: Record<string, any>;
  orchestration: Record<string, any>;
  termination_reason: string | null;
  async_run?: Record<string, any>;
  orchestration_backend?: string;
};

export type ScholarManuscript = {
  project_id: string;
  root_tex: string;
  project_hash: string;
  version: number;
  stale_sections: string[];
  sections: Array<Record<string, any>>;
  latest_build?: Record<string, any> | null;
  pdf_available: boolean;
  pdf_path?: string | null;
  pdf_url?: string | null;
  manuscript_available?: boolean;
  message?: string;
  facts?: Array<Record<string, any>>;
  contributions?: Array<Record<string, any>>;
  patches?: Array<Record<string, any>>;
};

export type ScholarDemoPayload = {
  demo: true;
  snapshot: ScholarRunSnapshot;
  events: ScholarRunEvent[];
  evidence: Array<Record<string, any>>;
  sections: Array<Record<string, any>>;
  evaluation: Record<string, any>;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail;
    const message = typeof detail === "string" ? detail : detail?.message || `请求失败 (${response.status})`;
    throw new Error(message);
  }
  return payload as T;
}

export const scholarApi = {
  runtime: () => request<Record<string, any>>("/api/scholar/runtime/status"),
  systemStatus: () => request<Record<string, any>>("/api/system/status"),
  snapshot: (runId: string) => request<ScholarRunSnapshot>(`/api/scholar/runs/${encodeURIComponent(runId)}`),
  project: (projectId: string) => request<Record<string, any>>(`/api/scholar/projects/${encodeURIComponent(projectId)}/state`),
  manuscript: (projectId: string) => request<ScholarManuscript>(`/api/scholar/projects/${encodeURIComponent(projectId)}/manuscript`),
  initializeManuscript: (projectId: string) => request<ScholarManuscript>(`/api/scholar/projects/${encodeURIComponent(projectId)}/manuscript/initialize`, { method: "POST" }),
  requestBuild: (projectId: string, patchId?: string) => {
    const query = patchId ? `?patch_id=${encodeURIComponent(patchId)}` : "";
    return request<Record<string, any>>(`/api/scholar/projects/${encodeURIComponent(projectId)}/build${query}`, { method: "POST" });
  },
  evidence: (projectId: string, runId?: string) => {
    const query = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
    return request<Record<string, any>>(`/api/scholar/projects/${encodeURIComponent(projectId)}/evidence${query}`);
  },
  evaluation: (runId: string) => request<Record<string, any>>(`/api/scholar/runs/${encodeURIComponent(runId)}/evaluation`),
  demo: () => request<ScholarDemoPayload>('/api/scholar/demo'),
  createRun: (payload: AsyncScholarRunRequest) =>
    request<AsyncScholarRun>('/api/scholar/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  runs: () => request<{ runs: AsyncScholarRun[] }>('/api/scholar/runs'),
  projects: () => request<{ projects: Array<Record<string, any>> }>('/api/scholar/projects'),
  approvals: () => request<{ approvals: Array<Record<string, any>> }>('/api/scholar/approvals'),
  workers: () => request<Record<string, any>>('/api/scholar/workers/status'),
  metrics: () => request<Record<string, any>>('/api/metrics'),
  cancelRun: (runId: string) => request<AsyncScholarRun>(`/api/scholar/runs/${encodeURIComponent(runId)}/cancel`, { method: 'POST' }),
  stopRun: (runId: string) => request<AsyncScholarRun>(`/api/scholar/runs/${encodeURIComponent(runId)}/stop`, { method: 'POST' }),
  resumeRun: (runId: string, resumeValue?: unknown) => request<AsyncScholarRun>(`/api/scholar/runs/${encodeURIComponent(runId)}/resume`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resume_value: resumeValue }),
  }),
  patch: (patchId: string) => request<Record<string, any>>(`/api/scholar/patches/${encodeURIComponent(patchId)}`),
  approve: (patchId: string, payload: { project_id: string; expected_base_hash: string; actor: string }) =>
    request<Record<string, any>>(`/api/scholar/patches/${encodeURIComponent(patchId)}/accept`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  reject: (patchId: string, payload: { project_id: string; expected_base_hash: string; actor: string }) =>
    request<Record<string, any>>(`/api/scholar/patches/${encodeURIComponent(patchId)}/reject`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
};

export function subscribeScholarEvents(
  runId: string,
  onEvent: (event: ScholarRunEvent) => void,
  onError: (error: Error) => void,
  after = 0,
): () => void {
  let source: EventSource | null = null;
  let retryTimer: ReturnType<typeof setTimeout> | undefined;
  let stopped = false;
  let lastCursor = Math.max(0, after);
  let retryDelay = 500;

  const connect = () => {
    if (stopped) return;
    const nextSource = new EventSource(`/api/scholar/runs/${encodeURIComponent(runId)}/events?after=${lastCursor}`);
    source = nextSource;
    const handleEvent = (event: Event) => {
      try {
        const message = event as MessageEvent;
        const payload = JSON.parse(message.data) as ScholarRunEvent;
        const cursor = Number(payload.cursor || message.lastEventId || 0);
        if (Number.isFinite(cursor) && cursor > lastCursor) lastCursor = cursor;
        retryDelay = 500;
        onEvent(payload);
      } catch (error) {
        onError(error instanceof Error ? error : new Error("Run Event 格式无效。"));
      }
    };
    nextSource.addEventListener("scholar_run", handleEvent);
    nextSource.addEventListener("end", () => {
      nextSource.close();
      if (source === nextSource) source = null;
    });
    nextSource.onerror = () => {
      if (source !== nextSource) return;
      nextSource.close();
      source = null;
      if (stopped) return;
      if (retryTimer !== undefined) return;
      onError(new Error("Run Event SSE 连接已中断，正在从最后 cursor 重连。"));
      retryTimer = setTimeout(() => {
        retryTimer = undefined;
        connect();
      }, retryDelay);
      retryDelay = Math.min(retryDelay * 2, 5_000);
    };
  };

  connect();
  return () => {
    stopped = true;
    if (retryTimer !== undefined) clearTimeout(retryTimer);
    source?.close();
    source = null;
  };
}
