export type ScholarRequest = {
  instruction: string;
  project_id: string;
  task_type: "WRITE_INTRODUCTION" | "SUPPORT_CLAIM" | "WRITE_CONCLUSION" | "WRITE_ABSTRACT";
  session_id?: string;
  thread_id?: string;
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
  harness: Record<string, any>;
  termination_reason: string | null;
};

export type ScholarDemoPayload = {
  demo: true;
  snapshot: ScholarRunSnapshot;
  events: ScholarRunEvent[];
  evidence: Array<Record<string, any>>;
  sections: Array<Record<string, any>>;
  evaluation: Record<string, any>;
};

export type ScholarResult = {
  status: string;
  task_type: string | null;
  skill_name: string | null;
  run_id: string;
  session_id: string;
  thread_id: string;
  trace_id: string;
  result_type: string | null;
  value: Record<string, any> | null;
  error_codes: string[];
  metadata: Record<string, any>;
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
  request: (payload: ScholarRequest) =>
    request<ScholarResult>("/api/scholar/requests", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  resume: (payload: { project_id: string; thread_id: string; instruction: string; session_id?: string; task_type?: ScholarRequest["task_type"]; resume_value: unknown }) =>
    request<ScholarResult>("/api/scholar/requests/resume", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  snapshot: (runId: string) => request<ScholarRunSnapshot>(`/api/scholar/runs/${encodeURIComponent(runId)}`),
  project: (projectId: string) => request<Record<string, any>>(`/api/scholar/projects/${encodeURIComponent(projectId)}/state`),
  evidence: (projectId: string, runId?: string) => {
    const query = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
    return request<Record<string, any>>(`/api/scholar/projects/${encodeURIComponent(projectId)}/evidence${query}`);
  },
  evaluation: (runId: string) => request<Record<string, any>>(`/api/scholar/runs/${encodeURIComponent(runId)}/evaluation`),
  demo: () => request<ScholarDemoPayload>('/api/scholar/demo'),
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
