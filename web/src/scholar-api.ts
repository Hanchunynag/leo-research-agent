export type ScholarRequest = {
  instruction: string;
  project_id: string;
  task_type: "WRITE_INTRODUCTION" | "SUPPORT_CLAIM" | "WRITE_CONCLUSION" | "WRITE_ABSTRACT";
  session_id?: string;
  thread_id?: string;
};

export type ScholarRunEvent = {
  event_id: string;
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
  const source = new EventSource(`/api/scholar/runs/${encodeURIComponent(runId)}/events?after=${after}`);
  const handleEvent = (event: Event) => {
    try {
      onEvent(JSON.parse((event as MessageEvent).data) as ScholarRunEvent);
    } catch (error) {
      onError(error instanceof Error ? error : new Error("Run Event 格式无效。"));
    }
  };
  source.addEventListener("scholar_run", handleEvent);
  source.addEventListener("end", () => source.close());
  source.onerror = () => {
    if (source.readyState !== EventSource.CLOSED) onError(new Error("Run Event SSE 连接已中断。"));
    source.close();
  };
  return () => source.close();
}
