export type PaperRecord = {
  paper_id: string;
  work_id: string | null;
  document_id: string;
  title: string;
  authors: string[];
  year: number | null;
  page_count: number;
  quality_issue_count: number;
};

export type SessionRecord = {
  session_id: string;
  title: string;
  active_topic_id: string | null;
  updated_at: string;
};

export type Evidence = {
  source_id?: string;
  evidence_id?: string;
  chunk_id: string;
  work_id: string;
  document_id: string;
  title: string;
  section_path: string[];
  page_start: number;
  page_end: number;
  content: string;
  origin?: string;
};

export type AgenticResult = {
  answerable: boolean;
  answer: string;
  refusal_reason: string | null;
  outcome?: {
    code: "answered" | "insufficient_evidence" | "generation_failed" | "validation_failed" | "budget_exhausted" | string;
    stage: string;
    message: string | null;
    retryable: boolean;
    details?: Record<string, any>;
  };
  claims: Array<{
    claim_id: string;
    text: string;
    source_ids: string[];
    category?: string;
  }>;
  context?: { evidence: Evidence[]; token_count: number };
  session: {
    session_id: string;
    topic_id: string;
    relation: string;
    standalone_query: string;
  };
  coverage: {
    overall_sufficient: boolean;
    coverage: Array<{
      subquestion_id: string;
      status: string;
      supporting_evidence_ids: string[];
      missing_information: string;
    }>;
  };
  diagnostics: Record<string, any>;
  workflow_details?: Record<string, any>;
  retrieval_rounds: Array<Record<string, any>>;
};

export type JobSnapshot = {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  events: Array<{
    sequence: number;
    stage: string;
    message: string;
    progress: number;
  }>;
  result: Record<string, any> | null;
  error: string | null;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `请求失败 (${response.status})`);
  }
  return payload as T;
}

export const api = {
  status: () => request<Record<string, any>>("/api/system/status"),
  papers: () => request<{ records: PaperRecord[]; status: Record<string, any> }>("/api/papers"),
  sessions: () => request<{ sessions: SessionRecord[] }>("/api/sessions"),
  evidence: (sessionId: string) =>
    request<{ evidence: Evidence[] }>(`/api/sessions/${encodeURIComponent(sessionId)}/evidence`),
  transcript: (sessionId: string) =>
    request<{ messages: Array<{ role: "user" | "assistant"; text: string; answerable?: boolean; outcome?: AgenticResult["outcome"] }> }>(
      `/api/sessions/${encodeURIComponent(sessionId)}/transcript`,
    ),
  answer: (query: string, sessionId: string | null, forceNewTopic = false) =>
    request<{ job_id: string }>("/api/answers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        session_id: sessionId,
        force_new_topic: forceNewTopic,
        include_context: true,
      }),
    }),
  resume: (threadId: string, userInput: string) =>
    request<{ job_id: string }>("/api/answers/resume", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: threadId, user_input: userInput }),
    }),
  upload: (file: File, signal?: AbortSignal) => {
    const data = new FormData();
    data.append("file", file);
    return request<{ job_id: string }>("/api/papers/upload", {
      method: "POST",
      body: data,
      signal,
    });
  },
  job: (jobId: string) => request<JobSnapshot>(`/api/jobs/${jobId}`),
  cancelJob: (jobId: string) => request<JobSnapshot>(`/api/jobs/${jobId}/cancel`, { method: "POST" }),
};

export function watchJob(
  jobId: string,
  onProgress: (event: { stage: string; message: string; progress: number }) => void,
): Promise<JobSnapshot> {
  return new Promise((resolve, reject) => {
    const terminalStatuses = new Set<JobSnapshot["status"]>(["succeeded", "failed", "cancelled"]);
    let source: EventSource | null = null;
    let pollTimer: ReturnType<typeof setTimeout> | undefined;
    let settled = false;
    let lastSequence = 0;
    let polling = false;
    let pollFailures = 0;

    const cleanup = () => {
      source?.close();
      source = null;
      if (pollTimer !== undefined) clearTimeout(pollTimer);
      pollTimer = undefined;
    };

    const finish = (snapshot: JobSnapshot) => {
      if (settled) return;
      settled = true;
      cleanup();
      if (snapshot.status === "failed") {
        reject(new Error(snapshot.error || "任务执行失败。"));
      } else {
        resolve(snapshot);
      }
    };

    const applySnapshot = (snapshot: JobSnapshot) => {
      snapshot.events.forEach((event) => {
        if (event.sequence <= lastSequence) return;
        lastSequence = event.sequence;
        onProgress(event);
      });
      if (terminalStatuses.has(snapshot.status)) finish(snapshot);
    };

    const poll = async () => {
      if (settled) return;
      try {
        const snapshot = await api.job(jobId);
        pollFailures = 0;
        applySnapshot(snapshot);
        if (!settled) pollTimer = setTimeout(() => void poll(), 1000);
      } catch (error) {
        pollFailures += 1;
        if (pollFailures >= 5) {
          settled = true;
          cleanup();
          reject(error);
          return;
        }
        pollTimer = setTimeout(() => void poll(), Math.min(1000 * pollFailures, 5000));
      }
    };

    const startPolling = () => {
      if (settled || polling) return;
      polling = true;
      void poll();
    };

    source = new EventSource(`/api/jobs/${jobId}/events`);
    source.addEventListener("progress", (event) => {
      if (settled) return;
      try {
        const payload = JSON.parse((event as MessageEvent).data) as {
          sequence: number;
          stage: string;
          message: string;
          progress: number;
        };
        if (payload.sequence <= lastSequence) return;
        lastSequence = payload.sequence;
        onProgress(payload);
      } catch (error) {
        settled = true;
        cleanup();
        reject(error);
      }
    });
    source.addEventListener("done", () => {
      void api.job(jobId).then((snapshot) => {
        if (terminalStatuses.has(snapshot.status)) finish(snapshot);
        else {
          applySnapshot(snapshot);
          startPolling();
        }
      }).catch(() => startPolling());
    });
    source.onerror = () => {
      // Reverse proxies and long OCR stages can close an idle SSE stream.
      // The durable Job API contains the same event log, so continue from
      // the last sequence instead of falsely reporting an import failure.
      cleanup();
      startPolling();
    };
  });
}
