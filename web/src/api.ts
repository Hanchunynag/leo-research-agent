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
  retrieval_rounds: Array<Record<string, any>>;
};

export type JobSnapshot = {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
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
    request<{ messages: Array<{ role: "user" | "assistant"; text: string; answerable?: boolean }> }>(
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
  upload: (file: File) => {
    const data = new FormData();
    data.append("file", file);
    return request<{ job_id: string }>("/api/papers/upload", {
      method: "POST",
      body: data,
    });
  },
  job: (jobId: string) => request<JobSnapshot>(`/api/jobs/${jobId}`),
};

export function watchJob(
  jobId: string,
  onProgress: (event: { stage: string; message: string; progress: number }) => void,
): Promise<JobSnapshot> {
  return new Promise((resolve, reject) => {
    const source = new EventSource(`/api/jobs/${jobId}/events`);
    source.addEventListener("progress", (event) => {
      onProgress(JSON.parse((event as MessageEvent).data));
    });
    source.addEventListener("done", async () => {
      source.close();
      try {
        const snapshot = await api.job(jobId);
        if (snapshot.status === "failed") {
          reject(new Error(snapshot.error || "任务执行失败。"));
        } else {
          resolve(snapshot);
        }
      } catch (error) {
        reject(error);
      }
    });
    source.onerror = () => {
      source.close();
      reject(new Error("与本地 RAG 服务的事件连接已中断。"));
    };
  });
}
