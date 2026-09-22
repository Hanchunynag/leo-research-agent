export type JobSnapshot = {
  job_id: string;
  kind: "parse";
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  events: Array<{ sequence: number; stage: string; message: string; progress: number }>;
  result: Record<string, any> | null;
  error: string | null;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `请求失败 (${response.status})`);
  return payload as T;
}

export const api = {
  upload: (file: File, signal?: AbortSignal) => {
    const data = new FormData();
    data.append("file", file);
    return request<{ job_id: string }>("/api/papers/upload", { method: "POST", body: data, signal });
  },
  job: (jobId: string) => request<JobSnapshot>(`/api/jobs/${jobId}`),
  cancelJob: (jobId: string) => request<JobSnapshot>(`/api/jobs/${jobId}/cancel`, { method: "POST" }),
};

export function watchJob(jobId: string, onProgress: (event: { stage: string; message: string; progress: number }) => void): Promise<JobSnapshot> {
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
      if (snapshot.status === "failed") reject(new Error(snapshot.error || "任务执行失败。"));
      else resolve(snapshot);
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
      const payload = JSON.parse((event as MessageEvent).data) as JobSnapshot["events"][number];
      if (payload.sequence <= lastSequence) return;
      lastSequence = payload.sequence;
      onProgress(payload);
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
      cleanup();
      startPolling();
    };
  });
}
