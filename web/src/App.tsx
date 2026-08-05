import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";
import { AgenticResult, api, Evidence, PaperRecord, SessionRecord, watchJob } from "./api";

type ChatMessage = {
  role: "user" | "assistant";
  text: string;
  answerable?: boolean;
  outcome?: AgenticResult["outcome"];
};

const OUTCOME_LABELS: Record<string, string> = {
  insufficient_evidence: "证据不足",
  generation_failed: "生成格式失败",
  validation_failed: "引用验证失败",
  budget_exhausted: "运行预算耗尽",
  legacy_refusal: "历史拒答",
};

const SUGGESTIONS = [
  "哪些观测量用于估计低轨卫星星历和时钟误差？",
  "这些论文的研究路线如何演进？",
  "比较不同方法的数据集和评价指标。",
];

function formatAuthors(authors: string[]) {
  if (!authors.length) return "作者未识别";
  return authors.length > 2 ? `${authors.slice(0, 2).join(", ")} 等` : authors.join(", ");
}

function App() {
  const [papers, setPapers] = useState<PaperRecord[]>([]);
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState({ stage: "ready", message: "就绪", progress: 0 });
  const [result, setResult] = useState<AgenticResult | null>(null);
  const [registryEvidence, setRegistryEvidence] = useState<Evidence[]>([]);
  const [activePanel, setActivePanel] = useState<"evidence" | "diagnostics">("evidence");
  const [selectedSource, setSelectedSource] = useState<string | null>(null);
  const [system, setSystem] = useState<Record<string, any>>({});
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = async () => {
    const [paperPayload, sessionPayload, systemPayload] = await Promise.all([
      api.papers(),
      api.sessions(),
      api.status(),
    ]);
    setPapers(paperPayload.records);
    setSessions(sessionPayload.sessions);
    setSystem(systemPayload);
  };

  useEffect(() => {
    refresh().catch((reason) => setError(String(reason.message || reason)));
  }, []);

  useEffect(() => {
    if (!sessionId) {
      setRegistryEvidence([]);
      return;
    }
    api.evidence(sessionId)
      .then((payload) => setRegistryEvidence(payload.evidence))
      .catch((reason) => setError(String(reason.message || reason)));
  }, [sessionId]);

  const currentEvidence = result?.context?.evidence || registryEvidence;
  const submitQuestion = async (value?: string) => {
    const cleaned = (value ?? query).trim();
    if (!cleaned || busy) return;
    setQuery("");
    setError(null);
    setMessages((items) => [...items, { role: "user", text: cleaned }]);
    setBusy(true);
    setResult(null);
    try {
      const created = await api.answer(cleaned, sessionId);
      const snapshot = await watchJob(created.job_id, setProgress);
      const answer = snapshot.result as AgenticResult;
      setResult(answer);
      setSessionId(answer.session.session_id);
      setMessages((items) => [
        ...items,
        {
          role: "assistant",
          text: answer.answerable ? answer.answer : answer.refusal_reason || "当前证据不足以回答。",
          answerable: answer.answerable,
          outcome: answer.outcome,
        },
      ]);
      setActivePanel("evidence");
      await refresh();
    } catch (reason: any) {
      setError(reason.message || String(reason));
    } finally {
      setBusy(false);
    }
  };

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    void submitQuestion();
  };

  const upload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setError(null);
    setBusy(true);
    try {
      const created = await api.upload(file);
      await watchJob(created.job_id, setProgress);
      await refresh();
    } catch (reason: any) {
      setError(reason.message || String(reason));
    } finally {
      setBusy(false);
      event.target.value = "";
    }
  };

  const openSession = async (id: string) => {
    setSessionId(id);
    setResult(null);
    setSelectedSource(null);
    try {
      const transcript = await api.transcript(id);
      setMessages(transcript.messages);
    } catch (reason: any) {
      setError(reason.message || String(reason));
    }
  };

  const newSession = () => {
    setSessionId(null);
    setMessages([]);
    setResult(null);
    setRegistryEvidence([]);
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark">LR</div>
        <div className="brand-copy">
          <strong>LEO Research Agent</strong>
          <span>Scientific evidence workspace</span>
        </div>
        <div className="runtime-status">
          <span className={`status-dot ${system.llm_configured ? "online" : "warning"}`} />
          <div>
            <strong>{system.llm_configured ? "本地服务已就绪" : "待配置 LLM"}</strong>
            <span>{system.models_initialized ? "模型已常驻" : "模型将在首次问答时加载"}</span>
          </div>
        </div>
      </header>

      <aside className="library-panel">
        <section className="panel-section library-heading">
          <div>
            <span className="eyebrow">LIBRARY</span>
            <h2>论文库</h2>
          </div>
          <button className="icon-button" onClick={() => fileInput.current?.click()} disabled={busy} aria-label="上传 PDF">
            +
          </button>
          <input ref={fileInput} type="file" accept="application/pdf,.pdf" hidden onChange={upload} />
        </section>
        <div className="paper-count">{papers.length} 篇已入库论文</div>
        <div className="paper-list">
          {papers.map((paper) => (
            <article className="paper-card" key={paper.document_id}>
              <div className="paper-year">{paper.year || "—"}</div>
              <div>
                <h3>{paper.title}</h3>
                <p>{formatAuthors(paper.authors)}</p>
                <div className="paper-meta">
                  <span>{paper.page_count} pages</span>
                  <span>{paper.quality_issue_count ? `${paper.quality_issue_count} issues` : "parsed"}</span>
                </div>
              </div>
            </article>
          ))}
          {!papers.length && <div className="empty-mini">尚无论文，点击右上角上传 PDF。</div>}
        </div>
        <section className="session-heading">
          <div>
            <span className="eyebrow">SESSIONS</span>
            <h2>研究会话</h2>
          </div>
          <button className="text-button" onClick={newSession}>NEW</button>
        </section>
        <div className="session-list">
          {sessions.map((session) => (
            <button
              key={session.session_id}
              className={`session-item ${sessionId === session.session_id ? "active" : ""}`}
              onClick={() => void openSession(session.session_id)}
            >
              <span>{session.title}</span>
              <small>{session.active_topic_id || "No topic"}</small>
            </button>
          ))}
        </div>
      </aside>

      <main className="conversation">
        <div className="conversation-header">
          <div>
            <span className="eyebrow">AGENTIC RAG</span>
            <h1>{sessionId ? "继续当前研究主题" : "从本地论文中寻找可验证答案"}</h1>
          </div>
          {sessionId && <code>{sessionId}</code>}
        </div>

        <div className="message-stream">
          {!messages.length && (
            <div className="welcome">
              <div className="welcome-index">01</div>
              <h2>不止给出答案，还要说清证据从哪里来。</h2>
              <p>系统将执行混合检索、精排、证据覆盖检查和 Claim-Citation 验证。</p>
              <div className="suggestions">
                {SUGGESTIONS.map((suggestion) => (
                  <button key={suggestion} onClick={() => void submitQuestion(suggestion)}>
                    <span>↗</span>{suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((message, index) => (
            <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
              <div className="message-label">{message.role === "user" ? "YOU" : "RESEARCH AGENT"}</div>
              <div className="message-body">
                {message.text}
                {message.role === "assistant" && message.answerable === false && (
                  <span className={`refusal-tag ${message.outcome?.code || "unknown"}`}>
                    {OUTCOME_LABELS[message.outcome?.code || ""] || "未完成"}
                  </span>
                )}
              </div>
            </article>
          ))}
          {busy && (
            <div className="working-card">
              <div className="working-line"><span>{progress.stage.replaceAll("_", " ")}</span><strong>{Math.round(progress.progress * 100)}%</strong></div>
              <div className="progress-track"><i style={{ width: `${progress.progress * 100}%` }} /></div>
              <p>{progress.message}</p>
            </div>
          )}
        </div>

        {error && <div className="error-banner"><strong>运行失败</strong><span>{error}</span><button onClick={() => setError(null)}>×</button></div>}
        <form className="composer" onSubmit={onSubmit}>
          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="输入论文问题，或继续追问“那为什么”……"
            rows={2}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void submitQuestion();
              }
            }}
          />
          <div className="composer-footer">
            <span>Enter 发送 · Shift + Enter 换行</span>
            <button type="submit" disabled={busy || !query.trim()}>检索并回答 <span>↑</span></button>
          </div>
        </form>
      </main>

      <aside className="inspection-panel">
        <div className="panel-tabs">
          <button className={activePanel === "evidence" ? "active" : ""} onClick={() => setActivePanel("evidence")}>证据</button>
          <button className={activePanel === "diagnostics" ? "active" : ""} onClick={() => setActivePanel("diagnostics")}>诊断</button>
        </div>
        {activePanel === "evidence" ? (
          <div className="evidence-panel">
            <div className="inspection-summary">
              <strong>{currentEvidence.length}</strong>
              <span>{result ? "当前答案证据" : "Session Evidence Registry"}</span>
            </div>
            {currentEvidence.map((item, index) => {
              const source = item.source_id || item.evidence_id || `E${index + 1}`;
              return (
                <button
                  className={`evidence-card ${selectedSource === source ? "selected" : ""}`}
                  key={`${item.chunk_id}-${source}`}
                  onClick={() => setSelectedSource(selectedSource === source ? null : source)}
                >
                  <div className="evidence-topline"><b>{source}</b><span>{item.origin === "reused" ? "REUSED" : "NEW"}</span></div>
                  <h3>{item.title}</h3>
                  <p className="evidence-location">{item.section_path?.join(" / ") || "Unknown section"} · p.{item.page_start}{item.page_end !== item.page_start ? `–${item.page_end}` : ""}</p>
                  <p className={`evidence-content ${selectedSource === source ? "expanded" : ""}`}>{item.content}</p>
                </button>
              );
            })}
            {!currentEvidence.length && <div className="empty-inspection">完成一次问答后，这里会显示可定位到页码和 Chunk 的证据。</div>}
          </div>
        ) : (
          <Diagnostics result={result} system={system} />
        )}
      </aside>
    </div>
  );
}

function Diagnostics({ result, system }: { result: AgenticResult | null; system: Record<string, any> }) {
  const harness = result?.diagnostics?.harness;
  const selection = result?.diagnostics?.evidence_selection;
  const trace = harness?.trace || [];
  return (
    <div className="diagnostics-panel">
      <section className="config-grid">
        <div><span>Embedding</span><strong>{system.embedding_model || "—"}</strong></div>
        <div><span>Reranker</span><strong>{system.reranker_model || "—"}</strong></div>
        <div><span>Candidate</span><strong>{system.candidate_limit || 20}</strong></div>
        <div><span>Final budget</span><strong>{system.final_top_k || 5}</strong></div>
      </section>
      {!result ? (
        <div className="empty-inspection">完成问答后可查看 Router、Retrieval、Coverage、Validation 和 Harness 轨迹。</div>
      ) : (
        <>
          <section className="diagnostic-block">
            <span className="eyebrow">ROUTING</span>
            <h3>{result.session.relation}</h3>
            <p>{result.session.standalone_query}</p>
          </section>
          <section className="diagnostic-block">
            <span className="eyebrow">COVERAGE</span>
            {result.coverage.coverage.map((item) => (
              <div className="coverage-row" key={item.subquestion_id}>
                <b>{item.subquestion_id}</b><span className={item.status}>{item.status}</span>
              </div>
            ))}
          </section>
          {selection && (
            <section className="diagnostic-block">
              <span className="eyebrow">EVIDENCE SELECTION</span>
              <div className="metric-row"><span>Candidate</span><b>{selection.candidate_count}</b></div>
              <div className="metric-row"><span>Selected</span><b>{selection.selected_count}</b></div>
              <div className="metric-row"><span>Redundant dropped</span><b>{selection.dropped_redundant_count}</b></div>
              <div className="metric-row"><span>Coverage preserved</span><b>{selection.coverage_preserved ? "YES" : "NO"}</b></div>
            </section>
          )}
          <section className="diagnostic-block trace-block">
            <span className="eyebrow">HARNESS TRACE</span>
            {trace.map((item: Record<string, any>) => (
              <div className="trace-row" key={item.ordinal}>
                <i />
                <div><b>{item.stage.replaceAll("_", " ")}</b><span>{item.elapsed_ms} ms</span></div>
              </div>
            ))}
          </section>
        </>
      )}
    </div>
  );
}

export default App;
