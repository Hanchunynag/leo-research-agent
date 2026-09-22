import * as vscode from "vscode";

type PatchPreview = {
  preview: {
    patch: {
      patch_id: string;
      project_id: string;
      base_hash: string;
      original_content: string;
      proposed_content: string;
      target_section: string;
      change_summary: string;
      warnings: string[];
    };
    status: string;
  };
};

const backendUrl = (): string =>
  vscode.workspace.getConfiguration("scholar").get<string>("backendUrl", "http://127.0.0.1:8000").replace(/\/$/, "");

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${backendUrl()}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  const body = await response.json() as T & { detail?: unknown };
  if (!response.ok) {
    throw new Error(typeof body.detail === "string" ? body.detail : `Scholar backend HTTP ${response.status}`);
  }
  return body;
}

async function patchId(): Promise<string | undefined> {
  return vscode.window.showInputBox({ prompt: "Scholar Runtime Patch ID" });
}

async function openPreview(): Promise<void> {
  const id = await patchId();
  if (!id) return;
  const payload = await requestJson<PatchPreview>(`/api/scholar/patches/${encodeURIComponent(id)}`);
  const patch = payload.preview.patch;
  const original = await vscode.workspace.openTextDocument({ language: "latex", content: patch.original_content });
  const proposed = await vscode.workspace.openTextDocument({ language: "latex", content: patch.proposed_content });
  await vscode.commands.executeCommand(
    "vscode.diff",
    original.uri,
    proposed.uri,
    `Scholar Patch ${patch.patch_id}: ${patch.target_section}`,
  );
  if (patch.change_summary) {
    void vscode.window.showInformationMessage(`Scholar Patch: ${patch.change_summary}`);
  }
  if (patch.warnings.length > 0) {
    void vscode.window.showWarningMessage(`Scholar Patch warnings: ${patch.warnings.join("; ")}`);
  }
}

async function decidePatch(decision: "accept" | "reject"): Promise<void> {
  const id = await patchId();
  if (!id) return;
  const payload = await requestJson<PatchPreview>(`/api/scholar/patches/${encodeURIComponent(id)}`);
  const patch = payload.preview.patch;
  const path = `/api/scholar/patches/${encodeURIComponent(id)}/${decision}`;
  const result = await requestJson<{ status: string; message: string }>(path, {
    method: "POST",
    body: JSON.stringify({
      project_id: patch.project_id,
      expected_base_hash: patch.base_hash,
      actor: "vscode-human",
    }),
  });
  void vscode.window.showInformationMessage(`Scholar Patch ${result.status}: ${result.message}`);
}

async function buildManuscript(): Promise<void> {
  const commands = await vscode.commands.getCommands(true);
  if (!commands.includes("latex-workshop.build")) {
    void vscode.window.showErrorMessage("LATEX_WORKSHOP_UNAVAILABLE: 请安装或启用 LaTeX Workshop。");
    return;
  }
  const projectId = await vscode.window.showInputBox({ prompt: "Scholar Runtime Project ID" });
  if (!projectId) return;
  const result = await requestJson<{ build_id: string; status: string }>(
    `/api/scholar/projects/${encodeURIComponent(projectId)}/build`,
    { method: "POST" },
  );
  await vscode.commands.executeCommand("latex-workshop.build");
  void vscode.window.showInformationMessage(`LaTeX Workshop build ${result.status}: ${result.build_id}`);
}

async function showDiagnostics(): Promise<void> {
  const projectId = await vscode.window.showInputBox({ prompt: "Scholar Runtime Project ID" });
  if (!projectId) return;
  const payload = await requestJson<{ diagnostics: Array<{ severity: string; message: string; file?: string; line?: number }> }>(
    `/api/scholar/projects/${encodeURIComponent(projectId)}/diagnostics`,
  );
  const output = vscode.window.createOutputChannel("Scholar Runtime Diagnostics");
  output.clear();
  for (const diagnostic of payload.diagnostics) {
    output.appendLine(`${diagnostic.severity} ${diagnostic.file ?? ""}:${diagnostic.line ?? ""} ${diagnostic.message}`);
  }
  output.show(true);
}

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("scholar.openPatchPreview", () => openPreview().catch(showError)),
    vscode.commands.registerCommand("scholar.acceptPatch", () => decidePatch("accept").catch(showError)),
    vscode.commands.registerCommand("scholar.rejectPatch", () => decidePatch("reject").catch(showError)),
    vscode.commands.registerCommand("scholar.buildManuscript", () => buildManuscript().catch(showError)),
    vscode.commands.registerCommand("scholar.showDiagnostics", () => showDiagnostics().catch(showError)),
  );
}

export function deactivate(): void {}

function showError(error: unknown): void {
  void vscode.window.showErrorMessage(`Scholar Runtime: ${error instanceof Error ? error.message : String(error)}`);
}
