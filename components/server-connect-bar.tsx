"use client";

import { useState, useEffect, useCallback } from "react";
import { Wifi, WifiOff, Loader2, Server, AlertTriangle, ExternalLink, Copy, Check, Menu, PanelLeftClose } from "lucide-react";
import { getServerUrl, setServerUrl } from "@/lib/server-store";
import { cn } from "@/lib/utils";

type Status = "idle" | "checking" | "connected" | "error";

interface ServerConnectBarProps {
  onStatusChange?: (connected: boolean, url: string) => void;
  onToggleSidebar?: () => void;
  sidebarOpen?: boolean;
}

// HTTPS 페이지에서 HTTP localhost로 요청하면 Mixed Content 차단됨
function isMixedContent(serverUrl: string): boolean {
  if (typeof window === "undefined") return false;
  const pageIsHttps = window.location.protocol === "https:";
  const serverIsHttp = serverUrl.startsWith("http://");
  return pageIsHttps && serverIsHttp;
}

export default function ServerConnectBar({ onStatusChange, onToggleSidebar, sidebarOpen }: ServerConnectBarProps) {
  const [url, setUrl] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [version, setVersion] = useState<string>("");
  const [ngrokUrl, setNgrokUrl] = useState<string>("");
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [showGuide, setShowGuide] = useState(false);
  const [copied, setCopied] = useState(false);

  const checkConnection = useCallback(async (targetUrl: string) => {
    if (isMixedContent(targetUrl)) {
      setStatus("error");
      onStatusChange?.(false, targetUrl);
      return;
    }
    setStatus("checking");
    try {
      const isNgrok = targetUrl.includes("ngrok");
      const res = await fetch(`${targetUrl.replace(/\/$/, "")}/ping`, {
        signal: AbortSignal.timeout(5000),
        headers: isNgrok ? { "ngrok-skip-browser-warning": "true" } : {},
      });
      if (res.ok) {
        const data = await res.json();
        setStatus("connected");
        setVersion(data.version || "");
        // 서버가 ngrok URL을 이미 알고 있으면 표시
        if (data.ngrok_url) setNgrokUrl(data.ngrok_url);
        onStatusChange?.(true, targetUrl);
      } else {
        setStatus("error");
        onStatusChange?.(false, targetUrl);
      }
    } catch {
      setStatus("error");
      onStatusChange?.(false, targetUrl);
    }
  }, [onStatusChange]);

  useEffect(() => {
    const saved = getServerUrl();
    setUrl(saved);
    setDraft(saved);
    checkConnection(saved);
  }, [checkConnection]);

  const handleSave = () => {
    const trimmed = draft.trim();
    if (!trimmed) return;
    setUrl(trimmed);
    setServerUrl(trimmed);
    setEditing(false);
    setNgrokUrl("");
    checkConnection(trimmed);
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const mixedContent = isMixedContent(url);

  const statusColor = {
    idle: "text-muted",
    checking: "text-warning",
    connected: "text-accent",
    error: mixedContent ? "text-warning" : "text-danger",
  }[status];

  const statusLabel = {
    idle: "연결 전",
    checking: "연결 확인 중...",
    connected: `연결됨${version ? ` v${version}` : ""}`,
    error: mixedContent ? "HTTPS 차단 (ngrok 필요)" : "연결 실패",
  }[status];

  return (
    <div className="flex flex-col bg-surface border-b border-border">
      {/* Main bar */}
      <div className="flex items-center gap-3 px-4 py-2.5">
        {/* 사이드바 토글 버튼 */}
        {onToggleSidebar && (
          <button
            onClick={onToggleSidebar}
            className="p-1.5 -ml-1 rounded text-muted hover:text-foreground hover:bg-surface-2 transition-colors shrink-0"
            aria-label={sidebarOpen ? "사이드바 닫기" : "사이드바 열기"}
          >
            {sidebarOpen ? (
              <PanelLeftClose className="w-4 h-4" />
            ) : (
              <Menu className="w-4 h-4" />
            )}
          </button>
        )}
        <Server className="w-4 h-4 text-muted shrink-0" />
        <span className="text-xs text-muted shrink-0 font-mono hidden sm:inline">로컬 서버</span>

        {editing ? (
          <div className="flex items-center gap-2 flex-1">
            <input
              type="text"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSave();
                if (e.key === "Escape") setEditing(false);
              }}
              className="flex-1 bg-surface-2 border border-border rounded text-xs px-2 py-1 font-mono text-foreground outline-none focus:border-accent transition-colors"
              autoFocus
              placeholder="https://xxxx.ngrok-free.app"
            />
            <button
              onClick={handleSave}
              className="text-xs px-3 py-1 bg-accent text-background rounded font-semibold hover:opacity-90 transition-opacity"
            >
              저장
            </button>
            <button
              onClick={() => setEditing(false)}
              className="text-xs px-3 py-1 bg-surface-2 border border-border text-muted rounded hover:text-foreground transition-colors"
            >
              취소
            </button>
          </div>
        ) : (
          <button
            onClick={() => { setDraft(url); setEditing(true); }}
            className="flex-1 text-left text-xs font-mono text-foreground hover:text-accent transition-colors truncate"
          >
            {url}
          </button>
        )}

        <div className={cn("flex items-center gap-1.5 text-xs shrink-0", statusColor)}>
          {status === "checking" ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : status === "connected" ? (
            <Wifi className="w-3.5 h-3.5" />
          ) : mixedContent ? (
            <AlertTriangle className="w-3.5 h-3.5" />
          ) : (
            <WifiOff className="w-3.5 h-3.5" />
          )}
          {/* 모바일: 아이콘만, sm 이상: 텍스트 표시 */}
          <span className="hidden sm:inline">{statusLabel}</span>
        </div>

        {status !== "checking" && (
          <button
            onClick={() => checkConnection(url)}
            className="text-xs text-muted hover:text-accent transition-colors shrink-0 hidden sm:inline"
            title="다시 연결"
          >
            재시도
          </button>
        )}

        {/* ngrok 가이드 토글 버튼 */}
        {(mixedContent || status === "error") && (
          <button
            onClick={() => setShowGuide((v) => !v)}
            className="text-xs px-2 py-1 rounded bg-warning/10 text-warning border border-warning/20 hover:bg-warning/20 transition-colors shrink-0"
          >
            <span className="hidden sm:inline">ngrok 연결 방법</span>
            <AlertTriangle className="w-3.5 h-3.5 sm:hidden" />
          </button>
        )}
      </div>

      {/* ngrok 가이드 패널 */}
      {showGuide && (
        <div className="mx-4 mb-3 p-4 rounded-lg bg-surface-2 border border-border text-xs space-y-3">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-warning shrink-0 mt-0.5" />
            <p className="text-foreground leading-relaxed">
              <span className="font-semibold text-warning">HTTPS 보안 정책</span>으로 인해 배포된 웹({typeof window !== "undefined" ? window.location.hostname : "vercel.app"})에서는
              <span className="font-mono text-danger"> http://localhost</span> 직접 연결이 차단됩니다.{" "}
              <span className="font-semibold text-accent">ngrok</span>을 사용하면 로컬 서버에 HTTPS URL을 발급받아 연결할 수 있습니다.
            </p>
          </div>

          <div className="space-y-2">
            <p className="font-semibold text-foreground">설치 및 실행 방법</p>

            <div className="space-y-1.5">
              <p className="text-muted">1. ngrok 패키지 설치</p>
              <div className="flex items-center gap-2">
                <code className="flex-1 bg-background px-3 py-1.5 rounded font-mono text-accent border border-border">
                  pip install ngrok
                </code>
                <button
                  onClick={() => handleCopy("pip install ngrok")}
                  className="p-1.5 rounded bg-surface border border-border text-muted hover:text-accent transition-colors"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-accent" /> : <Copy className="w-3.5 h-3.5" />}
                </button>
              </div>
            </div>

            <div className="space-y-1.5">
              <p className="text-muted">2. 서버 실행 (ngrok 자동 시작 포함)</p>
              <div className="flex items-center gap-2">
                <code className="flex-1 bg-background px-3 py-1.5 rounded font-mono text-accent border border-border">
                  python app.py --ngrok
                </code>
                <button
                  onClick={() => handleCopy("python app.py --ngrok")}
                  className="p-1.5 rounded bg-surface border border-border text-muted hover:text-accent transition-colors"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-accent" /> : <Copy className="w-3.5 h-3.5" />}
                </button>
              </div>
            </div>

            <div className="space-y-1.5">
              <p className="text-muted">3. 터미널 출력에서 ngrok URL 복사 후 위 입력란에 붙여넣기</p>
              <code className="block bg-background px-3 py-1.5 rounded font-mono text-muted border border-border">
                {">> ngrok 터널 활성화"}
                <br />
                {">> 공개 URL: https://xxxx.ngrok-free.app"}
              </code>
            </div>

            {ngrokUrl && (
              <div className="flex items-center gap-2 p-2 rounded bg-accent-dim border border-accent/20">
                <span className="text-muted shrink-0">감지된 ngrok URL:</span>
                <code className="flex-1 font-mono text-accent truncate">{ngrokUrl}</code>
                <button
                  onClick={() => {
                    setDraft(ngrokUrl);
                    setUrl(ngrokUrl);
                    setServerUrl(ngrokUrl);
                    setShowGuide(false);
                    checkConnection(ngrokUrl);
                  }}
                  className="text-xs px-2 py-1 rounded bg-accent text-background font-semibold hover:opacity-90 transition-opacity shrink-0"
                >
                  이 URL 사용
                </button>
              </div>
            )}
          </div>

          <div className="flex items-center gap-1.5 pt-1 border-t border-border">
            <ExternalLink className="w-3 h-3 text-muted" />
            <a
              href="https://ngrok.com/signup"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted hover:text-accent transition-colors"
            >
              ngrok.com 무료 계정 가입 (authtoken 발급)
            </a>
          </div>
        </div>
      )}
    </div>
  );
}
