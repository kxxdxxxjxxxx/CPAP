"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  Upload,
  Plus,
  Trash2,
  CheckCircle2,
  AlertCircle,
  Info,
  RefreshCw,
  Loader2,
  Save,
  Wand2,
  Eye,
  EyeOff,
  HelpCircle,
  Play,
  Download,
  X,
  Copy,
  Check,
  Zap,
} from "lucide-react";
import { apiFetch, getServerUrl } from "@/lib/server-store";
import { cn } from "@/lib/utils";
import KeypointCanvas, {
  PoseKeypoint,
  KEYPOINT_NAMES,
} from "@/components/keypoint-canvas";

interface PoseSession {
  session_id: string;
  image_count: number;
  labeled_count: number;
}

interface PoseImage {
  filename: string;
  width: number;
  height: number;
  labeled: boolean;
  url: string;
}

interface TrainJob {
  status: "idle" | "queued" | "processing" | "done" | "failed";
  error?: string;
  progress?: number;
  log?: string[];
  best_model?: string | null;
  activated?: boolean;
}

const EMPTY_KPTS: PoseKeypoint[] = Array.from({ length: 17 }, (_, i) => ({
  x: 0.5,
  y: 0.2 + (i / 17) * 0.6,
  v: 0,
}));

export default function PoseTrainPage() {
  // ── 세션 ────────────────────────────────────────
  const [sessions, setSessions] = useState<PoseSession[]>([]);
  const [sessionId, setSessionId] = useState("");
  const [newSessionName, setNewSessionName] = useState("");
  const [sessionMsg, setSessionMsg] = useState<{
    type: "info" | "error" | "success";
    text: string;
  } | null>(null);
  const [serverReachable, setServerReachable] = useState<boolean | null>(null);

  // ── 이미지 ───────────────────────────────────────
  const [images, setImages] = useState<PoseImage[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>("");
  const [imageSrc, setImageSrc] = useState<string>("");
  const [imageLoading, setImageLoading] = useState(false);

  // ── 라벨 ────────────────────────────────────────
  const [keypoints, setKeypoints] = useState<PoseKeypoint[]>(EMPTY_KPTS);
  const [selectedKpt, setSelectedKpt] = useState(-1);
  const [bootstrapLoading, setBootstrapLoading] = useState(false);
  const [saveStatus, setSaveStatus] = useState<
    "idle" | "saving" | "saved" | "error"
  >("idle");
  const [saveMsg, setSaveMsg] = useState("");
  const [dirty, setDirty] = useState(false);

  // ── 학습 ────────────────────────────────────────
  const [epochs, setEpochs] = useState(50);
  const [imgsz, setImgsz] = useState(640);
  const [useInitialWeights, setUseInitialWeights] = useState(true);
  const [initialWeights, setInitialWeights] = useState("yolo11l-pose.pt");
  // 스택 모드: 켜져 있으면 현재 활성 포즈 모델에서 이어 학습 (능동학습 누적)
  const [stackMode, setStackMode] = useState(true);
  const [activeModelPath, setActiveModelPath] = useState("");
  const [autoActivate, setAutoActivate] = useState(true);
  const [copiedPath, setCopiedPath] = useState(false);
  const [activating, setActivating] = useState(false);
  const [activateMsg, setActivateMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [trainJobId, setTrainJobId] = useState("");
  const [trainJob, setTrainJob] = useState<TrainJob>({ status: "idle" });
  const [downloadUrl, setDownloadUrl] = useState("");

  // ── 업로드 ─────────────────────────────────────
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 이전 blob URL 정리
  useEffect(() => {
    return () => {
      if (imageSrc.startsWith("blob:")) URL.revokeObjectURL(imageSrc);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── 세션 목록 로드 ───────────────────────────────
  const loadSessions = useCallback(async () => {
    try {
      const res = await apiFetch("/pose/sessions");
      // 404 = 서버는 살아있지만 /pose/* 엔드포인트가 없는 구버전 app.py
      if (res.status === 404) {
        setServerReachable(true);
        setSessionMsg({
          type: "error",
          text:
            "로컬 서버는 연결되었지만 포즈 학습 API(/pose/*)가 없습니다. " +
            "최신 app.py로 업데이트한 뒤 'python app.py --ngrok'로 재시작하세요. " +
            "(v0에서 ZIP 다운로드 → 로컬 폴더의 app.py 덮어쓰기)",
        });
        setSessions([]);
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const ct = res.headers.get("content-type") || "";
      if (ct.includes("text/html"))
        throw new Error("서버가 HTML을 반환 (ngrok 경고 페이지일 수 있음)");
      const json = await res.json();
      setSessions(json.sessions || []);
      setServerReachable(true);
      setSessionMsg(null);
    } catch (err) {
      setServerReachable(false);
      setSessionMsg({
        type: "error",
        text: `로컬 서버 응답 없음: ${err instanceof Error ? err.message : "오류"}. 상단 "로컬 서버" URL이 연결됐는지 확인하세요.`,
      });
    }
  }, []);

  useEffect(() => {
    loadSessions();
    const onStorage = () => loadSessions();
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [loadSessions]);

  // ── 활성 포즈 모델 경로 로드 (스택 모드용) ─────────────
  const loadActiveModel = useCallback(async () => {
    try {
      const res = await apiFetch("/params");
      if (!res.ok) return;
      const j = await res.json();
      const p = (j?.params?.pose_model_path as string | undefined) || "";
      setActiveModelPath(p);
      if (p && stackMode) setInitialWeights(p);
    } catch {
      // 무시
    }
  }, [stackMode]);

  useEffect(() => {
    loadActiveModel();
  }, [loadActiveModel]);

  useEffect(() => {
    if (trainJob.status === "done" && trainJob.activated) {
      loadActiveModel();
    }
  }, [trainJob.status, trainJob.activated, loadActiveModel]);

  // ── 세션 생성/삭제 ─────────────────────────────────
  const handleCreateSession = async () => {
    setSessionMsg({ type: "info", text: "세션 생성 중..." });
    const fd = new FormData();
    fd.append("name", newSessionName);
    try {
      const res = await apiFetch("/pose/sessions", {
        method: "POST",
        body: fd,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setSessionId(json.session_id);
      setNewSessionName("");
      setSessionMsg({
        type: "success",
        text: `세션 "${json.session_id}" 생성됨`,
      });
      setTimeout(() => setSessionMsg(null), 2500);
      await loadSessions();
    } catch (err) {
      setSessionMsg({
        type: "error",
        text: `세션 생성 실패: ${err instanceof Error ? err.message : "오류"}`,
      });
    }
  };

  const handleDeleteSession = async (sid: string) => {
    if (!confirm(`세션 "${sid}" 을(를) 삭제할까요? (이미지/라벨 영구 삭제)`))
      return;
    try {
      await apiFetch(`/pose/sessions/${sid}`, { method: "DELETE" });
      if (sessionId === sid) {
        setSessionId("");
        setImages([]);
        setSelectedFile("");
        setImageSrc("");
      }
      await loadSessions();
    } catch {
      /* noop */
    }
  };

  // ── 이미지 목록/선택 ─────────────────────────────
  const loadImages = useCallback(async (sid: string) => {
    if (!sid) {
      setImages([]);
      return;
    }
    try {
      const res = await apiFetch(`/pose/images/${sid}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setImages(json.images || []);
    } catch {
      setImages([]);
    }
  }, []);

  useEffect(() => {
    loadImages(sessionId);
    setSelectedFile("");
    if (imageSrc.startsWith("blob:")) URL.revokeObjectURL(imageSrc);
    setImageSrc("");
    setKeypoints(EMPTY_KPTS);
    setDirty(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const handleSelectImage = async (img: PoseImage) => {
    if (
      dirty &&
      !confirm(
        "라벨이 저장되지 않았습니다. 다른 이미지로 이동하면 변경사항이 사라집니다. 계속할까요?"
      )
    ) {
      return;
    }
    setSelectedFile(img.filename);
    setImageLoading(true);
    setSaveStatus("idle");
    setSaveMsg("");
    setSelectedKpt(-1);
    setDirty(false);

    if (imageSrc.startsWith("blob:")) URL.revokeObjectURL(imageSrc);

    try {
      // 이미지 다운로드 (ngrok 우회)
      const r = await apiFetch(img.url);
      if (!r.ok) throw new Error(`이�����지 로드 실패 HTTP ${r.status}`);
      const blob = await r.blob();
      const blobUrl = URL.createObjectURL(blob);
      setImageSrc(blobUrl);

      // 저장된 라벨 로드
      const lblRes = await apiFetch(
        `/pose/labels/${sessionId}/${encodeURIComponent(img.filename)}`
      );
      if (lblRes.ok) {
        const j = await lblRes.json();
        if (Array.isArray(j.keypoints) && j.keypoints.length === 17) {
          setKeypoints(
            j.keypoints.map((k: PoseKeypoint) => ({
              x: k.x,
              y: k.y,
              v: (k.v ?? 0) as 0 | 1 | 2,
            }))
          );
        } else {
          setKeypoints(EMPTY_KPTS);
        }
      } else {
        setKeypoints(EMPTY_KPTS);
      }
    } catch (err) {
      setSaveMsg(
        `이미지 로드 오류: ${err instanceof Error ? err.message : "오류"}`
      );
    } finally {
      setImageLoading(false);
    }
  };

  const handleDeleteImage = async (img: PoseImage) => {
    if (!confirm(`"${img.filename}" 삭제?`)) return;
    try {
      await apiFetch(
        `/pose/image/${sessionId}/${encodeURIComponent(img.filename)}`,
        { method: "DELETE" }
      );
      if (selectedFile === img.filename) {
        setSelectedFile("");
        setImageSrc("");
        setKeypoints(EMPTY_KPTS);
      }
      await loadImages(sessionId);
    } catch {
      /* noop */
    }
  };

  // ── 업로드 ─────────────────────────────────────────
  const handleFiles = async (files: FileList | File[]) => {
    if (!sessionId) {
      alert("먼저 세션을 선택하세요.");
      return;
    }
    const arr = Array.from(files).filter((f) => f.type.startsWith("image/"));
    if (arr.length === 0) return;
    setUploading(true);
    try {
      const fd = new FormData();
      for (const f of arr) fd.append("files", f);
      await apiFetch(`/pose/upload/${sessionId}`, {
        method: "POST",
        body: fd,
      });
      await loadImages(sessionId);
      await loadSessions();
    } finally {
      setUploading(false);
    }
  };

  // ── 라벨링 ─────────────────────────────────────────
  const handleKeypointsChange = (next: PoseKeypoint[]) => {
    setKeypoints(next);
    setDirty(true);
    setSaveStatus("idle");
  };

  const handleBootstrap = async () => {
    if (!sessionId || !selectedFile) return;
    setBootstrapLoading(true);
    setSaveMsg("");
    try {
      const res = await apiFetch(
        `/pose/predict/${sessionId}/${encodeURIComponent(selectedFile)}`,
        { method: "POST" }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      if (Array.isArray(json.keypoints) && json.keypoints.length === 17) {
        setKeypoints(
          json.keypoints.map((k: PoseKeypoint) => ({
            x: k.x,
            y: k.y,
            v: (k.v ?? 0) as 0 | 1 | 2,
          }))
        );
        setDirty(true);
        setSaveStatus("idle");
        setSaveMsg(
          json.found
            ? "현재 모델로 키포인트 부트스트랩 완료. 어긋난 점만 드래그/클릭으로 수정하세요."
            : "사람을 찾지 못했습니다. 모든 점이 중앙에 있으니 드래그하여 위치를 잡아주세요."
        );
      }
    } catch (err) {
      setSaveMsg(
        `부트스트랩 실패: ${err instanceof Error ? err.message : "오류"}`
      );
    } finally {
      setBootstrapLoading(false);
    }
  };

  const handleSaveLabels = async () => {
    if (!sessionId || !selectedFile) return;
    const visibleCount = keypoints.filter((k) => k.v > 0).length;
    if (visibleCount < 4) {
      setSaveStatus("error");
      setSaveMsg(
        `보임/가려짐으로 표시된 키포인트가 ${visibleCount}개 입니다. 최소 4개 이상 필요합니다.`
      );
      return;
    }
    setSaveStatus("saving");
    setSaveMsg("");
    try {
      const res = await apiFetch(
        `/pose/labels/${sessionId}/${encodeURIComponent(selectedFile)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ keypoints }),
        }
      );
      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status} ${txt.slice(0, 120)}`);
      }
      setSaveStatus("saved");
      setSaveMsg("라벨 저장됨");
      setDirty(false);
      await loadImages(sessionId);
      await loadSessions();
      setTimeout(() => setSaveStatus((s) => (s === "saved" ? "idle" : s)), 2000);
    } catch (err) {
      setSaveStatus("error");
      setSaveMsg(`저장 실패: ${err instanceof Error ? err.message : "오류"}`);
    }
  };

  const handleResetKeypoints = () => {
    if (!confirm("현재 라벨을 모두 초기화할까요?")) return;
    setKeypoints(EMPTY_KPTS);
    setDirty(true);
    setSaveStatus("idle");
  };

  const handleToggleAllVisibility = (target: 0 | 1 | 2) => {
    setKeypoints(keypoints.map((k) => ({ ...k, v: target })));
    setDirty(true);
  };

  // 우측 패널: 키포인트 행 클릭 → 해당 점을 화면 중앙으로 이동(=처음 라벨링 시)
  const handleKptRowClick = (i: number) => {
    setSelectedKpt(i);
    if (keypoints[i].v === 0) {
      // 화면 중앙에 배치하면서 visible 처리
      const next = [...keypoints];
      next[i] = { ...next[i], x: 0.5, y: 0.5, v: 2 };
      setKeypoints(next);
      setDirty(true);
    }
  };

  const handleKptVisibility = (i: number, v: 0 | 1 | 2) => {
    const next = [...keypoints];
    next[i] = { ...next[i], v };
    setKeypoints(next);
    setDirty(true);
    setSelectedKpt(i);
  };

  // ── 학습 ─────────────────────────────────────────
  const labeledCount = images.filter((i) => i.labeled).length;
  const canTrain = sessionId && labeledCount >= 2;
  const isTraining =
    trainJob.status === "queued" || trainJob.status === "processing";

  const pollTrain = useCallback(async (jid: string) => {
    try {
      const res = await apiFetch(`/train/status/${jid}`);
      if (!res.ok) return;
      const j = await res.json();
      setTrainJob({
        status: j.status,
        error: j.error,
        progress: j.progress ?? 0,
        log: j.log ?? [],
        best_model: j.best_model ?? null,
        activated: !!j.activated,
      });
      if (j.status === "queued" || j.status === "processing") {
        setTimeout(() => pollTrain(jid), 1500);
      }
    } catch {
      setTimeout(() => pollTrain(jid), 3000);
    }
  }, []);

  const handleStartTrain = async () => {
    if (!canTrain) return;
    setTrainJob({ status: "queued", log: [], progress: 0 });
    try {
      const params = new URLSearchParams({
        epochs: String(epochs),
        imgsz: String(imgsz),
        initial_weights: useInitialWeights ? initialWeights : "",
        auto_activate: String(autoActivate),
      });
      const res = await apiFetch(
        `/pose/train/${sessionId}?${params.toString()}`,
        { method: "POST" }
      );
      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status} ${txt.slice(0, 120)}`);
      }
      const json = await res.json();
      setTrainJobId(json.train_job_id);
      const base = getServerUrl().replace(/\/$/, "");
      setDownloadUrl(`${base}/train/download/${json.train_job_id}`);
      pollTrain(json.train_job_id);
    } catch (err) {
      setTrainJob({
        status: "failed",
        error: err instanceof Error ? err.message : "오류",
      });
    }
  };

  const isDone = trainJob.status === "done";

  // ──────────────────────────────────────────────────
  return (
    <div className="max-w-7xl mx-auto py-6 px-4 md:px-6">
      <header className="mb-5">
        <h1 className="text-xl font-bold text-foreground mb-1">
          포즈 학습 (키포인트 라벨링 + 파인튜닝)
        </h1>
        <p className="text-xs text-muted leading-relaxed">
          벽에 붙은 클라이밍 자세 사진을 업로드한 뒤 17개 키포인트를 클릭/드래그로
          라벨링하고, 그 데이터로 yolo11l-pose 모델을 파인튜닝합니다. "예측
          가져오기"로 현재 모델의 추정값을 바로 띄우고, 어긋난 점만 보정하면
          빠릅니다.
        </p>
      </header>

      {/* ─── 세션 ─── */}
      <section className="mb-5 p-4 bg-surface rounded-lg border border-border">
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs font-semibold text-muted uppercase tracking-wider">
            1. 라벨링 세션
          </p>
          <button
            onClick={loadSessions}
            className="text-xs text-muted hover:text-foreground flex items-center gap-1"
          >
            <RefreshCw className="w-3 h-3" /> 새로고침
          </button>
        </div>

        {serverReachable === false && (
          <div className="mb-3 p-2.5 rounded text-xs flex items-start gap-2 bg-warning/10 border border-warning/20 text-warning">
            <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            <span>
              로컬 서버 미연결. 상단 "로컬 서버" 바에서 ngrok URL을 입력하고
              연결되어야 합니다.
            </span>
          </div>
        )}

        {sessionMsg && (
          <div
            className={cn(
              "mb-3 p-2.5 rounded text-xs flex items-start gap-2 border",
              sessionMsg.type === "error" &&
                "bg-danger/10 border-danger/30 text-danger",
              sessionMsg.type === "success" &&
                "bg-accent-dim border-accent/30 text-accent",
              sessionMsg.type === "info" &&
                "bg-surface-2 border-border text-muted"
            )}
          >
            {sessionMsg.type === "error" ? (
              <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            ) : sessionMsg.type === "success" ? (
              <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            ) : (
              <Info className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            )}
            <span>{sessionMsg.text}</span>
          </div>
        )}

        <div className="flex gap-2 mb-3">
          <input
            type="text"
            value={newSessionName}
            onChange={(e) => setNewSessionName(e.target.value)}
            placeholder="세션 이름 (영문/숫자, 비우면 자동 생성)"
            className="flex-1 px-3 py-2 rounded bg-surface-2 border border-border text-sm text-foreground placeholder:text-muted focus:outline-none focus:border-accent"
          />
          <button
            onClick={handleCreateSession}
            disabled={serverReachable === false}
            className="flex items-center gap-1.5 px-4 py-2 rounded bg-accent text-background text-sm font-medium hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Plus className="w-4 h-4" /> 세션 생성
          </button>
        </div>

        <div className="flex flex-wrap gap-2">
          {sessions.length === 0 && (
            <p className="text-xs text-muted py-2">세션 없음</p>
          )}
          {sessions.map((s) => (
            <button
              key={s.session_id}
              onClick={() => setSessionId(s.session_id)}
              className={cn(
                "group flex items-center gap-2 px-3 py-1.5 rounded text-xs border transition-colors",
                sessionId === s.session_id
                  ? "bg-accent-dim text-accent border-accent/30"
                  : "bg-surface-2 text-foreground border-border hover:border-foreground/30"
              )}
            >
              <span className="font-mono">{s.session_id}</span>
              <span className="text-muted">
                {s.labeled_count}/{s.image_count}
              </span>
              <span
                role="button"
                tabIndex={0}
                onClick={(e) => {
                  e.stopPropagation();
                  handleDeleteSession(s.session_id);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    e.stopPropagation();
                    handleDeleteSession(s.session_id);
                  }
                }}
                className="opacity-0 group-hover:opacity-100 hover:text-danger inline-flex"
                aria-label="세션 삭제"
              >
                <Trash2 className="w-3 h-3" />
              </span>
            </button>
          ))}
        </div>
      </section>

      {/* ─── 업로드 ─── */}
      {sessionId && (
        <section className="mb-5 p-4 bg-surface rounded-lg border border-border">
          <p className="text-xs font-semibold text-muted uppercase tracking-wider mb-3">
            2. 클라이밍 자세 사진 업로드
          </p>
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              handleFiles(e.dataTransfer.files);
            }}
            onClick={() => fileInputRef.current?.click()}
            className={cn(
              "flex flex-col items-center justify-center gap-2 border-2 border-dashed rounded-lg py-6 cursor-pointer transition-colors",
              dragOver
                ? "border-accent bg-accent-dim"
                : "border-border bg-surface-2 hover:border-foreground/30"
            )}
          >
            <Upload className="w-6 h-6 text-muted" />
            <p className="text-sm text-foreground">
              {uploading
                ? "업로드 중..."
                : "이미지를 드래그하거나 클릭하여 선택"}
            </p>
            <p className="text-xs text-muted">
              여러 장 가능 · jpg / png / webp
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={(e) => {
                if (e.target.files) handleFiles(e.target.files);
                e.target.value = "";
              }}
            />
          </div>
        </section>
      )}

      {/* ─── 라벨링 작업영역 ─── */}
      {sessionId && (
        <section className="mb-5">
          <p className="text-xs font-semibold text-muted uppercase tracking-wider mb-2">
            3. 키포인트 라벨링
          </p>
          <div className="grid grid-cols-12 gap-3">
            {/* 좌측: 이미지 ��스트 */}
            <div className="col-span-12 md:col-span-3 bg-surface border border-border rounded-lg overflow-hidden flex flex-col max-h-[680px]">
              <div className="px-3 py-2 border-b border-border flex items-center justify-between">
                <span className="text-xs font-semibold text-muted">
                  이미지 ({images.length})
                </span>
                <span className="text-[10px] text-muted">
                  라벨 {labeledCount}/{images.length}
                </span>
              </div>
              <div className="flex-1 overflow-y-auto">
                {images.length === 0 && (
                  <p className="px-3 py-4 text-xs text-muted">
                    이미지를 업로드하세요.
                  </p>
                )}
                {images.map((img) => (
                  <button
                    key={img.filename}
                    onClick={() => handleSelectImage(img)}
                    className={cn(
                      "w-full flex items-center gap-2 px-3 py-2 text-left border-b border-border/50 transition-colors",
                      selectedFile === img.filename
                        ? "bg-accent-dim"
                        : "hover:bg-surface-2"
                    )}
                  >
                    {img.labeled ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-accent shrink-0" />
                    ) : (
                      <div className="w-3.5 h-3.5 rounded-full border border-border shrink-0" />
                    )}
                    <span
                      className={cn(
                        "flex-1 truncate text-xs",
                        selectedFile === img.filename
                          ? "text-accent"
                          : "text-foreground"
                      )}
                    >
                      {img.filename}
                    </span>
                    <span
                      role="button"
                      tabIndex={0}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteImage(img);
                      }}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          e.stopPropagation();
                          handleDeleteImage(img);
                        }
                      }}
                      className="text-muted hover:text-danger inline-flex"
                      aria-label="이미지 삭제"
                    >
                      <X className="w-3 h-3" />
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* 중앙: 캔버스 */}
            <div className="col-span-12 md:col-span-6 flex flex-col gap-3">
              {selectedFile ? (
                <>
                  <div className="bg-surface border border-border rounded-lg p-3 flex flex-col gap-3">
                    {imageLoading ? (
                      <div className="flex items-center justify-center h-80 bg-black rounded-lg">
                        <Loader2 className="w-5 h-5 animate-spin text-muted" />
                      </div>
                    ) : (
                      <KeypointCanvas
                        imageSrc={imageSrc}
                        imageWidth={
                          images.find((i) => i.filename === selectedFile)
                            ?.width || 0
                        }
                        imageHeight={
                          images.find((i) => i.filename === selectedFile)
                            ?.height || 0
                        }
                        keypoints={keypoints}
                        onKeypointsChange={handleKeypointsChange}
                        selectedIndex={selectedKpt}
                        onSelect={setSelectedKpt}
                      />
                    )}

                    {/* 액션 버튼 */}
                    <div className="flex flex-wrap gap-2">
                      <button
                        onClick={handleBootstrap}
                        disabled={bootstrapLoading || imageLoading}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-surface-2 border border-border text-xs font-medium text-foreground hover:bg-surface-2/70 disabled:opacity-50"
                        title="현재 yolo11l-pose 모델로 자동 추정"
                      >
                        {bootstrapLoading ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <Wand2 className="w-3.5 h-3.5" />
                        )}
                        예측 가져오기
                      </button>
                      <button
                        onClick={() => handleToggleAllVisibility(0)}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-surface-2 border border-border text-xs text-muted hover:text-foreground"
                      >
                        <EyeOff className="w-3.5 h-3.5" /> 모두 없음
                      </button>
                      <button
                        onClick={handleResetKeypoints}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-surface-2 border border-border text-xs text-muted hover:text-foreground"
                      >
                        <Trash2 className="w-3.5 h-3.5" /> 초기화
                      </button>
                      <div className="flex-1" />
                      <button
                        onClick={handleSaveLabels}
                        disabled={
                          saveStatus === "saving" || imageLoading || !dirty
                        }
                        className={cn(
                          "flex items-center gap-1.5 px-4 py-1.5 rounded text-xs font-medium",
                          saveStatus === "saving"
                            ? "bg-surface-2 text-muted border border-border cursor-not-allowed"
                            : saveStatus === "saved"
                              ? "bg-accent-dim text-accent border border-accent/30"
                              : "bg-accent text-background hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed"
                        )}
                      >
                        {saveStatus === "saving" ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : saveStatus === "saved" ? (
                          <CheckCircle2 className="w-3.5 h-3.5" />
                        ) : (
                          <Save className="w-3.5 h-3.5" />
                        )}
                        라벨 저장
                      </button>
                    </div>

                    {saveMsg && (
                      <p
                        className={cn(
                          "text-[11px] flex items-start gap-1.5 leading-relaxed",
                          saveStatus === "error"
                            ? "text-danger"
                            : saveStatus === "saved"
                              ? "text-accent"
                              : "text-muted"
                        )}
                      >
                        {saveStatus === "error" ? (
                          <AlertCircle className="w-3 h-3 mt-0.5 shrink-0" />
                        ) : (
                          <Info className="w-3 h-3 mt-0.5 shrink-0" />
                        )}
                        <span>{saveMsg}</span>
                      </p>
                    )}

                    {/* 사용 안내 */}
                    <div className="text-[11px] text-muted leading-relaxed flex items-start gap-1.5 p-2.5 rounded bg-surface-2 border border-border">
                      <HelpCircle className="w-3 h-3 mt-0.5 shrink-0" />
                      <div>
                        <p>
                          <span className="text-foreground font-medium">드래그</span>{" "}
                          → 키포인트 위치 이동.{" "}
                          <span className="text-foreground font-medium">클릭</span>{" "}
                          → 가시성 사이클(보임 → 가려짐 → 없음 → 보임).
                        </p>
                        <p className="mt-1">
                          <span className="text-accent">●</span> 진한 점 = 보임 ·{" "}
                          <span className="text-accent">○</span> 점선 테두리 ={" "}
                          <span className="text-foreground font-medium">가려짐</span>
                          (위치는 추정) · 옅은 점 ={" "}
                          <span className="text-foreground font-medium">없음</span>(라벨
                          제외)
                        </p>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="bg-surface border border-border rounded-lg p-8 text-center text-sm text-muted">
                  좌측에서 이미지를 선택하세요.
                </div>
              )}
            </div>

            {/* 우측: 키포인트 패널 */}
            <div className="col-span-12 md:col-span-3 bg-surface border border-border rounded-lg overflow-hidden flex flex-col max-h-[680px]">
              <div className="px-3 py-2 border-b border-border">
                <span className="text-xs font-semibold text-muted">
                  키포인트 17개
                </span>
              </div>
              <div className="flex-1 overflow-y-auto">
                {keypoints.map((kp, i) => {
                  const isSel = selectedKpt === i;
                  const colorClass =
                    i === 0
                      ? "bg-foreground"
                      : i >= 1 && i <= 4
                        ? "bg-muted"
                        : i % 2 === 1
                          ? "bg-accent"
                          : "bg-warning";
                  return (
                    <div
                      key={i}
                      className={cn(
                        "flex items-center gap-2 px-3 py-1.5 border-b border-border/50 cursor-pointer",
                        isSel ? "bg-accent-dim" : "hover:bg-surface-2"
                      )}
                      onClick={() => handleKptRowClick(i)}
                    >
                      <span
                        className={cn(
                          "w-2.5 h-2.5 rounded-full shrink-0",
                          colorClass,
                          kp.v === 0 && "opacity-30"
                        )}
                      />
                      <span className="text-[11px] text-muted font-mono w-4 shrink-0">
                        {i}
                      </span>
                      <span
                        className={cn(
                          "flex-1 text-xs truncate",
                          isSel ? "text-accent" : "text-foreground"
                        )}
                      >
                        {KEYPOINT_NAMES[i]}
                      </span>
                      {/* 가시성 토글 */}
                      <div className="flex items-center gap-0.5">
                        {([2, 1, 0] as const).map((v) => (
                          <button
                            key={v}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleKptVisibility(i, v);
                            }}
                            className={cn(
                              "w-5 h-5 rounded text-[10px] font-mono",
                              kp.v === v
                                ? v === 2
                                  ? "bg-accent text-background"
                                  : v === 1
                                    ? "bg-warning text-background"
                                    : "bg-surface-2 text-muted border border-border"
                                : "text-muted hover:bg-surface-2"
                            )}
                            title={
                              v === 2
                                ? "보임 (visible)"
                                : v === 1
                                  ? "가려짐 (occluded, 위치 추정)"
                                  : "없음 (라벨 제외)"
                            }
                          >
                            {v === 2 ? "V" : v === 1 ? "O" : "X"}
                          </button>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="px-3 py-2 border-t border-border bg-surface-2/50 text-[10px] text-muted leading-relaxed space-y-0.5">
                <p>
                  <span className="text-accent">V</span> 보임 ·{" "}
                  <span className="text-warning">O</span> 가려짐 ·{" "}
                  <span className="text-foreground">X</span> 없음
                </p>
                <p>
                  보임/가려짐 합쳐 4개 이상 라벨링되어야 저장됩니다.
                </p>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ─── 학습 시작 ─── */}
      {sessionId && (
        <section className="mb-5 p-4 bg-surface rounded-lg border border-border">
          <p className="text-xs font-semibold text-muted uppercase tracking-wider mb-3">
            4. 파인튜닝 학습
          </p>

          <div className="grid grid-cols-2 gap-3 mb-3">
            <div>
              <label className="block text-xs text-muted mb-1.5">
                Epochs
              </label>
              <input
                type="number"
                min={1}
                max={500}
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value) || 50)}
                className="w-full px-3 py-2 rounded bg-surface-2 border border-border text-sm text-foreground focus:outline-none focus:border-accent"
              />
            </div>
            <div>
              <label className="block text-xs text-muted mb-1.5">
                ImgSize
              </label>
              <input
                type="number"
                min={320}
                max={1280}
                step={32}
                value={imgsz}
                onChange={(e) => setImgsz(Number(e.target.value) || 640)}
                className="w-full px-3 py-2 rounded bg-surface-2 border border-border text-sm text-foreground focus:outline-none focus:border-accent"
              />
            </div>
          </div>

          <label className="flex items-center gap-2 mb-3 cursor-pointer">
            <input
              type="checkbox"
              checked={useInitialWeights}
              onChange={(e) => setUseInitialWeights(e.target.checked)}
              className="accent-accent"
            />
            <span className="text-sm text-foreground">
              사전학습/사용자 가중치에서 fine-tuning
            </span>
          </label>

          {useInitialWeights && (
            <div className="mb-3">
              {/* 스택 모드 토글 — 활성 포즈 모델에서 이어 학습 */}
              <label className="flex items-start gap-2 mb-2 cursor-pointer p-2.5 rounded bg-surface-2 border border-border">
                <input
                  type="checkbox"
                  checked={stackMode}
                  onChange={(e) => {
                    const v = e.target.checked;
                    setStackMode(v);
                    if (v && activeModelPath) setInitialWeights(activeModelPath);
                  }}
                  className="accent-accent mt-0.5"
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-sm text-foreground font-medium">현재 활성 포즈 모델에서 이어 학습 (누적)</span>
                    {activeModelPath && stackMode && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent-dim text-accent border border-accent/20">
                        v1 → v2 → v3 ...
                      </span>
                    )}
                  </div>
                  {activeModelPath ? (
                    <p className="text-[11px] text-muted mt-0.5 font-mono truncate" title={activeModelPath}>
                      활성: {activeModelPath}
                    </p>
                  ) : (
                    <p className="text-[11px] text-muted mt-0.5">
                      활성 포즈 모델 없음 — 아래 경로(기본값)에서 시작합니다.
                    </p>
                  )}
                </div>
              </label>

              <label className="block text-xs text-muted mb-1.5">
                초기 가중치 경로
              </label>
              <input
                type="text"
                value={initialWeights}
                onChange={(e) => {
                  setInitialWeights(e.target.value);
                  if (stackMode && e.target.value !== activeModelPath) setStackMode(false);
                }}
                placeholder="yolo11l-pose.pt 또는 C:/Project/.../best.pt"
                className="w-full px-3 py-2 rounded bg-surface-2 border border-border text-sm font-mono text-foreground focus:outline-none focus:border-accent"
              />
              <p className="text-[11px] text-muted mt-1.5 leading-relaxed">
                스택 모드를 켜두면 페이지 로드 시 자동으로 현재 활성 best.pt에서 이어 학습합니다. 직접 수정하면 그 경로에서 fine-tuning합니다. 체크박스 자체를 풀면 <span className="font-mono">yolo11l-pose.pt</span>(사전학습)에서 새로 시작합니다.
              </p>
            </div>
          )}

          <label className="flex items-start gap-2 mb-4 cursor-pointer p-3 rounded bg-accent-dim border border-accent/20">
            <input
              type="checkbox"
              checked={autoActivate}
              onChange={(e) => setAutoActivate(e.target.checked)}
              className="accent-accent mt-0.5"
            />
            <div className="flex-1">
              <span className="text-sm text-foreground font-medium">
                학습 완료 시 활성 포즈 모델로 자동 적용
              </span>
              <p className="text-[11px] text-muted mt-0.5 leading-relaxed">
                체크하면 학습이 끝나는 즉시 새 best.pt가 영상 분석에 사용됩니다. 파라미터 페이지에서 별도로 경로를 저장할 필요가 없고, 서버 재시작 후에도 그대로 유지됩니다.
              </p>
            </div>
          </label>

          <div className="flex items-center gap-3">
            <button
              onClick={handleStartTrain}
              disabled={!canTrain || isTraining}
              className="flex items-center gap-1.5 px-4 py-2 rounded bg-accent text-background text-sm font-medium hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {isTraining ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              학습 시작
            </button>
            <span className="text-xs text-muted">
              라벨링된 이미지 {labeledCount}장 (최소 2장)
            </span>
            {isDone && downloadUrl && (
              <a
                href={downloadUrl}
                className="flex items-center gap-1.5 text-xs text-accent hover:underline font-medium ml-auto"
              >
                <Download className="w-3.5 h-3.5" />
                best.pt 다운로드
              </a>
            )}
          </div>

          {/* 진행 / 로그 */}
          {(isTraining || isDone || trainJob.status === "failed") && (
            <div className="mt-4 p-3 rounded bg-surface-2 border border-border">
              <div className="flex items-center justify-between mb-2">
                <span
                  className={cn(
                    "text-xs font-medium flex items-center gap-1.5",
                    trainJob.status === "done" && "text-accent",
                    trainJob.status === "failed" && "text-danger",
                    isTraining && "text-foreground"
                  )}
                >
                  {trainJob.status === "done" ? (
                    <CheckCircle2 className="w-3.5 h-3.5" />
                  ) : trainJob.status === "failed" ? (
                    <AlertCircle className="w-3.5 h-3.5" />
                  ) : (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  )}
                  {trainJob.status === "queued"
                    ? "대기 중"
                    : trainJob.status === "processing"
                      ? "학습 진행 중"
                      : trainJob.status === "done"
                        ? "학습 완료"
                        : "학습 실패"}
                </span>
                <span className="text-[11px] font-mono text-muted">
                  {trainJob.progress ?? 0}%
                </span>
              </div>
              <div className="h-1.5 bg-background rounded overflow-hidden mb-2">
                <div
                  className={cn(
                    "h-full transition-all",
                    trainJob.status === "failed"
                      ? "bg-danger"
                      : trainJob.status === "done"
                        ? "bg-accent"
                        : "bg-foreground"
                  )}
                  style={{ width: `${trainJob.progress ?? 0}%` }}
                />
              </div>
              {trainJob.error && (
                <p className="text-xs text-danger mb-2 font-mono break-all">
                  {trainJob.error}
                </p>
              )}
              {trainJob.log && trainJob.log.length > 0 && (
                <div className="max-h-48 overflow-y-auto text-[10px] font-mono text-muted leading-relaxed bg-background rounded p-2">
                  {trainJob.log.map((l, i) => (
                    <div key={i}>{l}</div>
                  ))}
                </div>
              )}

              {isDone && trainJob.best_model && (
                <div className="mt-3 pt-3 border-t border-border space-y-2">
                  {trainJob.activated ? (
                    <div className="flex items-center gap-1.5 text-xs text-accent">
                      <Zap className="w-3.5 h-3.5" />
                      <span className="font-medium">활성 포즈 모델로 적용됨</span>
                      <span className="text-muted">— 분석에 즉시 반영, 재시작 후에도 유지</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-1.5 text-xs text-muted">
                      <span>이 모델은 아직 활성화되지 않았습니다. 아래 "활성 모델로 적용" 버튼을 누르세요.</span>
                    </div>
                  )}
                  <div>
                    <p className="text-[11px] text-muted mb-1">best.pt 경로 (서버 로컬)</p>
                    <div className="flex items-center gap-2 flex-wrap">
                      <code className="flex-1 min-w-0 px-2.5 py-1.5 rounded bg-background border border-border text-[11px] font-mono text-foreground overflow-x-auto whitespace-nowrap">
                        {trainJob.best_model}
                      </code>
                      <button
                        onClick={async () => {
                          try {
                            await navigator.clipboard.writeText(trainJob.best_model || "");
                            setCopiedPath(true);
                            setTimeout(() => setCopiedPath(false), 1500);
                          } catch {
                            // 클립보드 권한 거부 시 무시
                          }
                        }}
                        className={cn(
                          "flex items-center gap-1 px-2.5 py-1.5 rounded text-[11px] font-medium border transition-colors shrink-0",
                          copiedPath
                            ? "bg-accent-dim border-accent/30 text-accent"
                            : "bg-background border-border text-muted hover:text-foreground hover:border-accent/50"
                        )}
                        title="경로 복사 (파라미터 페이지에 붙여넣기)"
                      >
                        {copiedPath ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                        {copiedPath ? "복사됨" : "복사"}
                      </button>
                      <button
                        onClick={async () => {
                          if (!trainJob.best_model) return;
                          setActivating(true);
                          setActivateMsg(null);
                          try {
                            const res = await apiFetch("/params/activate-model", {
                              method: "POST",
                              headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({ path: trainJob.best_model, kind: "pose" }),
                            });
                            if (!res.ok) {
                              const txt = await res.text().catch(() => "");
                              throw new Error(`HTTP ${res.status} ${txt.slice(0, 120)}`);
                            }
                            setTrainJob((prev) => ({ ...prev, activated: true }));
                            setActivateMsg({ type: "success", text: "활성 포즈 모델로 적용되었습니다." });
                          } catch (err) {
                            setActivateMsg({
                              type: "error",
                              text: `활성화 실패: ${err instanceof Error ? err.message : "오류"}`,
                            });
                          } finally {
                            setActivating(false);
                          }
                        }}
                        disabled={activating || !!trainJob.activated}
                        className={cn(
                          "flex items-center gap-1 px-3 py-1.5 rounded text-[11px] font-medium transition-colors shrink-0",
                          trainJob.activated
                            ? "bg-accent-dim text-accent border border-accent/30 cursor-default"
                            : "bg-accent text-background hover:opacity-90 disabled:opacity-50",
                        )}
                        title="이 best.pt를 즉시 활성 포즈 모델로 적용"
                      >
                        {activating ? (
                          <Loader2 className="w-3 h-3 animate-spin" />
                        ) : (
                          <Zap className="w-3 h-3" />
                        )}
                        {trainJob.activated ? "적용됨" : "활성 모델로 적용"}
                      </button>
                    </div>
                    {activateMsg && (
                      <p className={cn(
                        "text-[11px] mt-1.5",
                        activateMsg.type === "error" ? "text-danger" : "text-accent",
                      )}>
                        {activateMsg.text}
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
