"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import {
  Upload, Play, Download, AlertCircle, CheckCircle2,
  Loader2, Image as ImageIcon, FileText, Trash2, Info, FolderOpen
} from "lucide-react";
import { apiUrl, apiFetch } from "@/lib/server-store";
import { cn } from "@/lib/utils";

type TrainStatus = "idle" | "uploading" | "queued" | "processing" | "done" | "failed";

interface TrainJob {
  status: TrainStatus;
  error?: string;
  progress?: number;
  log?: string[];
  best_model?: string;
}

interface UploadedFile {
  name: string;
  type: "image" | "label";
  size: number;
}

export default function TrainPage() {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [rawFiles, setRawFiles] = useState<File[]>([]);
  const [sessionId, setSessionId] = useState<string>("");
  const [trainJobId, setTrainJobId] = useState<string>("");
  const [trainJob, setTrainJob] = useState<TrainJob>({ status: "idle" });
  const [epochs, setEpochs] = useState(50);
  const [imgsz, setImgsz] = useState(640);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const logBottomRef = useRef<HTMLDivElement>(null);

  const imageCount = files.filter((f) => f.type === "image").length;
  const labelCount = files.filter((f) => f.type === "label").length;

  const isRunning = ["uploading", "queued", "processing"].includes(trainJob.status);
  const isDone = trainJob.status === "done";
  const isFailed = trainJob.status === "failed";

  const handleFiles = useCallback((incoming: File[]) => {
    const valid = incoming.filter((f) => {
      const ext = f.name.split(".").pop()?.toLowerCase() || "";
      return ["jpg", "jpeg", "png", "bmp", "webp", "txt"].includes(ext);
    });
    setRawFiles((prev) => {
      const existing = new Set(prev.map((f) => f.name));
      return [...prev, ...valid.filter((f) => !existing.has(f.name))];
    });
    setFiles((prev) => {
      const existing = new Set(prev.map((f) => f.name));
      const newEntries: UploadedFile[] = valid
        .filter((f) => !existing.has(f.name))
        .map((f) => {
          const ext = f.name.split(".").pop()?.toLowerCase() || "";
          return {
            name: f.name,
            type: ext === "txt" ? "label" : "image",
            size: f.size,
          };
        });
      return [...prev, ...newEntries];
    });
  }, []);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    handleFiles(Array.from(e.dataTransfer.files));
  };

  const handleRemove = (name: string) => {
    setFiles((prev) => prev.filter((f) => f.name !== name));
    setRawFiles((prev) => prev.filter((f) => f.name !== name));
  };

  const pollTrainStatus = useCallback((id: string) => {
    pollRef.current = setInterval(async () => {
      try {
        const res = await apiFetch(`/train/status/${id}`);
        const data: TrainJob = await res.json();
        setTrainJob(data);
        if (data.status === "done" || data.status === "failed") {
          clearInterval(pollRef.current!);
        }
      } catch {
        // 연결 문제 - 계속 폴링
      }
    }, 2000);
  }, []);

  // 로그 자동 스크롤
  useEffect(() => {
    logBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [trainJob.log]);

  const handleUploadAndTrain = async () => {
    if (rawFiles.length === 0) return;
    setTrainJob({ status: "uploading", progress: 0, log: [] });

    // 1. 파일 업로드
    const fd = new FormData();
    for (const f of rawFiles) fd.append("files", f);

    try {
      const uploadRes = await apiFetch("/train/upload", { method: "POST", body: fd });
      const uploadData = await uploadRes.json();
      const sid = uploadData.session_id;
      setSessionId(sid);
      setTrainJob((prev) => ({ ...prev, status: "queued", log: [`세션 생성됨: ${sid}`, `파일 ${uploadData.uploaded?.length ?? 0}개 업로드 완료`] }));

      // 2. 학습 시작
      const trainRes = await apiFetch(
        `/train/start/${sid}?epochs=${epochs}&imgsz=${imgsz}`,
        { method: "POST" }
      );
      const trainData = await trainRes.json();
      const tid = trainData.train_job_id;
      setTrainJobId(tid);
      setTrainJob((prev) => ({ ...prev, status: "processing", log: [...(prev.log || []), `학습 시작 (Job: ${tid})`] }));
      pollTrainStatus(tid);
    } catch {
      setTrainJob({ status: "failed", error: "로컬 서버에 연결할 수 없습니다." });
    }
  };

  const handleReset = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    setFiles([]);
    setRawFiles([]);
    setSessionId("");
    setTrainJobId("");
    setTrainJob({ status: "idle" });
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const statusLabel: Record<string, string> = {
    idle: "대기",
    uploading: "파일 업로드 중...",
    queued: "학습 준비 중...",
    processing: "학습 진행 중...",
    done: "학습 완료",
    failed: "학습 실패",
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-foreground text-balance">홀드 탐지 학습</h1>
        <p className="mt-1 text-sm text-muted leading-relaxed">
          클라이밍 벽 이미지와 YOLO 형식의 라벨(txt) 파일을 업로드하면 로컬 서버에서 YOLOv11 파인튜닝을 실행합니다.
        </p>
      </div>

      {/* File upload zone */}
      <div
        className={cn(
          "border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all",
          isDragOver ? "border-accent bg-accent-dim" : "border-border hover:border-accent/50 bg-surface"
        )}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
      >
        <div className="flex flex-col items-center gap-3">
          <div className="w-14 h-14 rounded-full bg-surface-2 flex items-center justify-center">
            <FolderOpen className="w-6 h-6 text-muted" />
          </div>
          <div>
            <p className="text-sm font-medium text-foreground">이미지 + 라벨 파일을 드래그하거나 클릭하여 선택</p>
            <p className="text-xs text-muted mt-1">이미지: JPG, PNG, BMP, WebP | 라벨: TXT (YOLO 포맷)</p>
          </div>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".jpg,.jpeg,.png,.bmp,.webp,.txt"
          className="hidden"
          onChange={(e) => { if (e.target.files) handleFiles(Array.from(e.target.files)); }}
        />
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="mt-4 bg-surface rounded-lg border border-border overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center justify-between">
            <div className="flex items-center gap-4">
              <span className="text-xs text-muted">
                <span className="text-foreground font-medium">{imageCount}</span>개 이미지
              </span>
              <span className="text-xs text-muted">
                <span className="text-foreground font-medium">{labelCount}</span>개 라벨
              </span>
            </div>
            <button onClick={handleReset} className="text-xs text-muted hover:text-danger transition-colors">전체 삭제</button>
          </div>
          <div className="max-h-52 overflow-y-auto">
            {files.map((f) => (
              <div key={f.name} className="flex items-center gap-3 px-4 py-2.5 border-b border-border/50 last:border-0">
                {f.type === "image" ? (
                  <ImageIcon className="w-4 h-4 text-accent shrink-0" />
                ) : (
                  <FileText className="w-4 h-4 text-warning shrink-0" />
                )}
                <span className="flex-1 text-xs text-foreground truncate">{f.name}</span>
                <span className="text-xs text-muted font-mono shrink-0">{(f.size / 1024).toFixed(0)}KB</span>
                <button
                  onClick={() => handleRemove(f.name)}
                  className="text-muted hover:text-danger transition-colors shrink-0"
                  disabled={isRunning}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Training settings */}
      <div className="mt-4 p-4 bg-surface rounded-lg border border-border">
        <p className="text-xs font-semibold text-muted uppercase tracking-wider mb-4">학습 설정</p>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-muted mb-1.5">Epochs (학습 반복 횟수)</label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min={5} max={200} step={5}
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                disabled={isRunning}
                className="flex-1"
              />
              <span className="text-sm font-mono text-foreground w-10 text-right">{epochs}</span>
            </div>
          </div>
          <div>
            <label className="block text-xs text-muted mb-1.5">Image Size (px)</label>
            <div className="flex gap-2">
              {[320, 416, 640, 832].map((sz) => (
                <button
                  key={sz}
                  onClick={() => setImgsz(sz)}
                  disabled={isRunning}
                  className={cn(
                    "flex-1 py-1.5 rounded text-xs font-mono font-medium transition-all",
                    imgsz === sz
                      ? "bg-accent text-background"
                      : "bg-surface-2 text-muted border border-border hover:text-foreground"
                  )}
                >
                  {sz}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Action */}
      <div className="mt-4 flex gap-3">
        <button
          onClick={handleUploadAndTrain}
          disabled={files.length === 0 || isRunning}
          className={cn(
            "flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold transition-all",
            files.length === 0 || isRunning
              ? "bg-surface-2 text-muted cursor-not-allowed border border-border"
              : "bg-accent text-background hover:opacity-90"
          )}
        >
          {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          업로드 및 학습 시작
        </button>
      </div>

      {/* Status + progress */}
      {trainJob.status !== "idle" && (
        <div className={cn(
          "mt-4 rounded-lg border overflow-hidden",
          isDone ? "border-accent/30" : isFailed ? "border-danger/30" : "border-border"
        )}>
          <div className={cn(
            "px-4 py-3 flex items-center gap-3",
            isDone ? "bg-accent-dim" : isFailed ? "bg-red-950/30" : "bg-surface"
          )}>
            {isDone ? (
              <CheckCircle2 className="w-4 h-4 text-accent shrink-0" />
            ) : isFailed ? (
              <AlertCircle className="w-4 h-4 text-danger shrink-0" />
            ) : (
              <Loader2 className="w-4 h-4 text-warning animate-spin shrink-0" />
            )}
            <div className="flex-1">
              <span className={cn(
                "text-sm font-medium",
                isDone ? "text-accent" : isFailed ? "text-danger" : "text-foreground"
              )}>
                {statusLabel[trainJob.status]}
              </span>
              {trainJob.progress !== undefined && !isDone && !isFailed && (
                <span className="text-xs text-muted ml-2">{trainJob.progress}%</span>
              )}
            </div>
            {isDone && trainJobId && (
              <a
                href={apiUrl(`/train/download/${trainJobId}`)}
                className="flex items-center gap-1.5 text-xs text-accent hover:underline font-medium"
              >
                <Download className="w-3.5 h-3.5" />
                best.pt 다운로드
              </a>
            )}
          </div>

          {/* Progress bar */}
          {!isDone && !isFailed && trainJob.progress !== undefined && (
            <div className="h-1 bg-surface-2">
              <div
                className="h-full bg-accent transition-all duration-500"
                style={{ width: `${trainJob.progress}%` }}
              />
            </div>
          )}

          {/* Log */}
          {(trainJob.log?.length ?? 0) > 0 && (
            <div className="bg-black/40 p-3 max-h-52 overflow-y-auto font-mono text-xs text-muted">
              {trainJob.log?.map((line, i) => (
                <div key={i} className="py-0.5">{line}</div>
              ))}
              <div ref={logBottomRef} />
            </div>
          )}

          {isFailed && trainJob.error && (
            <div className="px-4 py-3 text-xs text-danger/80 bg-red-950/20">
              {trainJob.error}
            </div>
          )}
        </div>
      )}

      {/* Label format guide */}
      <div className="mt-6 p-4 bg-surface rounded-lg border border-border">
        <div className="flex gap-2 mb-3">
          <Info className="w-4 h-4 text-muted shrink-0 mt-0.5" />
          <p className="text-xs font-semibold text-foreground">YOLO 라벨 포맷 안내</p>
        </div>
        <div className="ml-6 space-y-2">
          <p className="text-xs text-muted leading-relaxed">
            각 이미지와 동일한 파일명의 <span className="text-foreground font-mono">.txt</span> 라벨 파일이 필요합니다.
          </p>
          <div className="bg-surface-2 rounded p-3 font-mono text-xs text-foreground">
            <p className="text-muted mb-1"># 형식: class_id  cx  cy  w  h (모두 0~1 상대좌표)</p>
            <p>0 0.512 0.334 0.089 0.124</p>
            <p>0 0.231 0.667 0.072 0.098</p>
          </div>
          <p className="text-xs text-muted">
            class_id는 홀드(0) 하나만 사용합니다. LabelImg, Roboflow 등의 도구로 라벨링 후 YOLO 포맷으로 내보내면 됩니다.
          </p>
        </div>
      </div>
    </div>
  );
}
