"use client";

import { useState, useRef, useCallback } from "react";
import {
  Upload, Play, Download, RotateCcw, CheckCircle2,
  AlertCircle, Loader2, Film, Info
} from "lucide-react";
import { apiUrl, apiFetch } from "@/lib/server-store";
import { cn } from "@/lib/utils";

type DisplayOption = "no_ears" | "tips";
type JobStatus = "idle" | "uploading" | "queued" | "processing" | "done" | "failed";

interface JobInfo {
  status: JobStatus;
  error?: string;
  output?: string;
}

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [displayOption, setDisplayOption] = useState<DisplayOption>("no_ears");
  const [jobId, setJobId] = useState<string>("");
  const [jobInfo, setJobInfo] = useState<JobInfo>({ status: "idle" });
  const [resultUrl, setResultUrl] = useState<string>("");
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const generateThumbnail = useCallback((videoFile: File) => {
    const video = document.createElement("video");
    video.crossOrigin = "anonymous";
    video.src = URL.createObjectURL(videoFile);
    video.onloadeddata = () => { video.currentTime = 0.5; };
    video.onseeked = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d")?.drawImage(video, 0, 0);
      setPreviewUrl(canvas.toDataURL());
    };
  }, []);

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("video/")) return;
    setFile(f);
    setJobInfo({ status: "idle" });
    setResultUrl("");
    setJobId("");
    generateThumbnail(f);
  }, [generateThumbnail]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const pollStatus = useCallback((id: string) => {
    pollRef.current = setInterval(async () => {
      try {
        const res = await apiFetch(`/status/${id}`);
        const data: JobInfo = await res.json();
        setJobInfo(data);
        if (data.status === "done") {
          clearInterval(pollRef.current!);
          setResultUrl(apiUrl(`/video/${id}`));
        } else if (data.status === "failed") {
          clearInterval(pollRef.current!);
        }
      } catch {
        // 서버 연결 문제 - 폴링 계속
      }
    }, 1500);
  }, []);

  const handleAnalyze = async () => {
    if (!file) return;
    setJobInfo({ status: "uploading" });
    setResultUrl("");

    const fd = new FormData();
    fd.append("file", file);
    fd.append("display_option", displayOption);

    try {
      const res = await apiFetch("/upload", { method: "POST", body: fd });
      const data = await res.json();
      const id = data.job_id;
      setJobId(id);
      setJobInfo({ status: "queued" });
      pollStatus(id);
    } catch {
      setJobInfo({ status: "failed", error: "로컬 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요." });
    }
  };

  const handleReset = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    setFile(null);
    setPreviewUrl("");
    setJobId("");
    setJobInfo({ status: "idle" });
    setResultUrl("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const isRunning = ["uploading", "queued", "processing"].includes(jobInfo.status);
  const isDone = jobInfo.status === "done";
  const isFailed = jobInfo.status === "failed";

  const statusLabel: Record<string, string> = {
    idle: "분석 대기",
    uploading: "영상 업로드 중...",
    queued: "대기열 진입...",
    processing: "분석 처리 중...",
    done: "분석 완료",
    failed: "분석 실패",
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-foreground text-balance">영상 분석</h1>
        <p className="mt-1 text-sm text-muted leading-relaxed">
          클라이밍 영상을 업로드하면 로컬 서버에서 YOLO 포즈 감지 및 무게중심 분석을 실행합니다.
        </p>
      </div>

      {/* Drop zone */}
      <div
        className={cn(
          "relative rounded-lg border-2 border-dashed transition-all cursor-pointer",
          isDragOver ? "border-accent bg-accent-dim" : "border-border hover:border-accent/50",
          file ? "bg-surface p-4" : "bg-surface p-10"
        )}
        onClick={() => !file && fileInputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
      >
        {file ? (
          <div className="flex items-start gap-4">
            {previewUrl ? (
              <img src={previewUrl} alt="영상 썸네일" className="w-32 h-20 object-cover rounded-lg shrink-0 bg-surface-2" />
            ) : (
              <div className="w-32 h-20 rounded-lg bg-surface-2 flex items-center justify-center shrink-0">
                <Film className="w-6 h-6 text-muted" />
              </div>
            )}
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-foreground truncate">{file.name}</p>
              <p className="text-xs text-muted mt-0.5">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
              {!isRunning && !isDone && (
                <button
                  onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
                  className="mt-2 text-xs text-accent hover:underline"
                >
                  다른 파일 선택
                </button>
              )}
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 text-center">
            <div className="w-14 h-14 rounded-full bg-surface-2 flex items-center justify-center">
              <Upload className="w-6 h-6 text-muted" />
            </div>
            <div>
              <p className="text-sm font-medium text-foreground">영상 파일을 드래그하거나 클릭하여 선택</p>
              <p className="text-xs text-muted mt-1">MP4, MOV, AVI 등 영상 형식 지원</p>
            </div>
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => { if (e.target.files?.[0]) handleFile(e.target.files[0]); }}
        />
      </div>

      {/* Options */}
      <div className="mt-4 p-4 bg-surface rounded-lg border border-border">
        <p className="text-xs font-semibold text-muted uppercase tracking-wider mb-3">분석 모드</p>
        <div className="flex flex-col sm:flex-row gap-3">
          {[
            { value: "no_ears" as DisplayOption, label: "전체 관절", desc: "코·귀 제외한 전체 포즈 시각화" },
            { value: "tips" as DisplayOption, label: "손끝·발끝만", desc: "손목·발목 4개 관절만 표시" },
          ].map((opt) => (
            <label
              key={opt.value}
              className={cn(
                "flex-1 flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all",
                displayOption === opt.value
                  ? "border-accent bg-accent-dim"
                  : "border-border bg-surface-2 hover:border-accent/40"
              )}
            >
              <input
                type="radio"
                name="display_option"
                value={opt.value}
                checked={displayOption === opt.value}
                onChange={() => setDisplayOption(opt.value)}
                className="mt-0.5 accent-[var(--color-accent)]"
                disabled={isRunning}
              />
              <div>
                <span className={cn("text-sm font-medium", displayOption === opt.value ? "text-accent" : "text-foreground")}>
                  {opt.label}
                </span>
                <p className="text-xs text-muted mt-0.5 leading-relaxed">{opt.desc}</p>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Action buttons */}
      <div className="mt-4 flex gap-3">
        <button
          onClick={handleAnalyze}
          disabled={!file || isRunning}
          className={cn(
            "flex items-center gap-2 px-5 py-2.5 rounded-lg font-semibold text-sm transition-all",
            !file || isRunning
              ? "bg-surface-2 text-muted cursor-not-allowed border border-border"
              : "bg-accent text-background hover:opacity-90"
          )}
        >
          {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          분석 시작
        </button>
        <button
          onClick={handleReset}
          className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm text-muted bg-surface-2 border border-border hover:text-foreground transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
          초기화
        </button>
      </div>

      {/* Status */}
      {jobInfo.status !== "idle" && (
        <div className={cn(
          "mt-4 p-4 rounded-lg border flex items-start gap-3",
          isDone ? "bg-accent-dim border-accent/30"
            : isFailed ? "bg-red-950/30 border-danger/30"
              : "bg-surface border-border"
        )}>
          {isDone ? (
            <CheckCircle2 className="w-5 h-5 text-accent shrink-0 mt-0.5" />
          ) : isFailed ? (
            <AlertCircle className="w-5 h-5 text-danger shrink-0 mt-0.5" />
          ) : (
            <Loader2 className="w-5 h-5 text-warning animate-spin shrink-0 mt-0.5" />
          )}
          <div className="flex-1 min-w-0">
            <p className={cn(
              "text-sm font-medium",
              isDone ? "text-accent" : isFailed ? "text-danger" : "text-foreground"
            )}>
              {statusLabel[jobInfo.status]}
            </p>
            {jobId && <p className="text-xs text-muted mt-0.5 font-mono">Job ID: {jobId}</p>}
            {isFailed && jobInfo.error && (
              <p className="text-xs text-danger/80 mt-1 leading-relaxed">{jobInfo.error}</p>
            )}
          </div>

          {/* Progress bar for running */}
          {isRunning && (
            <div className="w-full absolute left-0 bottom-0 h-0.5 rounded-full bg-surface-2 overflow-hidden">
              <div className="h-full bg-accent/60 animate-pulse w-2/3" />
            </div>
          )}
        </div>
      )}

      {/* Result video */}
      {isDone && resultUrl && (
        <div className="mt-4 bg-surface rounded-lg border border-border overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center justify-between">
            <p className="text-sm font-semibold text-foreground">분석 결과 영상</p>
            <a
              href={resultUrl}
              download={`analyzed_${jobId}.mp4`}
              className="flex items-center gap-1.5 text-xs text-accent hover:underline font-medium"
            >
              <Download className="w-3.5 h-3.5" />
              다��로드
            </a>
          </div>
          <video
            key={resultUrl}
            controls
            className="w-full max-h-[480px] bg-black"
            style={{ objectFit: "contain" }}
          >
            <source src={resultUrl} type="video/mp4" />
          </video>
        </div>
      )}

      {/* Tip */}
      <div className="mt-6 flex gap-2 p-3 bg-surface rounded-lg border border-border">
        <Info className="w-4 h-4 text-muted shrink-0 mt-0.5" />
        <p className="text-xs text-muted leading-relaxed">
          분석에는 영상 길이 및 로컬 PC 성능에 따라 수 분이 소요될 수 있습니다.
          상단 서버 연결 바에서 로컬 서버 주소(기본값: http://localhost:8000)를 확인하세요.
        </p>
      </div>

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}
