"use client";

import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import {
  Upload, Play, Download, RotateCcw, CheckCircle2, AlertCircle, Loader2, Film, Info,
  Camera, ChevronDown, User, Save, Trash2, Gauge, Timer, Ruler, Activity,
} from "lucide-react";
import { apiFetch, apiUrl } from "@/lib/server-store";
import {
  ClimberProfile, loadProfiles, upsertProfile, deleteProfile, getLastProfileId, setLastProfileId,
  validBody, HEIGHT_RANGE, WINGSPAN_RANGE,
} from "@/lib/profiles";
import { cn } from "@/lib/utils";

type JobStatus = "idle" | "uploading" | "queued" | "processing" | "done" | "failed";

interface JobInfo {
  status: JobStatus;
  progress?: number;
  stage?: string;
  stage_detail?: string;
  queue_position?: number;
  error?: string;
}

interface Move { time: number; speed: number; direction: string }
interface Summary {
  climb_start: number;
  climb_end: number;
  duration: number;
  mean_speed: number;
  max_speed: number;
  px_per_m: number;
  scale_method: "calibration_pose" | "body_proportion";
  calibration_detected: boolean;
  calibration_time?: [number, number];
  moves: Move[];
  speed_series: (number | null)[];
  series_dt: number;
}

const NEW_PROFILE = "__new__";

export default function AnalyzePage() {
  // ── 클라이머 정보 ──
  const [profiles, setProfiles] = useState<ClimberProfile[]>([]);
  const [profileId, setProfileId] = useState<string>(NEW_PROFILE);
  const [name, setName] = useState("");
  const [height, setHeight] = useState("");
  const [wingspan, setWingspan] = useState("");
  const [savedMsg, setSavedMsg] = useState("");

  // ── 영상 / 작업 ──
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [jobId, setJobId] = useState("");
  const [jobInfo, setJobInfo] = useState<JobInfo>({ status: "idle" });
  const [resultUrl, setResultUrl] = useState("");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [showGuide, setShowGuide] = useState(true);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // 저장된 프로필 불러오기
  useEffect(() => {
    const list = loadProfiles();
    setProfiles(list);
    const last = getLastProfileId();
    const p = list.find((x) => x.id === last) || list[0];
    if (p) applyProfile(p);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const applyProfile = (p: ClimberProfile) => {
    setProfileId(p.id);
    setName(p.name);
    setHeight(String(p.heightCm));
    setWingspan(String(p.wingspanCm));
    setLastProfileId(p.id);
  };

  const handleProfileSelect = (id: string) => {
    setSavedMsg("");
    if (id === NEW_PROFILE) {
      setProfileId(NEW_PROFILE);
      setName(""); setHeight(""); setWingspan("");
      return;
    }
    const p = profiles.find((x) => x.id === id);
    if (p) applyProfile(p);
  };

  const h = parseFloat(height);
  const w = parseFloat(wingspan);
  const bodyError = height || wingspan ? validBody(h, w) : "키와 윙스팬을 입력하세요.";

  const handleSaveProfile = () => {
    if (bodyError || !name.trim()) return;
    const p = upsertProfile({ id: profileId === NEW_PROFILE ? undefined : profileId, name, heightCm: h, wingspanCm: w });
    setProfiles(loadProfiles());
    setProfileId(p.id);
    setLastProfileId(p.id);
    setSavedMsg("저장했어요");
    setTimeout(() => setSavedMsg(""), 1500);
  };

  const handleDeleteProfile = () => {
    if (profileId === NEW_PROFILE) return;
    deleteProfile(profileId);
    const list = loadProfiles();
    setProfiles(list);
    if (list[0]) applyProfile(list[0]);
    else handleProfileSelect(NEW_PROFILE);
  };

  // ── 영상 선택 ──
  const generateThumbnail = useCallback((videoFile: File) => {
    const video = document.createElement("video");
    video.muted = true;
    video.src = URL.createObjectURL(videoFile);
    video.onloadeddata = () => { video.currentTime = 0.5; };
    video.onseeked = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d")?.drawImage(video, 0, 0);
      setPreviewUrl(canvas.toDataURL("image/jpeg", 0.7));
      URL.revokeObjectURL(video.src);
    };
  }, []);

  const clearResult = () => {
    if (resultUrl.startsWith("blob:")) URL.revokeObjectURL(resultUrl);
    setResultUrl("");
    setSummary(null);
  };

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("video/")) return;
    setFile(f);
    setJobInfo({ status: "idle" });
    setJobId("");
    clearResult();
    generateThumbnail(f);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [generateThumbnail, resultUrl]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  // ── 결과 불러오기 ──
  const loadResult = useCallback(async (id: string) => {
    try {
      const sres = await apiFetch(`/summary/${id}`);
      if (sres.ok) setSummary((await sres.json()).summary);
    } catch { /* 요약 없이도 영상은 보여준다 */ }
    try {
      // ngrok 무료 플랜은 <video src> 직접 요청에 경고 페이지를 돌려주므로, 헤더를 붙여 받아서 재생
      const vres = await apiFetch(`/video/${id}`);
      if (!vres.ok) throw new Error();
      setResultUrl(URL.createObjectURL(await vres.blob()));
    } catch {
      setResultUrl(apiUrl(`/video/${id}`));
    }
  }, []);

  const pollStatus = useCallback((id: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const res = await apiFetch(`/status/${id}`);
        if (!res.ok) return;
        const data: JobInfo = await res.json();
        setJobInfo(data);
        if (data.status === "done") {
          clearInterval(pollRef.current!);
          loadResult(id);
        } else if (data.status === "failed") {
          clearInterval(pollRef.current!);
        }
      } catch {
        // 서버 연결 문제 - 폴링 계속
      }
    }, 1500);
  }, [loadResult]);

  const handleAnalyze = async () => {
    if (!file || bodyError) return;
    setJobInfo({ status: "uploading" });
    clearResult();

    const fd = new FormData();
    fd.append("file", file);
    fd.append("height_cm", String(h));
    fd.append("wingspan_cm", String(w));
    fd.append("climber_name", name.trim());

    try {
      const res = await apiFetch("/upload", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) {
        setJobInfo({ status: "failed", error: typeof data.detail === "string" ? data.detail : "업로드에 실패했습니다." });
        return;
      }
      setJobId(data.job_id);
      setJobInfo({ status: "queued" });
      pollStatus(data.job_id);
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
    clearResult();
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const seek = (t: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = Math.max(0, t - 0.5);
    v.play().catch(() => {});
  };

  const isRunning = ["uploading", "queued", "processing"].includes(jobInfo.status);
  const isDone = jobInfo.status === "done";
  const isFailed = jobInfo.status === "failed";
  const canAnalyze = !!file && !bodyError && !isRunning;

  const statusLabel: Record<JobStatus, string> = {
    idle: "분석 대기",
    uploading: "영상 업로드 중...",
    queued: "대기열에서 기다리는 중...",
    processing: "분석 중",
    done: "분석 완료",
    failed: "분석 실패",
  };

  return (
    <div className="p-4 sm:p-6 max-w-4xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-foreground text-balance">영상 분석</h1>
        <p className="mt-1 text-sm text-muted leading-relaxed">
          클라이머의 키와 윙스팬을 입력하고 영상을 올리면, 무게중심 궤적과 이동 속도(m/s)를 분석합니다.
        </p>
      </div>

      {/* 촬영 가이드 */}
      <div className="mb-4 rounded-lg border border-border bg-surface">
        <button
          onClick={() => setShowGuide((v) => !v)}
          className="w-full flex items-center gap-2 px-4 py-3 text-left"
        >
          <Camera className="w-4 h-4 text-accent shrink-0" />
          <span className="text-sm font-semibold text-foreground flex-1">촬영 가이드</span>
          <ChevronDown className={cn("w-4 h-4 text-muted transition-transform", showGuide && "rotate-180")} />
        </button>
        {showGuide && (
          <ol className="px-4 pb-4 space-y-2 text-sm text-muted leading-relaxed list-decimal list-inside">
            <li>
              <span className="text-foreground font-medium">보정 자세</span>: 등반 전, 출발 위치 근처에서 벽에 등을 붙이고
              카메라를 정면으로 보며 <span className="text-foreground">양팔을 어깨 높이로 쭉 펴고 2초</span> 멈춰 주세요.
              이 장면으로 실제 거리를 정확히 계산합니다. (없으면 신체 비율로 추정)
            </li>
            <li>폰은 <span className="text-foreground">삼각대에 고정</span>하고, 촬영 중에는 위치와 줌을 바꾸지 마세요.</li>
            <li>머리부터 발끝까지, 등반하는 전체 구간이 화면에 들어오게 찍어 주세요.</li>
            <li>가능하면 1080p로 촬영하세요. 30fps면 충분합니다.</li>
          </ol>
        )}
      </div>

      {/* 1. 클라이머 정보 */}
      <div className="mb-4 p-4 bg-surface rounded-lg border border-border">
        <div className="flex items-center gap-2 mb-3">
          <User className="w-4 h-4 text-muted" />
          <p className="text-xs font-semibold text-muted uppercase tracking-wider">1. 클라이머 정보</p>
        </div>

        <div className="flex flex-col sm:flex-row gap-2 mb-3">
          <select
            value={profileId}
            onChange={(e) => handleProfileSelect(e.target.value)}
            disabled={isRunning}
            className="flex-1 min-w-0 bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground"
          >
            <option value={NEW_PROFILE}>+ 새 클라이머</option>
            {profiles.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} · 키 {p.heightCm} · 윙스팬 {p.wingspanCm}
              </option>
            ))}
          </select>
          {profileId !== NEW_PROFILE && (
            <button
              onClick={handleDeleteProfile}
              disabled={isRunning}
              className="flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs text-muted bg-surface-2 border border-border hover:text-danger"
            >
              <Trash2 className="w-3.5 h-3.5" /> 삭제
            </button>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <label className="block">
            <span className="text-xs text-muted">이름</span>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="예: 김민수"
              maxLength={40}
              disabled={isRunning}
              className="mt-1 w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground"
            />
          </label>
          <label className="block">
            <span className="text-xs text-muted">키 (cm)</span>
            <input
              type="number" inputMode="decimal" min={HEIGHT_RANGE[0]} max={HEIGHT_RANGE[1]} step="0.1"
              value={height}
              onChange={(e) => setHeight(e.target.value)}
              placeholder="175"
              disabled={isRunning}
              className="mt-1 w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground"
            />
          </label>
          <label className="block">
            <span className="text-xs text-muted">윙스팬 (cm)</span>
            <input
              type="number" inputMode="decimal" min={WINGSPAN_RANGE[0]} max={WINGSPAN_RANGE[1]} step="0.1"
              value={wingspan}
              onChange={(e) => setWingspan(e.target.value)}
              placeholder="178"
              disabled={isRunning}
              className="mt-1 w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground"
            />
          </label>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button
            onClick={handleSaveProfile}
            disabled={!!bodyError || !name.trim() || isRunning}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border",
              bodyError || !name.trim()
                ? "text-muted border-border bg-surface-2 cursor-not-allowed"
                : "text-accent border-accent/30 bg-accent-dim hover:opacity-90"
            )}
          >
            <Save className="w-3.5 h-3.5" /> 프로필 저장
          </button>
          {savedMsg && <span className="text-xs text-accent">{savedMsg}</span>}
          {(height || wingspan) && bodyError && <span className="text-xs text-warning">{bodyError}</span>}
          <span className="text-xs text-muted">프로필은 이 브라우저에만 저장됩니다.</span>
        </div>
      </div>

      {/* 2. 영상 */}
      <p className="text-xs font-semibold text-muted uppercase tracking-wider mb-2">2. 영상</p>
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
              {!isRunning && (
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
              <p className="text-xs text-muted mt-1">MP4, MOV 등 영상 형식 지원</p>
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

      {/* 실행 버튼 */}
      <div className="mt-4 flex gap-3">
        <button
          onClick={handleAnalyze}
          disabled={!canAnalyze}
          className={cn(
            "flex items-center gap-2 px-5 py-2.5 rounded-lg font-semibold text-sm transition-all",
            !canAnalyze
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
      {file && bodyError && !isRunning && (
        <p className="mt-2 text-xs text-warning">분석하려면 클라이머의 키와 윙스팬을 먼저 입력하세요.</p>
      )}

      {/* 진행 상태 */}
      {jobInfo.status !== "idle" && (
        <div className={cn(
          "mt-4 p-4 rounded-lg border",
          isDone ? "bg-accent-dim border-accent/30" : isFailed ? "bg-red-950/30 border-danger/30" : "bg-surface border-border"
        )}>
          <div className="flex items-start gap-3">
            {isDone ? (
              <CheckCircle2 className="w-5 h-5 text-accent shrink-0 mt-0.5" />
            ) : isFailed ? (
              <AlertCircle className="w-5 h-5 text-danger shrink-0 mt-0.5" />
            ) : (
              <Loader2 className="w-5 h-5 text-warning animate-spin shrink-0 mt-0.5" />
            )}
            <div className="flex-1 min-w-0">
              <p className={cn("text-sm font-medium", isDone ? "text-accent" : isFailed ? "text-danger" : "text-foreground")}>
                {statusLabel[jobInfo.status]}
                {jobInfo.status === "processing" && jobInfo.stage ? ` · ${jobInfo.stage}` : ""}
                {jobInfo.status === "queued" && jobInfo.queue_position && jobInfo.queue_position > 1
                  ? ` (앞에 ${jobInfo.queue_position - 1}개)` : ""}
              </p>
              {jobId && <p className="text-xs text-muted mt-0.5 font-mono">Job ID: {jobId}</p>}
              {isFailed && jobInfo.error && (
                <p className="text-xs text-danger/80 mt-1 leading-relaxed">{jobInfo.error}</p>
              )}
            </div>
            {jobInfo.status === "processing" && (
              <span className="text-sm font-mono text-foreground tabular-nums">{Math.round(jobInfo.progress || 0)}%</span>
            )}
          </div>
          {isRunning && (
            <div className="mt-3 h-1.5 rounded-full bg-surface-2 overflow-hidden">
              <div
                className={cn("h-full bg-accent transition-all duration-500", jobInfo.status !== "processing" && "animate-pulse w-1/12")}
                style={jobInfo.status === "processing" ? { width: `${Math.max(2, jobInfo.progress || 0)}%` } : undefined}
              />
            </div>
          )}
        </div>
      )}

      {/* 결과 영상 */}
      {isDone && resultUrl && (
        <div className="mt-4 bg-surface rounded-lg border border-border overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center justify-between">
            <p className="text-sm font-semibold text-foreground">분석 결과 영상</p>
            <a
              href={resultUrl}
              download={`${name.trim() || "analyzed"}_${jobId.slice(0, 8)}.mp4`}
              className="flex items-center gap-1.5 text-xs text-accent hover:underline font-medium"
            >
              <Download className="w-3.5 h-3.5" />
              다운로드
            </a>
          </div>
          <video
            ref={videoRef}
            key={resultUrl}
            src={resultUrl}
            controls
            playsInline
            className="w-full max-h-[560px] bg-black"
            style={{ objectFit: "contain" }}
          />
        </div>
      )}

      {/* 분석 요약 */}
      {isDone && summary && <SummaryPanel summary={summary} onSeek={seek} />}

      <div className="mt-6 flex gap-2 p-3 bg-surface rounded-lg border border-border">
        <Info className="w-4 h-4 text-muted shrink-0 mt-0.5" />
        <p className="text-xs text-muted leading-relaxed">
          분석은 로컬 PC에서 실행되며, 영상 길이와 PC 성능에 따라 수 분이 걸릴 수 있습니다.
          여러 영상을 올리면 순서대로 하나씩 처리합니다.
        </p>
      </div>

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// 분석 요약: 지표 카드 + 속도 그래프 + 동작 목록 (누르면 해당 시점으로 이동)
// ────────────────────────────────────────────────────────────────────────────
function SummaryPanel({ summary: s, onSeek }: { summary: Summary; onSeek: (t: number) => void }) {
  const stats = [
    { icon: Timer, label: "등반 시간", value: `${s.duration.toFixed(1)}초` },
    { icon: Activity, label: "평균 COG 속도", value: `${s.mean_speed.toFixed(2)} m/s` },
    { icon: Gauge, label: "최고 COG 속도", value: `${s.max_speed.toFixed(2)} m/s` },
    {
      icon: Ruler,
      label: "거리 기준",
      value: s.calibration_detected ? "보정 자세" : "신체 비율",
      hint: s.calibration_detected
        ? `${s.calibration_time?.[0].toFixed(1)}~${s.calibration_time?.[1].toFixed(1)}초 장면 사용`
        : "보정 자세가 없어 추정값입니다",
    },
  ];

  return (
    <div className="mt-4 bg-surface rounded-lg border border-border">
      <div className="px-4 py-3 border-b border-border">
        <p className="text-sm font-semibold text-foreground">분석 요약</p>
      </div>

      <div className="p-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
        {stats.map(({ icon: Icon, label, value, hint }) => (
          <div key={label} className="p-3 rounded-lg bg-surface-2 border border-border">
            <div className="flex items-center gap-1.5 text-xs text-muted">
              <Icon className="w-3.5 h-3.5" /> {label}
            </div>
            <p className="mt-1 text-base font-semibold text-foreground tabular-nums">{value}</p>
            {hint && <p className="mt-0.5 text-[11px] text-muted leading-snug">{hint}</p>}
          </div>
        ))}
      </div>

      <div className="px-4">
        <p className="text-xs text-muted mb-1">COG 속도 (그래프를 누르면 그 시점으로 이동)</p>
        <SpeedChart summary={s} onSeek={onSeek} />
      </div>

      <div className="p-4">
        <p className="text-xs text-muted mb-2">동작별 최고 속도 · {s.moves.length}개</p>
        {s.moves.length === 0 ? (
          <p className="text-sm text-muted">뚜렷한 동작 구간을 찾지 못했어요.</p>
        ) : (
          <ul className="divide-y divide-border rounded-lg border border-border overflow-hidden">
            {s.moves.map((m, i) => (
              <li key={i}>
                <button
                  onClick={() => onSeek(m.time)}
                  className="w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-surface-2 transition-colors"
                >
                  <span className="w-14 text-xs font-mono text-muted tabular-nums">{m.time.toFixed(1)}s</span>
                  <span className="flex-1 text-sm text-foreground">{m.direction || "-"}</span>
                  <span className={cn("text-sm font-semibold tabular-nums", m.speed >= 1.0 ? "text-warning" : "text-foreground")}>
                    {m.speed.toFixed(2)} m/s
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function SpeedChart({ summary: s, onSeek }: { summary: Summary; onSeek: (t: number) => void }) {
  const W = 800, H = 140, PAD_L = 34, PAD_B = 18, PAD_T = 8;
  const vals = s.speed_series;
  const vmax = Math.max(1.5, ...vals.map((v) => v ?? 0)) * 1.05;
  const n = vals.length;
  const x = (i: number) => PAD_L + (i / Math.max(1, n - 1)) * (W - PAD_L - 6);
  const y = (v: number) => PAD_T + (1 - v / vmax) * (H - PAD_T - PAD_B);
  const t0 = s.climb_start;

  const path = useMemo(() => {
    let d = "";
    let pen = false;
    vals.forEach((v, i) => {
      if (v == null) { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    return d;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vals, vmax]);

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const i = Math.round(((px - PAD_L) / (W - PAD_L - 6)) * (n - 1));
    onSeek(t0 + Math.min(Math.max(i, 0), n - 1) * s.series_dt);
  };

  const ticks = [0.5, 1.0, 1.5, 2.0, 2.5].filter((v) => v < vmax);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto cursor-pointer select-none" onClick={handleClick} role="img" aria-label="COG 속도 그래프">
      {ticks.map((v) => (
        <g key={v}>
          <line x1={PAD_L} x2={W - 6} y1={y(v)} y2={y(v)} stroke="var(--color-border)" strokeWidth={1} />
          <text x={PAD_L - 6} y={y(v) + 3} textAnchor="end" fontSize={10} fill="var(--color-muted)">{v.toFixed(1)}</text>
        </g>
      ))}
      <line x1={PAD_L} x2={W - 6} y1={y(0)} y2={y(0)} stroke="var(--color-border)" strokeWidth={1} />
      <path d={path} fill="none" stroke="var(--color-accent)" strokeWidth={2} strokeLinejoin="round" />
      {s.moves.map((m, i) => {
        const idx = (m.time - t0) / s.series_dt;
        return <circle key={i} cx={x(idx)} cy={y(m.speed)} r={3.5} fill="var(--color-warning)" />;
      })}
      <text x={PAD_L} y={H - 4} fontSize={10} fill="var(--color-muted)">{t0.toFixed(1)}s</text>
      <text x={W - 6} y={H - 4} fontSize={10} fill="var(--color-muted)" textAnchor="end">{s.climb_end.toFixed(1)}s</text>
    </svg>
  );
}
