"use client";

import { useState, useEffect } from "react";
import { Save, RotateCcw, CheckCircle2, AlertCircle, Loader2, Info, FolderOpen } from "lucide-react";
import useSWR from "swr";
import { apiUrl, apiFetch, swrFetcher } from "@/lib/server-store";
import { cn } from "@/lib/utils";

interface Params {
  occlusion_threshold: number;
  occlusion_search_radius: number;
  smoothing_alpha: number;
  hold_conf: number;
  pose_conf: number;
  arm_extension: number;
  leg_extension: number;
  hold_model_path: string;
  snap_only_body_occluded: boolean;
  body_occlusion_overlap_threshold: number;
}

const DEFAULT_PARAMS: Params = {
  occlusion_threshold: 0.3,
  occlusion_search_radius: 150,
  smoothing_alpha: 0.5,
  hold_conf: 0.1,
  pose_conf: 0.35,
  arm_extension: 0.15,
  leg_extension: 0.10,
  hold_model_path: "",
  snap_only_body_occluded: true,
  body_occlusion_overlap_threshold: 0.4,
};

type SaveStatus = "idle" | "saving" | "saved" | "error";

interface SliderFieldProps {
  label: string;
  description: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format?: (v: number) => string;
  onChange: (v: number) => void;
}

function SliderField({ label, description, value, min, max, step, format, onChange }: SliderFieldProps) {
  const display = format ? format(value) : value.toFixed(2);
  return (
    <div className="py-4 border-b border-border last:border-0">
      <div className="flex items-start justify-between mb-2">
        <div>
          <p className="text-sm font-medium text-foreground">{label}</p>
          <p className="text-xs text-muted mt-0.5 leading-relaxed">{description}</p>
        </div>
        <span className="text-sm font-mono text-accent bg-accent-dim px-2 py-0.5 rounded ml-4 shrink-0">
          {display}
        </span>
      </div>
      <div className="flex items-center gap-3">
        <span className="text-xs text-muted font-mono w-8 shrink-0">{format ? format(min) : min}</span>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="flex-1"
        />
        <span className="text-xs text-muted font-mono w-8 shrink-0 text-right">{format ? format(max) : max}</span>
      </div>
    </div>
  );
}

export default function ParamsPage() {
  const [paramsUrl, setParamsUrl] = useState<string | null>(null);

  useEffect(() => {
    setParamsUrl(apiUrl("/params"));
  }, []);

  const { data: serverParams, isLoading, mutate } = useSWR<Params>(
    paramsUrl,
    swrFetcher,
    { revalidateOnFocus: false }
  );

  const [params, setParams] = useState<Params>(DEFAULT_PARAMS);
  const [saveStatus, setSaveStatus] = useState<SaveStatus>("idle");

  useEffect(() => {
    if (serverParams) setParams(serverParams);
  }, [serverParams]);

  const set = (key: keyof Params, val: number | string | boolean) => {
    setParams((prev) => ({ ...prev, [key]: val }));
    setSaveStatus("idle");
  };

  const handleSave = async () => {
    setSaveStatus("saving");
    try {
      const res = await apiFetch("/params", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });
      if (res.ok) {
        setSaveStatus("saved");
        mutate();
        setTimeout(() => setSaveStatus("idle"), 2500);
      } else {
        setSaveStatus("error");
      }
    } catch {
      setSaveStatus("error");
    }
  };

  const handleReset = () => {
    setParams(serverParams || DEFAULT_PARAMS);
    setSaveStatus("idle");
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-6 h-6 text-muted animate-spin" />
        <span className="ml-2 text-sm text-muted">서버에서 파라미터 불러오는 중...</span>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-3xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-foreground text-balance">파라미터 튜닝</h1>
        <p className="mt-1 text-sm text-muted leading-relaxed">
          관절 폐색 보정, 좌표 스무딩, 홀드/포즈 탐지 신뢰도 등 분석 파라미터를 실시간으로 조정하고 저장합니다.
          변경사항은 다음 영상 분석부터 즉시 적용됩니다.
        </p>
      </div>

      {/* Occlusion section */}
      <div className="bg-surface rounded-lg border border-border mb-4 overflow-hidden">
        <div className="px-5 py-3 border-b border-border bg-surface-2">
          <p className="text-xs font-semibold text-muted uppercase tracking-wider">관절 폐색 보정</p>
        </div>
        <div className="px-5">
          <SliderField
            label="폐색 임계값 (Occlusion Threshold)"
            description="이 신뢰도 미만의 관절을 폐색으로 판단합니다. 낮을수록 보정이 드물게, 높을수록 보정이 자주 일어납니다."
            value={params.occlusion_threshold}
            min={0.05} max={0.8} step={0.01}
            onChange={(v) => set("occlusion_threshold", v)}
          />
          <SliderField
            label="홀드 탐색 반경 (Search Radius, px)"
            description="폐색된 관절에서 이전 프레임 위치 기준으로 이 반경 내의 홀드를 보정 좌표 후보로 탐색합니다."
            value={params.occlusion_search_radius}
            min={30} max={400} step={10}
            format={(v) => `${v}px`}
            onChange={(v) => set("occlusion_search_radius", v)}
          />
        </div>
      </div>

      {/* Wrist hold-snap mode */}
      <div className="bg-surface rounded-lg border border-border mb-4 overflow-hidden">
        <div className="px-5 py-3 border-b border-border bg-surface-2">
          <p className="text-xs font-semibold text-muted uppercase tracking-wider">손목 hold-snap 모드</p>
        </div>
        <div className="px-5 py-4 space-y-3">
          <label className="flex items-start gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={params.snap_only_body_occluded}
              onChange={(e) => set("snap_only_body_occluded", e.target.checked)}
              className="accent-accent mt-0.5"
            />
            <div className="flex-1">
              <span className="text-sm font-medium text-foreground">
                몸에 가려진 홀드만 snap 후보로 사용 (권장)
              </span>
              <p className="text-[11px] text-muted mt-0.5 leading-relaxed">
                손목이 가려져 위치 추정이 필요할 때, 클라이머의 어깨-엉덩이 4각형 영역과
                겹치는 홀드만 후보로 사용합니다. 잘 보이는 홀드로 손이 잘못 점프하는
                현상을 방지합니다. 끄면 화면의 모든 홀드가 후보가 됩니다(이전 동작).
              </p>
            </div>
          </label>

          <div className={cn("transition-opacity", params.snap_only_body_occluded ? "opacity-100" : "opacity-40 pointer-events-none")}>
            <SliderField
              label="몸 겹침 비율 임계값 (Body Overlap Threshold)"
              description="홀드 박스가 몸 4각형에 이 비율 이상 들어있으면 가려진 것으로 봅니다. 낮추면 더 관대하게(주변 홀드도 후보), 높이면 더 엄격하게(완전히 몸 뒤의 홀드만) 동작합니다."
              value={params.body_occlusion_overlap_threshold}
              min={0.1}
              max={1.0}
              step={0.05}
              format={(v) => `${Math.round(v * 100)}%`}
              onChange={(v) => set("body_occlusion_overlap_threshold", v)}
            />
          </div>
        </div>
      </div>

      {/* Smoothing section */}
      <div className="bg-surface rounded-lg border border-border mb-4 overflow-hidden">
        <div className="px-5 py-3 border-b border-border bg-surface-2">
          <p className="text-xs font-semibold text-muted uppercase tracking-wider">좌표 스무딩</p>
        </div>
        <div className="px-5">
          <SliderField
            label="스무딩 알파 (Smoothing Alpha)"
            description="현재 프레임 좌표와 이전 프레임 좌표를 혼합하는 비율. 낮을수록 부드럽고, 높을수록 반응이 빠릅니다."
            value={params.smoothing_alpha}
            min={0.1} max={1.0} step={0.05}
            onChange={(v) => set("smoothing_alpha", v)}
          />
        </div>
      </div>

      {/* Detection section */}
      <div className="bg-surface rounded-lg border border-border mb-4 overflow-hidden">
        <div className="px-5 py-3 border-b border-border bg-surface-2">
          <p className="text-xs font-semibold text-muted uppercase tracking-wider">탐지 신뢰도</p>
        </div>
        <div className="px-5">
          <SliderField
            label="홀드 탐지 신뢰도 (Hold Conf)"
            description="홀드 탐지 모델의 최소 신뢰도. 낮을수록 더 많은 홀드를 탐지하지만 오탐이 늘어납니다."
            value={params.hold_conf}
            min={0.01} max={0.7} step={0.01}
            onChange={(v) => set("hold_conf", v)}
          />
          <SliderField
            label="포즈 탐지 신뢰도 (Pose Conf)"
            description="YOLO 포즈 모델의 최소 신뢰도. 높을수록 명확한 자세에서만 탐지합니다."
            value={params.pose_conf}
            min={0.1} max={0.8} step={0.01}
            onChange={(v) => set("pose_conf", v)}
          />
        </div>
      </div>

      {/* Limb extension section */}
      <div className="bg-surface rounded-lg border border-border mb-4 overflow-hidden">
        <div className="px-5 py-3 border-b border-border bg-surface-2">
          <p className="text-xs font-semibold text-muted uppercase tracking-wider">팔다리 끝 연장</p>
        </div>
        <div className="px-5">
          <SliderField
            label="팔(손목) 연장 비율 (Arm Extension)"
            description="손목 관절을 팔꿈치 방향으로 연장하는 비율. 0이면 연장 없음."
            value={params.arm_extension}
            min={0} max={0.5} step={0.01}
            format={(v) => `×${v.toFixed(2)}`}
            onChange={(v) => set("arm_extension", v)}
          />
          <SliderField
            label="다리(발목) 연장 비율 (Leg Extension)"
            description="발목 관절을 무릎 방향으로 연장하는 비율. 0이면 연장 없음."
            value={params.leg_extension}
            min={0} max={0.5} step={0.01}
            format={(v) => `×${v.toFixed(2)}`}
            onChange={(v) => set("leg_extension", v)}
          />
        </div>
      </div>

      {/* Hold model path */}
      <div className="bg-surface rounded-lg border border-border mb-4 overflow-hidden">
        <div className="px-5 py-3 border-b border-border bg-surface-2">
          <p className="text-xs font-semibold text-muted uppercase tracking-wider">홀드 탐지 모델 경로</p>
        </div>
        <div className="px-5 py-4">
          <p className="text-xs text-muted mb-2 leading-relaxed">
            로컬 PC의 학습된 best.pt 파일 경로를 입력하세요. 저장 시 즉시 해당 모델을 로드합니다.
          </p>
          <div className="flex items-center gap-2">
            <FolderOpen className="w-4 h-4 text-muted shrink-0" />
            <input
              type="text"
              value={params.hold_model_path}
              onChange={(e) => set("hold_model_path", e.target.value)}
              placeholder="C:/Project/runs/detect/train/weights/best.pt"
              className="flex-1 bg-surface-2 border border-border rounded-lg text-xs px-3 py-2 font-mono text-foreground outline-none focus:border-accent transition-colors"
            />
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-3 mt-2">
        <button
          onClick={handleSave}
          disabled={saveStatus === "saving"}
          className={cn(
            "flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold transition-all",
            saveStatus === "saving"
              ? "bg-surface-2 text-muted cursor-not-allowed border border-border"
              : "bg-accent text-background hover:opacity-90"
          )}
        >
          {saveStatus === "saving" ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Save className="w-4 h-4" />
          )}
          저장 및 적용
        </button>
        <button
          onClick={handleReset}
          className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm text-muted bg-surface-2 border border-border hover:text-foreground transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
          변경 취소
        </button>

        {saveStatus === "saved" && (
          <div className="flex items-center gap-1.5 text-xs text-accent">
            <CheckCircle2 className="w-4 h-4" />
            저장 완료. 다음 분석부터 적용됩니다.
          </div>
        )}
        {saveStatus === "error" && (
          <div className="flex items-center gap-1.5 text-xs text-danger">
            <AlertCircle className="w-4 h-4" />
            저장 실패. 서버 연결을 확인하세요.
          </div>
        )}
      </div>

      {/* Info */}
      <div className="mt-6 flex gap-2 p-3 bg-surface rounded-lg border border-border">
        <Info className="w-4 h-4 text-muted shrink-0 mt-0.5" />
        <p className="text-xs text-muted leading-relaxed">
          파라미터는 로컬 서버의 <span className="font-mono text-foreground">params.json</span>에 저장됩니다.
          서버를 재시작해도 저장된 값이 유지됩니다.
          기본값으로 되돌리려면 로컬 서버의 params.json 파일을 삭제하세요.
        </p>
      </div>
    </div>
  );
}
