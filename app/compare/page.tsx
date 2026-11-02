"use client";

import { useRef, useState } from "react";
import useSWR from "swr";
import { Play, Pause, SkipBack, ChevronLeft, ChevronRight } from "lucide-react";
import ZoomableVideo, { ZoomableVideoHandle } from "@/components/zoomable-video";
import { apiUrl, swrFetcher } from "@/lib/server-store";
import { cn } from "@/lib/utils";

interface VideoItem {
  name: string;
  url: string;
  job_id?: string;
}

const STEP = 0.1; // 한 번에 이동하는 시간 (초)
const fmt = (t: number) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;

export default function ComparePage() {
  const { data, isLoading, error } = useSWR<{ videos: VideoItem[] }>(apiUrl("/videos"), swrFetcher, { refreshInterval: 5000 });
  const videos = data?.videos || [];
  const keyOf = (v: VideoItem) => v.job_id || v.url;

  const [selA, setSelA] = useState("");
  const [selB, setSelB] = useState("");
  const [speed, setSpeed] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [durA, setDurA] = useState(0);
  const [durB, setDurB] = useState(0);

  const refA = useRef<ZoomableVideoHandle>(null);
  const refB = useRef<ZoomableVideoHandle>(null);
  const both = (f: (r: ZoomableVideoHandle) => void) => { [refA, refB].forEach((r) => r.current && f(r.current)); };

  const pathOf = (key: string) => videos.find((v) => keyOf(v) === key)?.url || "";
  const duration = Math.max(durA, durB);
  const hasVideo = !!(selA || selB);

  const select = (side: "A" | "B", key: string) => {
    both((r) => r.pause());
    setIsPlaying(false);
    setTime(0);
    if (side === "A") { setSelA(key); setDurA(0); } else { setSelB(key); setDurB(0); }
  };

  const play = () => { both((r) => { r.setPlaybackRate(speed); r.play(); }); setIsPlaying(true); };
  const pause = () => { both((r) => r.pause()); setIsPlaying(false); };
  const seekAll = (t: number) => { both((r) => r.seek(t)); setTime(t); };
  const step = (d: number) => { pause(); seekAll(Math.min(Math.max(0, time + d), duration)); };
  const changeSpeed = (r: number) => { setSpeed(r); both((v) => v.setPlaybackRate(r)); };

  // 더 긴 영상의 시간을 공통 재생바에 표시
  const onTime = (side: "A" | "B") => (t: number, d: number) => {
    if (side === "A") setDurA(d); else setDurB(d);
    const leader = durA >= durB ? "A" : "B";
    if (side === leader || !(selA && selB)) {
      setTime(t);
      if (d && t >= d - 0.05) setIsPlaying(false);
    }
  };

  // 컴포넌트가 아닌 함수로 그려야 재생 중 다시 그려져도 선택 상자가 닫히지 않는다
  const picker = (side: "A" | "B", value: string) => (
    <label className="block min-w-0">
      <span className="text-xs font-semibold text-muted uppercase tracking-wider">{side} 영상</span>
      <select
        value={value}
        onChange={(e) => select(side, e.target.value)}
        className="mt-1 w-full bg-surface-2 border border-border rounded-lg px-2 py-2 text-sm text-foreground truncate"
      >
        <option value="">{isLoading ? "불러오는 중..." : videos.length ? "선택하세요" : "분석된 영상 없음"}</option>
        {videos.map((v) => (
          <option key={keyOf(v)} value={keyOf(v)}>{v.name}</option>
        ))}
      </select>
    </label>
  );

  return (
    <div className="p-3 sm:p-6 max-w-5xl mx-auto">
      <div className="mb-4">
        <h1 className="text-2xl font-bold text-foreground text-balance">영상 비교</h1>
        <p className="mt-1 text-sm text-muted leading-relaxed">
          분석한 두 영상을 나란히 놓고 동시에 재생·정지·이동하며 비교합니다.
        </p>
        {error && <p className="mt-2 text-xs text-danger">영상 목록을 불러오지 못했어요. 서버 연결을 확인해 주세요.</p>}
      </div>

      <div className="grid grid-cols-2 gap-3 mb-3">
        {picker("A", selA)}
        {picker("B", selB)}
      </div>

      {/* 공통 조작 */}
      <div className="mb-3 p-3 bg-surface rounded-lg border border-border space-y-3">
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-muted tabular-nums w-12">{fmt(time)}</span>
          <input
            type="range" min={0} max={duration || 0} step={0.05} value={Math.min(time, duration || 0)}
            disabled={!hasVideo || !duration}
            onChange={(e) => { pause(); seekAll(parseFloat(e.target.value)); }}
            className="flex-1 accent-[var(--color-accent)]"
            aria-label="재생 위치"
          />
          <span className="text-xs font-mono text-muted tabular-nums w-12 text-right">{fmt(duration)}</span>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-1">
            <button onClick={() => { pause(); seekAll(0); }} disabled={!hasVideo} aria-label="처음으로"
              className="p-2 rounded-lg bg-surface-2 border border-border text-muted hover:text-foreground disabled:opacity-40">
              <SkipBack className="w-4 h-4" />
            </button>
            <button onClick={() => step(-STEP)} disabled={!hasVideo} aria-label="0.1초 뒤로"
              className="p-2 rounded-lg bg-surface-2 border border-border text-muted hover:text-foreground disabled:opacity-40">
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={isPlaying ? pause : play}
              disabled={!hasVideo}
              className={cn(
                "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold",
                !hasVideo ? "bg-surface-2 text-muted cursor-not-allowed" : "bg-accent text-background hover:opacity-90"
              )}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isPlaying ? "정지" : "동시 재생"}
            </button>
            <button onClick={() => step(STEP)} disabled={!hasVideo} aria-label="0.1초 앞으로"
              className="p-2 rounded-lg bg-surface-2 border border-border text-muted hover:text-foreground disabled:opacity-40">
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>

          <div className="flex items-center gap-1">
            {[0.25, 0.5, 1, 1.5, 2].map((r) => (
              <button
                key={r}
                onClick={() => changeSpeed(r)}
                className={cn(
                  "px-2 py-1 rounded text-xs font-mono font-medium",
                  speed === r ? "bg-accent text-background" : "bg-surface-2 text-muted hover:text-foreground border border-border"
                )}
              >
                {r}x
              </button>
            ))}
          </div>
        </div>
        <p className="text-[11px] text-muted">확대: 휠·두 손가락·더블클릭 · 확대한 뒤 드래그로 이동 · ◀ ▶ 0.1초씩 이동</p>
      </div>
      {/* 세로 영상 두 개를 휴대폰에서도 나란히 */}
      <div className="grid grid-cols-2 gap-2 sm:gap-4">
        <ZoomableVideo ref={refA} path={pathOf(selA)} label="A" onTime={onTime("A")} />
        <ZoomableVideo ref={refB} path={pathOf(selB)} label="B" onTime={onTime("B")} />
      </div>

    </div>
  );
}
