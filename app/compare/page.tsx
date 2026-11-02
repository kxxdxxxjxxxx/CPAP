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

  type Target = "A" | "B" | "both";
  const [selA, setSelA] = useState("");
  const [selB, setSelB] = useState("");
  const [target, setTarget] = useState<Target>("both");   // 조작할 영상
  const [speed, setSpeed] = useState(1);
  const [playA, setPlayA] = useState(false);
  const [playB, setPlayB] = useState(false);
  const [tA, setTA] = useState(0);
  const [tB, setTB] = useState(0);
  const [durA, setDurA] = useState(0);
  const [durB, setDurB] = useState(0);

  const refA = useRef<ZoomableVideoHandle>(null);
  const refB = useRef<ZoomableVideoHandle>(null);
  const pathOf = (key: string) => videos.find((v) => keyOf(v) === key)?.url || "";

  // 지금 조작 대상인 영상들 (선택되지 않은 영상은 제외)
  const sides = (): ("A" | "B")[] =>
    (target === "both" ? (["A", "B"] as const) : [target]).filter((x) => (x === "A" ? selA : selB)) as ("A" | "B")[];
  const refOf = (x: "A" | "B") => (x === "A" ? refA : refB).current;
  const each = (f: (r: ZoomableVideoHandle, x: "A" | "B") => void) => sides().forEach((x) => { const r = refOf(x); if (r) f(r, x); });

  const hasTarget = sides().length > 0;
  const playing = sides().some((x) => (x === "A" ? playA : playB));
  // 재생바는 동시 모드면 A(없으면 B) 기준, 개별 모드면 그 영상 기준
  const lead: "A" | "B" = target === "B" || (target === "both" && !selA) ? "B" : "A";
  const time = lead === "A" ? tA : tB;
  const duration = lead === "A" ? durA : durB;

  const select = (side: "A" | "B", key: string) => {
    refOf(side)?.pause();
    if (side === "A") { setSelA(key); setDurA(0); setTA(0); } else { setSelB(key); setDurB(0); setTB(0); }
  };

  const play = () => each((r) => { r.setPlaybackRate(speed); r.play(); });
  const pause = () => each((r) => r.pause());
  // 동시 모드에서 이동하면 두 영상을 같은 만큼 움직여 서로 맞춰둔 시작점이 유지된다
  const moveBy = (d: number) => { pause(); each((r) => r.seek(r.currentTime() + d)); };
  const seekTo = (t: number) => { const cur = refOf(lead)?.currentTime() ?? time; moveBy(t - cur); };
  const toStart = () => {
    pause();
    if (target === "both") { const m = Math.min(...sides().map((x) => refOf(x)?.currentTime() ?? 0)); moveBy(-m); }
    else each((r) => r.seek(0));
  };
  const changeSpeed = (r: number) => { setSpeed(r); [refA, refB].forEach((x) => x.current?.setPlaybackRate(r)); };

  const onTime = (side: "A" | "B") => (t: number, d: number) => {
    if (side === "A") { setTA(t); setDurA(d); } else { setTB(t); setDurB(d); }
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
          분석한 두 영상을 나란히 놓고, 하나씩 또는 동시에 재생·이동하며 비교합니다.
        </p>
        {error && <p className="mt-2 text-xs text-danger">영상 목록을 불러오지 못했어요. 서버 연결을 확인해 주세요.</p>}
      </div>

      <div className="grid grid-cols-2 gap-3 mb-3">
        {picker("A", selA)}
        {picker("B", selB)}
      </div>

      {/* 조작 */}
      <div className="mb-3 p-3 bg-surface rounded-lg border border-border space-y-3">
        <div className="grid grid-cols-3 gap-1 p-1 bg-surface-2 rounded-lg" role="radiogroup" aria-label="조작할 영상">
          {([["A", "A 영상"], ["B", "B 영상"], ["both", "동시"]] as const).map(([v, l]) => (
            <button
              key={v}
              role="radio"
              aria-checked={target === v}
              onClick={() => setTarget(v)}
              className={cn(
                "py-1.5 rounded-md text-sm font-medium transition-colors",
                target === v ? "bg-accent text-background" : "text-muted hover:text-foreground"
              )}
            >
              {l}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-muted tabular-nums w-12">{fmt(time)}</span>
          <input
            type="range" min={0} max={duration || 0} step={0.05} value={Math.min(time, duration || 0)}
            disabled={!hasTarget || !duration}
            onChange={(e) => seekTo(parseFloat(e.target.value))}
            className="flex-1 accent-[var(--color-accent)]"
            aria-label="재생 위치"
          />
          <span className="text-xs font-mono text-muted tabular-nums w-12 text-right">{fmt(duration)}</span>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-1">
            <button onClick={toStart} disabled={!hasTarget} aria-label="처음으로"
              className="p-2 rounded-lg bg-surface-2 border border-border text-muted hover:text-foreground disabled:opacity-40">
              <SkipBack className="w-4 h-4" />
            </button>
            <button onClick={() => moveBy(-STEP)} disabled={!hasTarget} aria-label="0.1초 뒤로"
              className="p-2 rounded-lg bg-surface-2 border border-border text-muted hover:text-foreground disabled:opacity-40">
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={playing ? pause : play}
              disabled={!hasTarget}
              className={cn(
                "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold min-w-[7.5rem] justify-center",
                !hasTarget ? "bg-surface-2 text-muted cursor-not-allowed" : "bg-accent text-background hover:opacity-90"
              )}
            >
              {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {(target === "both" ? "동시 " : `${target} `) + (playing ? "정지" : "재생")}
            </button>
            <button onClick={() => moveBy(STEP)} disabled={!hasTarget} aria-label="0.1초 앞으로"
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
        <p className="text-[11px] text-muted leading-relaxed">
          시작점 맞추기: A 영상·B 영상을 하나씩 골라 같은 동작 순간에 멈춘 뒤 &apos;동시&apos;로 재생하세요. · ◀ ▶ 0.1초씩 이동 · 확대: 휠·두 손가락·더블클릭
        </p>
      </div>
      {/* 세로 영상 두 개를 휴대폰에서도 나란히 */}
      <div className="grid grid-cols-2 gap-2 sm:gap-4">
        <ZoomableVideo ref={refA} path={pathOf(selA)} label={`A · ${fmt(tA)}`} onTime={onTime("A")}
          onPlayingChange={setPlayA} active={target !== "B" && !!selA} />
        <ZoomableVideo ref={refB} path={pathOf(selB)} label={`B · ${fmt(tB)}`} onTime={onTime("B")}
          onPlayingChange={setPlayB} active={target !== "A" && !!selB} />
      </div>

    </div>
  );
}
