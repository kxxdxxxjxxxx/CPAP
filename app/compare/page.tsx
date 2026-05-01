"use client";

import { useRef, useState } from "react";
import useSWR from "swr";
import { Play, Pause, ChevronRight } from "lucide-react";
import ZoomableVideo, { ZoomableVideoHandle } from "@/components/zoomable-video";
import { apiUrl, swrFetcher } from "@/lib/server-store";
import { cn } from "@/lib/utils";

interface VideoItem {
  name: string;
  url: string;
}

export default function ComparePage() {
  const { data, isLoading } = useSWR<{ videos: VideoItem[] }>(
    apiUrl("/videos"),
    swrFetcher,
    { refreshInterval: 5000 }
  );

  const [srcA, setSrcA] = useState("");
  const [srcB, setSrcB] = useState("");
  const [selectedA, setSelectedA] = useState("");
  const [selectedB, setSelectedB] = useState("");
  const [speed, setSpeed] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);

  const refA = useRef<ZoomableVideoHandle>(null);
  const refB = useRef<ZoomableVideoHandle>(null);

  const handleSelect = (side: "A" | "B", video: VideoItem) => {
    const fullUrl = apiUrl(video.url) + "?t=" + Date.now();
    if (side === "A") { setSrcA(fullUrl); setSelectedA(video.name); }
    else { setSrcB(fullUrl); setSelectedB(video.name); }
  };

  const handlePlay = () => {
    refA.current?.play();
    refB.current?.play();
    setIsPlaying(true);
  };

  const handlePause = () => {
    refA.current?.pause();
    refB.current?.pause();
    setIsPlaying(false);
  };

  const handleSpeedChange = (rate: number) => {
    setSpeed(rate);
    refA.current?.setPlaybackRate(rate);
    refB.current?.setPlaybackRate(rate);
  };

  const videos = data?.videos || [];

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-foreground text-balance">영상 비교</h1>
        <p className="mt-1 text-sm text-muted leading-relaxed">
          분석된 두 영상을 나란히 배치하여 동시 재생·정지·속도 조절로 비교합니다.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-5">
        {/* List A */}
        <div className="bg-surface rounded-lg border border-border p-4">
          <p className="text-xs font-semibold text-muted uppercase tracking-wider mb-3">
            A 영상 선택
          </p>
          <div className="flex flex-col gap-1 max-h-48 overflow-y-auto pr-1">
            {isLoading ? (
              <p className="text-xs text-muted py-2">불러오는 중...</p>
            ) : videos.length === 0 ? (
              <p className="text-xs text-muted py-2">분석된 영상이 없습니다.</p>
            ) : (
              videos.map((v) => (
                <button
                  key={v.name + "A"}
                  onClick={() => handleSelect("A", v)}
                  className={cn(
                    "flex items-center justify-between px-3 py-2.5 rounded-lg text-left text-xs transition-all",
                    selectedA === v.name
                      ? "bg-accent-dim text-accent border border-accent/20"
                      : "text-muted hover:text-foreground hover:bg-surface-2 border border-transparent"
                  )}
                >
                  <span className="truncate flex-1">{v.name}</span>
                  {selectedA === v.name && <ChevronRight className="w-3 h-3 shrink-0 ml-1" />}
                </button>
              ))
            )}
          </div>
        </div>

        {/* List B */}
        <div className="bg-surface rounded-lg border border-border p-4">
          <p className="text-xs font-semibold text-muted uppercase tracking-wider mb-3">
            B 영상 선택
          </p>
          <div className="flex flex-col gap-1 max-h-48 overflow-y-auto pr-1">
            {isLoading ? (
              <p className="text-xs text-muted py-2">불러오는 중...</p>
            ) : videos.length === 0 ? (
              <p className="text-xs text-muted py-2">분석된 영상이 없습니다.</p>
            ) : (
              videos.map((v) => (
                <button
                  key={v.name + "B"}
                  onClick={() => handleSelect("B", v)}
                  className={cn(
                    "flex items-center justify-between px-3 py-2.5 rounded-lg text-left text-xs transition-all",
                    selectedB === v.name
                      ? "bg-accent-dim text-accent border border-accent/20"
                      : "text-muted hover:text-foreground hover:bg-surface-2 border border-transparent"
                  )}
                >
                  <span className="truncate flex-1">{v.name}</span>
                  {selectedB === v.name && <ChevronRight className="w-3 h-3 shrink-0 ml-1" />}
                </button>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3 mb-5 p-3 bg-surface rounded-lg border border-border">
        <button
          onClick={isPlaying ? handlePause : handlePlay}
          disabled={!srcA && !srcB}
          className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all",
            !srcA && !srcB
              ? "bg-surface-2 text-muted cursor-not-allowed"
              : "bg-accent text-background hover:opacity-90"
          )}
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {isPlaying ? "동시 정지" : "동시 재생"}
        </button>

        <div className="flex items-center gap-2">
          <span className="text-xs text-muted">재생 속도</span>
          <div className="flex gap-1">
            {[0.25, 0.5, 1, 1.5, 2].map((r) => (
              <button
                key={r}
                onClick={() => handleSpeedChange(r)}
                className={cn(
                  "px-2.5 py-1 rounded text-xs font-mono font-medium transition-all",
                  speed === r
                    ? "bg-accent text-background"
                    : "bg-surface-2 text-muted hover:text-foreground border border-border"
                )}
              >
                {r}x
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Video players */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ZoomableVideo ref={refA} src={srcA} label="A" />
        <ZoomableVideo ref={refB} src={srcB} label="B" />
      </div>
    </div>
  );
}
