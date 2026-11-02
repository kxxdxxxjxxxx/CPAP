"use client";

import { useRef, useState, useEffect, forwardRef, useImperativeHandle } from "react";
import { Loader2, ZoomIn, ZoomOut } from "lucide-react";
import { apiFetch } from "@/lib/server-store";

interface ZoomableVideoProps {
  /** 서버 경로 (예: /outputs/xxx_lesson.mp4). 비우면 빈 칸 */
  path: string;
  label: string;
  onTime?: (t: number, duration: number) => void;
}

export interface ZoomableVideoHandle {
  play: () => Promise<void> | void;
  pause: () => void;
  setPlaybackRate: (rate: number) => void;
  seek: (t: number) => void;
  duration: () => number;
}

const MAX_ZOOM = 5;

/**
 * 세로(9:16) 비율 영상 칸. 휠·핀치·버튼으로 확대, 드래그로 이동.
 * ngrok 무료 플랜은 <video src> 직접 요청에 경고 페이지를 주고, HTTPS 사이트에서 localhost 영상을 막는 브라우저도 있어
 * 영상을 헤더와 함께 받아와(blob) 재생한다.
 */
const ZoomableVideo = forwardRef<ZoomableVideoHandle, ZoomableVideoProps>(({ path, label, onTime }, ref) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [blobUrl, setBlobUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [scale, setScale] = useState(1);
  const [tx, setTx] = useState(0);
  const [ty, setTy] = useState(0);
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const drag = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);
  const pinch = useRef<{ dist: number; scale: number } | null>(null);

  // 영상 받아오기
  useEffect(() => {
    let cancelled = false;
    let url = "";
    setError("");
    setBlobUrl("");
    resetZoom();
    if (!path) return;
    setLoading(true);
    apiFetch(path)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const blob = await res.blob();
        if (!blob.type.startsWith("video/") && blob.size < 20000) throw new Error("영상이 아닌 응답");
        url = URL.createObjectURL(blob);
        if (!cancelled) setBlobUrl(url);
      })
      .catch(() => { if (!cancelled) setError("영상을 불러오지 못했어요. 서버 연결을 확인해 주세요."); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [path]);

  useImperativeHandle(ref, () => ({
    play: () => videoRef.current?.play().catch(() => {}),
    pause: () => videoRef.current?.pause(),
    setPlaybackRate: (rate) => { if (videoRef.current) videoRef.current.playbackRate = rate; },
    seek: (t) => {
      const v = videoRef.current;
      if (v && Number.isFinite(v.duration)) v.currentTime = Math.min(Math.max(0, t), v.duration);
    },
    duration: () => videoRef.current?.duration || 0,
  }));

  function resetZoom() { setScale(1); setTx(0); setTy(0); }
  const clampScale = (s: number) => Math.min(MAX_ZOOM, Math.max(1, s));
  const zoomBy = (d: number) => setScale((s) => { const n = clampScale(s + d); if (n === 1) { setTx(0); setTy(0); } return n; });

  const onWheel = (e: React.WheelEvent) => { if (blobUrl) zoomBy(e.deltaY > 0 ? -0.2 : 0.2); };

  const onPointerDown = (e: React.PointerEvent) => {
    if (!blobUrl) return;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.current.size === 1) {
      drag.current = { x: e.clientX, y: e.clientY, tx, ty };
    } else if (pointers.current.size === 2) {
      const [a, b] = Array.from(pointers.current.values());
      pinch.current = { dist: Math.hypot(a.x - b.x, a.y - b.y), scale };
      drag.current = null;
    }
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!pointers.current.has(e.pointerId)) return;
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pinch.current && pointers.current.size >= 2) {
      const [a, b] = Array.from(pointers.current.values());
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      setScale(clampScale(pinch.current.scale * (d / Math.max(1, pinch.current.dist))));
    } else if (drag.current && scale > 1) {
      setTx(drag.current.tx + e.clientX - drag.current.x);
      setTy(drag.current.ty + e.clientY - drag.current.y);
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    pointers.current.delete(e.pointerId);
    if (pointers.current.size < 2) pinch.current = null;
    if (pointers.current.size === 0) drag.current = null;
  };

  return (
    <div className="flex flex-col gap-2 min-w-0">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-semibold text-muted uppercase tracking-wider">{label}</span>
        <div className="flex items-center gap-1">
          <button onClick={() => zoomBy(-0.5)} disabled={!blobUrl || scale <= 1} aria-label="축소"
            className="p-1 rounded text-muted hover:text-foreground disabled:opacity-30"><ZoomOut className="w-4 h-4" /></button>
          <button onClick={() => zoomBy(0.5)} disabled={!blobUrl || scale >= MAX_ZOOM} aria-label="확대"
            className="p-1 rounded text-muted hover:text-foreground disabled:opacity-30"><ZoomIn className="w-4 h-4" /></button>
          {scale > 1 && (
            <button onClick={resetZoom} className="text-xs text-accent hover:underline ml-1">원래 크기</button>
          )}
        </div>
      </div>

      {/* 클라이밍 영상은 세로가 많아 9:16 칸. 화면 높이를 넘지 않게 제한 */}
      <div
        className="relative w-full max-w-[calc(72vh*9/16)] aspect-[9/16] mx-auto bg-black rounded-lg overflow-hidden select-none touch-none"
        style={{ cursor: blobUrl ? (scale > 1 ? "grab" : "zoom-in") : "default" }}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={() => (scale > 1 ? resetZoom() : zoomBy(1))}
      >
        {blobUrl ? (
          <video
            ref={videoRef}
            src={blobUrl}
            muted
            playsInline
            preload="auto"
            onTimeUpdate={(e) => onTime?.(e.currentTarget.currentTime, e.currentTarget.duration || 0)}
            onLoadedMetadata={(e) => onTime?.(0, e.currentTarget.duration || 0)}
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
            style={{ transform: `translate(${tx}px, ${ty}px) scale(${scale})`, transformOrigin: "center center" }}
          />
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-muted text-sm px-4 text-center">
            {loading ? (<><Loader2 className="w-5 h-5 animate-spin" />영상 불러오는 중...</>)
              : error ? <span className="text-danger/80">{error}</span>
              : "목록에서 영상을 선택하세요"}
          </div>
        )}
        {scale > 1 && (
          <div className="absolute top-2 right-2 bg-black/60 text-white text-xs px-2 py-0.5 rounded-full font-mono pointer-events-none">
            {scale.toFixed(1)}x
          </div>
        )}
      </div>
    </div>
  );
});

ZoomableVideo.displayName = "ZoomableVideo";
export default ZoomableVideo;
