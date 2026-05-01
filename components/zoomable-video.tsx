"use client";

import { useRef, useState, useCallback, forwardRef, useImperativeHandle } from "react";

interface ZoomableVideoProps {
  src: string;
  label: string;
}

export interface ZoomableVideoHandle {
  play: () => void;
  pause: () => void;
  setPlaybackRate: (rate: number) => void;
}

const ZoomableVideo = forwardRef<ZoomableVideoHandle, ZoomableVideoProps>(
  ({ src, label }, ref) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const wrapRef = useRef<HTMLDivElement>(null);
    const [scale, setScale] = useState(1);
    const [tx, setTx] = useState(0);
    const [ty, setTy] = useState(0);
    const dragging = useRef(false);
    const startX = useRef(0);
    const startY = useRef(0);

    useImperativeHandle(ref, () => ({
      play: () => videoRef.current?.play(),
      pause: () => videoRef.current?.pause(),
      setPlaybackRate: (rate) => { if (videoRef.current) videoRef.current.playbackRate = rate; },
    }));

    const handleWheel = useCallback((e: React.WheelEvent) => {
      e.preventDefault();
      setScale((s) => Math.min(5, Math.max(1, s + (e.deltaY > 0 ? -0.15 : 0.15))));
    }, []);

    const handleMouseDown = (e: React.MouseEvent) => {
      dragging.current = true;
      startX.current = e.clientX - tx;
      startY.current = e.clientY - ty;
    };

    const handleMouseMove = (e: React.MouseEvent) => {
      if (!dragging.current) return;
      setTx(e.clientX - startX.current);
      setTy(e.clientY - startY.current);
    };

    const handleMouseUp = () => { dragging.current = false; };

    const resetZoom = () => { setScale(1); setTx(0); setTy(0); };

    return (
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-muted uppercase tracking-wider">{label}</span>
          {scale > 1 && (
            <button onClick={resetZoom} className="text-xs text-accent hover:underline">
              줌 초기화
            </button>
          )}
        </div>
        <div
          ref={wrapRef}
          className="relative w-full h-80 bg-black rounded-lg overflow-hidden cursor-grab active:cursor-grabbing select-none"
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          {src ? (
            <video
              ref={videoRef}
              src={src}
              controls
              muted
              playsInline
              className="w-full h-full object-contain pointer-events-none"
              style={{ transform: `translate(${tx}px, ${ty}px) scale(${scale})`, transformOrigin: "center center" }}
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center text-muted text-sm">
              영상을 선택하세요
            </div>
          )}
          {scale > 1 && (
            <div className="absolute top-2 right-2 bg-black/60 text-white text-xs px-2 py-0.5 rounded-full font-mono pointer-events-none">
              {scale.toFixed(1)}x
            </div>
          )}
        </div>
        <p className="text-xs text-muted">휠: 줌 | 드래그: 이동</p>
      </div>
    );
  }
);

ZoomableVideo.displayName = "ZoomableVideo";
export default ZoomableVideo;
