"use client";

import React, { useRef, useState, useCallback } from "react";

export interface PoseKeypoint {
  x: number; // 0..1 정규화 좌표 (이미지 contain 영역 기준)
  y: number; // 0..1
  v: 0 | 1 | 2; // 0=없음(라벨 안 함), 1=가려짐(추정 위치), 2=보임
}

export const KEYPOINT_NAMES = [
  "코",
  "왼눈",
  "오른눈",
  "왼귀",
  "오른귀",
  "왼어깨",
  "오른어깨",
  "왼팔꿈치",
  "오른팔꿈치",
  "왼손목",
  "오른손목",
  "왼골반",
  "오른골반",
  "왼무릎",
  "오른무릎",
  "왼발목",
  "오른발목",
];

export const SKELETON: [number, number][] = [
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  [5, 6],
  [5, 7],
  [6, 8],
  [7, 9],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [12, 14],
  [13, 15],
  [14, 16],
];

// 좌측 신체 인덱스 (홀수): 5,7,9,11,13,15
// 우측 신체 인덱스 (짝수): 6,8,10,12,14,16
function colorForIndex(i: number): string {
  if (i === 0) return "var(--color-foreground)"; // 코
  if (i >= 1 && i <= 4) return "var(--color-muted)"; // 눈/귀
  return i % 2 === 1 ? "var(--color-accent)" : "var(--color-warning)";
}

interface KeypointCanvasProps {
  imageSrc: string;
  imageWidth: number;
  imageHeight: number;
  keypoints: PoseKeypoint[]; // 길이 17
  onKeypointsChange: (kpts: PoseKeypoint[]) => void;
  selectedIndex?: number;
  onSelect?: (i: number) => void;
}

export default function KeypointCanvas({
  imageSrc,
  imageWidth,
  imageHeight,
  keypoints,
  onKeypointsChange,
  selectedIndex = -1,
  onSelect,
}: KeypointCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef<number | null>(null);
  const movedRef = useRef(false);
  const [hoverIdx, setHoverIdx] = useState(-1);

  // 이미지 contain 영역 비율: 컨테이너에 이미지 비율 적용
  const aspect =
    imageWidth && imageHeight ? imageWidth / imageHeight : 16 / 9;

  const eventToNorm = useCallback(
    (e: React.PointerEvent | PointerEvent) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return null;
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;
      return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
    },
    []
  );

  const onPointerDown = (i: number) => (e: React.PointerEvent) => {
    e.stopPropagation();
    draggingRef.current = i;
    movedRef.current = false;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    onSelect?.(i);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (draggingRef.current === null) return;
    const pos = eventToNorm(e);
    if (!pos) return;
    // 드래그 거리가 작으면 클릭으로 간주
    const idx = draggingRef.current;
    const cur = keypoints[idx];
    const dx = pos.x - cur.x;
    const dy = pos.y - cur.y;
    if (!movedRef.current && Math.hypot(dx, dy) < 0.005) return;
    movedRef.current = true;
    const next = [...keypoints];
    // 드래그 시 자동으로 visible 처리 (v=0 이었다면 v=2로)
    next[idx] = {
      ...cur,
      x: pos.x,
      y: pos.y,
      v: cur.v === 0 ? 2 : cur.v,
    };
    onKeypointsChange(next);
  };

  const onPointerUp = (i: number) => (e: React.PointerEvent) => {
    if (draggingRef.current !== i) return;
    if (!movedRef.current) {
      // 클릭만 한 경우 → visibility 사이클: 2 → 1 → 0 → 2
      const next = [...keypoints];
      const cur = next[i].v;
      const nv: 0 | 1 | 2 = cur === 2 ? 1 : cur === 1 ? 0 : 2;
      next[i] = { ...next[i], v: nv };
      onKeypointsChange(next);
    }
    draggingRef.current = null;
    movedRef.current = false;
    try {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full bg-black rounded-lg overflow-hidden select-none touch-none"
      style={{ aspectRatio: String(aspect) }}
      onPointerMove={onPointerMove}
    >
      {imageSrc ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={imageSrc}
          alt=""
          draggable={false}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center text-muted text-sm">
          이미지를 선택하세요
        </div>
      )}

      {/* 스켈레톤 라인 */}
      <svg
        className="absolute inset-0 w-full h-full pointer-events-none"
        preserveAspectRatio="none"
        viewBox="0 0 1 1"
      >
        {SKELETON.map(([a, b], idx) => {
          const ka = keypoints[a];
          const kb = keypoints[b];
          if (!ka || !kb || ka.v === 0 || kb.v === 0) return null;
          const occ = ka.v === 1 || kb.v === 1;
          return (
            <line
              key={idx}
              x1={ka.x}
              y1={ka.y}
              x2={kb.x}
              y2={kb.y}
              stroke={colorForIndex(b)}
              strokeWidth={0.005}
              strokeLinecap="round"
              opacity={occ ? 0.5 : 0.85}
              strokeDasharray={occ ? "0.012 0.008" : undefined}
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
      </svg>

      {/* 키포인트 점들 */}
      {keypoints.map((kp, i) => {
        const isSel = selectedIndex === i;
        const visible = kp.v > 0;
        const occ = kp.v === 1;
        const color = colorForIndex(i);
        return (
          <button
            key={i}
            type="button"
            onPointerDown={onPointerDown(i)}
            onPointerUp={onPointerUp(i)}
            onPointerEnter={() => setHoverIdx(i)}
            onPointerLeave={() => setHoverIdx(-1)}
            className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full cursor-grab active:cursor-grabbing"
            style={{
              left: `${kp.x * 100}%`,
              top: `${kp.y * 100}%`,
              width: visible ? 14 : 10,
              height: visible ? 14 : 10,
              backgroundColor: visible && !occ ? color : "transparent",
              border: `2px ${occ ? "dashed" : "solid"} ${color}`,
              boxShadow: isSel
                ? `0 0 0 2px var(--color-foreground)`
                : visible
                  ? `0 0 0 1px rgba(0,0,0,0.6)`
                  : "none",
              opacity: visible ? 1 : 0.5,
              zIndex: isSel ? 30 : visible ? 20 : 10,
            }}
            aria-label={`${KEYPOINT_NAMES[i]} (v=${kp.v})`}
          >
            {(isSel || hoverIdx === i) && (
              <span className="absolute left-1/2 -translate-x-1/2 -top-6 text-[10px] font-mono text-foreground bg-black/85 px-1.5 py-0.5 rounded whitespace-nowrap pointer-events-none">
                {i} · {KEYPOINT_NAMES[i]}
                {occ ? " · 가려짐" : ""}
                {kp.v === 0 ? " · 없음" : ""}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
