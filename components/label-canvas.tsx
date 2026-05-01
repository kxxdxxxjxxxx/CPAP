"use client";

import { useEffect, useRef, useState } from "react";

export type LabelBox = {
  cx: number; // 0..1 (normalized)
  cy: number; // 0..1
  w: number;  // 0..1
  h: number;  // 0..1
};

interface LabelCanvasProps {
  imageSrc: string;     // blob: URL 권장 (ngrok 우회 위해 부모에서 fetch 후 전달)
  imageWidth: number;   // 원본 픽셀 (없으면 0)
  imageHeight: number;
  boxes: LabelBox[];
  boxSize: number;      // 단순 클릭 시 기본 박스 크기 (normalized, 0..1)
  onBoxesChange: (next: LabelBox[]) => void;
}

// 클릭과 드래그 구분 임계치 (이미지 표시 영역 픽셀 기준)
const DRAG_THRESHOLD_PX = 5;

/**
 * 클릭/드래그로 박스를 추가하고 박스 클릭으로 삭제하는 라벨링 캔버스.
 * - 단순 클릭(드래그 거리 < 5px): boxSize 만큼의 정사각 박스(이미지 비율 보정) 추가
 * - 드래그: 드래그한 영역 그대로 박스 추가
 * - 기존 박스 클릭: 해당 박스 삭제
 */
export default function LabelCanvas({
  imageSrc,
  imageWidth,
  imageHeight,
  boxes,
  boxSize,
  onBoxesChange,
}: LabelCanvasProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  // 표시 영역 안에서의 이미지 실제 표시 위치 (object-contain 결과)
  const [layout, setLayout] = useState({ ox: 0, oy: 0, dw: 0, dh: 0 });

  // 드래그 상태 (display 좌표계, 이미지 영역 기준)
  const dragStart = useRef<{ x: number; y: number } | null>(null);
  const [dragRect, setDragRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const movedPx = useRef(0);

  // 컨테이너 크기 변경 / 이미지 로드 시 contain 영역 계산
  const recompute = () => {
    if (!wrapRef.current || imageWidth <= 0 || imageHeight <= 0) return;
    const cw = wrapRef.current.clientWidth;
    const ch = wrapRef.current.clientHeight;
    const ratioImg = imageWidth / imageHeight;
    const ratioBox = cw / ch;
    let dw: number, dh: number;
    if (ratioImg > ratioBox) {
      dw = cw;
      dh = cw / ratioImg;
    } else {
      dh = ch;
      dw = ch * ratioImg;
    }
    setLayout({ ox: (cw - dw) / 2, oy: (ch - dh) / 2, dw, dh });
  };

  useEffect(() => {
    recompute();
    const ro = new ResizeObserver(recompute);
    if (wrapRef.current) ro.observe(wrapRef.current);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageWidth, imageHeight, imageSrc]);

  // 마우스 위치 → 이미지 영역 내 픽셀 좌표 (이미지 바깥이면 null)
  const toImagePx = (clientX: number, clientY: number) => {
    if (!wrapRef.current || layout.dw <= 0) return null;
    const rect = wrapRef.current.getBoundingClientRect();
    const px = clientX - rect.left - layout.ox;
    const py = clientY - rect.top - layout.oy;
    if (px < 0 || py < 0 || px > layout.dw || py > layout.dh) return null;
    return { px, py };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    // 좌클릭만
    if (e.button !== 0) return;
    const p = toImagePx(e.clientX, e.clientY);
    if (!p) return;
    dragStart.current = { x: p.px, y: p.py };
    movedPx.current = 0;
    setDragRect(null);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!dragStart.current) return;
    const p = toImagePx(e.clientX, e.clientY);
    if (!p) return;
    const dx = p.px - dragStart.current.x;
    const dy = p.py - dragStart.current.y;
    movedPx.current = Math.max(movedPx.current, Math.hypot(dx, dy));
    if (movedPx.current >= DRAG_THRESHOLD_PX) {
      const x = Math.min(dragStart.current.x, p.px);
      const y = Math.min(dragStart.current.y, p.py);
      const w = Math.abs(dx);
      const h = Math.abs(dy);
      setDragRect({ x, y, w, h });
    }
  };

  const handleMouseUp = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!dragStart.current) return;
    const start = dragStart.current;
    const moved = movedPx.current;
    dragStart.current = null;

    const p = toImagePx(e.clientX, e.clientY);

    // 드래그 모드: 임계치 이상 움직였고 끝점이 이미지 영역 안
    if (moved >= DRAG_THRESHOLD_PX && p) {
      const x = Math.min(start.x, p.px);
      const y = Math.min(start.y, p.py);
      const w = Math.abs(p.px - start.x);
      const h = Math.abs(p.py - start.y);
      setDragRect(null);
      // 너무 작은 드래그는 무시 (실수 방지)
      if (w < 4 || h < 4) return;
      const cx = (x + w / 2) / layout.dw;
      const cy = (y + h / 2) / layout.dh;
      const nw = w / layout.dw;
      const nh = h / layout.dh;
      onBoxesChange([...boxes, { cx, cy, w: nw, h: nh }]);
      return;
    }

    setDragRect(null);

    // 단순 클릭 모드: 기본 박스 크기로 추가 (화면상 정사각형이 되도록 normalized 보정)
    if (!p) return;
    const cx = start.x / layout.dw;
    const cy = start.y / layout.dh;
    const s = Math.max(0.005, Math.min(1, boxSize));
    let nw = s;
    let nh = s;
    if (imageWidth > 0 && imageHeight > 0) {
      // 화면상 정사각형 → h_norm = w_norm * (W/H)
      nh = (s * imageWidth) / imageHeight;
    }
    onBoxesChange([...boxes, { cx, cy, w: nw, h: nh }]);
  };

  const handleMouseLeave = () => {
    // 캔버스 밖으로 나가면 진행 중 드래그 취소
    dragStart.current = null;
    setDragRect(null);
  };

  const handleBoxMouseDown = (e: React.MouseEvent) => {
    // 박스 위에서는 새 박스 드래그가 시작되지 않도록 차단
    e.stopPropagation();
  };

  const handleBoxClick = (e: React.MouseEvent, idx: number) => {
    e.stopPropagation();
    const next = boxes.filter((_, i) => i !== idx);
    onBoxesChange(next);
  };

  return (
    <div
      ref={wrapRef}
      className="relative w-full h-[480px] bg-black/60 rounded-lg overflow-hidden cursor-crosshair select-none"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseLeave}
    >
      {imageSrc ? (
        <img
          src={imageSrc || "/placeholder.svg"}
          alt="label target"
          className="absolute pointer-events-none"
          draggable={false}
          style={{
            left: `${layout.ox}px`,
            top: `${layout.oy}px`,
            width: `${layout.dw}px`,
            height: `${layout.dh}px`,
            objectFit: "fill",
          }}
          onLoad={recompute}
        />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center text-muted text-sm">
          이미지를 선택하세요
        </div>
      )}

      {/* 기존 박스 오버레이 */}
      {imageSrc && layout.dw > 0 && boxes.map((b, i) => {
        const left = layout.ox + (b.cx - b.w / 2) * layout.dw;
        const top = layout.oy + (b.cy - b.h / 2) * layout.dh;
        const w = b.w * layout.dw;
        const h = b.h * layout.dh;
        return (
          <div
            key={i}
            onMouseDown={handleBoxMouseDown}
            onClick={(e) => handleBoxClick(e, i)}
            className="absolute border-2 border-accent bg-accent/15 hover:bg-danger/30 hover:border-danger transition-colors cursor-pointer"
            style={{ left, top, width: w, height: h }}
            title="클릭해서 삭제"
          >
            <span className="absolute -top-5 left-0 text-[10px] font-mono bg-accent text-background px-1 rounded">
              {i + 1}
            </span>
          </div>
        );
      })}

      {/* 드래그 미리보기 */}
      {dragRect && (
        <div
          className="absolute border-2 border-dashed border-accent bg-accent/10 pointer-events-none"
          style={{
            left: layout.ox + dragRect.x,
            top: layout.oy + dragRect.y,
            width: dragRect.w,
            height: dragRect.h,
          }}
        />
      )}
    </div>
  );
}
