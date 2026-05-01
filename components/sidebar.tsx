"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Video, GitCompare, Target, Sliders, ChevronRight, X } from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  {
    href: "/",
    icon: Video,
    label: "영상 분석",
    description: "클라이밍 영상 업로드 및 포즈 분석",
  },
  {
    href: "/compare",
    icon: GitCompare,
    label: "영상 비교",
    description: "분석된 영상 A/B 비교 재생",
  },
  {
    href: "/train",
    icon: Target,
    label: "홀드 학습",
    description: "벽 사진으로 홀드 탐지 모델 학습",
  },
  {
    href: "/params",
    icon: Sliders,
    label: "파라미터 튜닝",
    description: "관절 보정 및 분석 파라미터 조정",
  },
];

interface SidebarProps {
  onClose?: () => void;
}

export default function Sidebar({ onClose }: SidebarProps) {
  const pathname = usePathname();

  return (
    <aside className="w-64 shrink-0 bg-surface border-r border-border flex flex-col h-full">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-border">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded bg-accent flex items-center justify-center shrink-0">
            <svg viewBox="0 0 24 24" fill="none" className="w-4 h-4 text-background" stroke="currentColor" strokeWidth={2.5}>
              <path d="M12 2a7 7 0 1 1 0 14A7 7 0 0 1 12 2z" strokeLinecap="round" />
              <path d="M12 9v3l2 2" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M4.93 17.07A10 10 0 0 0 12 22a10 10 0 0 0 7.07-2.93" strokeLinecap="round" />
            </svg>
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-bold text-foreground leading-tight">Climbing</div>
            <div className="text-xs text-muted leading-tight">Pose Analyzer</div>
          </div>
          {/* 닫기 버튼 */}
          {onClose && (
            <button
              onClick={onClose}
              className="p-1 rounded text-muted hover:text-foreground hover:bg-surface-2 transition-colors shrink-0"
              aria-label="사이드바 닫기"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 flex flex-col gap-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "group flex items-center gap-3 px-3 py-3 rounded-lg transition-all",
                isActive
                  ? "bg-accent-dim text-accent border border-accent/20"
                  : "text-muted hover:text-foreground hover:bg-surface-2 border border-transparent"
              )}
            >
              <Icon className={cn("w-4 h-4 shrink-0", isActive ? "text-accent" : "text-muted group-hover:text-foreground")} />
              <div className="flex-1 min-w-0">
                <div className={cn("text-sm font-medium", isActive ? "text-accent" : "")}>{item.label}</div>
                <div className="text-xs text-muted truncate leading-relaxed">{item.description}</div>
              </div>
              {isActive && <ChevronRight className="w-3 h-3 text-accent shrink-0" />}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-border space-y-2">
        <p className="text-xs text-muted leading-relaxed">
          로컬 FastAPI 서버와 연결하여 <br />클라이밍 동작을 분석합니다.
        </p>
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
          <p className="text-xs text-muted">
            외부 접속:{" "}
            <code className="font-mono text-accent/80 text-[10px]">python app.py --ngrok</code>
          </p>
        </div>
      </div>
    </aside>
  );
}
