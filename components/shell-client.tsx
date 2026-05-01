"use client";

import { useState, useEffect, useCallback } from "react";
import { usePathname } from "next/navigation";
import Sidebar from "@/components/sidebar";
import ServerConnectBar from "@/components/server-connect-bar";

export default function ShellClient({ children }: { children: React.ReactNode }) {
  // 데스크톱: 기본 열림 / 모바일(< 768px): 기본 닫힘
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isMobile, setIsMobile] = useState(false);
  const pathname = usePathname();

  // 창 크기 감지 - 모바일 여부 판단
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 767px)");
    const handler = (e: MediaQueryListEvent) => {
      setIsMobile(e.matches);
      // 데스크톱으로 넓어지면 자동으로 열기
      if (!e.matches) setSidebarOpen(true);
    };
    setIsMobile(mq.matches);
    if (mq.matches) setSidebarOpen(false);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  // 모바일에서 페이지 이동 시 자동 닫기
  useEffect(() => {
    if (isMobile) setSidebarOpen(false);
  }, [pathname, isMobile]);

  const toggleSidebar = useCallback(() => setSidebarOpen((v) => !v), []);
  const closeSidebar = useCallback(() => setSidebarOpen(false), []);

  return (
    <div className="flex h-screen overflow-hidden relative">
      {/* 모바일 오버레이 - 사이드바 열릴 때 배경 어둡게 */}
      {isMobile && sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/60 backdrop-blur-sm"
          onClick={closeSidebar}
          aria-hidden="true"
        />
      )}

      {/* 사이드바 */}
      <div
        className={[
          "h-full flex flex-col shrink-0 transition-[width,transform] duration-300 ease-in-out",
          isMobile
            // 모바일: fixed + slide in/out (너비는 항상 w-64, transform으로 슬라이드)
            ? "fixed z-30 top-0 left-0 bottom-0 w-64 " +
              (sidebarOpen ? "translate-x-0 shadow-2xl" : "-translate-x-full")
            // 데스크톱: inline collapse (너비를 0 ↔ 64으로 트랜지션)
            : sidebarOpen
            ? "w-64 overflow-hidden"
            : "w-0 overflow-hidden",
        ].join(" ")}
      >
        <Sidebar onClose={closeSidebar} />
      </div>

      {/* 콘텐츠 영역 */}
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <ServerConnectBar onToggleSidebar={toggleSidebar} sidebarOpen={sidebarOpen} />
        <main className="flex-1 overflow-y-auto">
          {children}
        </main>
      </div>
    </div>
  );
}
