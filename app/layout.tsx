import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import ShellClient from "@/components/shell-client";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Climbing Pose Analyzer",
  description: "클라이밍 영상 포즈 분석, 홀드 탐지 학습, 관절 보정 파라미터 튜닝 플랫폼",
  keywords: ["클라이밍", "포즈 분석", "YOLO", "AI", "스포츠 분석"],
};

export const viewport: Viewport = {
  themeColor: "#0c0d10",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko" className="bg-background">
      <body className={`${geistSans.variable} ${geistMono.variable} font-sans antialiased`}>
        <ShellClient>{children}</ShellClient>
      </body>
    </html>
  );
}
