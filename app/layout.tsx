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
  description: "클라이밍 영상의 무게중심 궤적과 이동 속도를 분석하는 강습 보조 도구",
  keywords: ["클라이밍", "무게중심", "동작 분석", "강습"],
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
