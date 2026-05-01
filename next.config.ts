import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // 로컬 FastAPI 서버로 API 프록시 (개발 환경에서 CORS 우회용 - 선택적)
  // 사용자가 웹 UI에서 직접 URL을 입력하는 방식이므로 프록시 불필요
};

export default nextConfig;
