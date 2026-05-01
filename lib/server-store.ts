"use client";

// 로컬 서버 URL을 앱 전역에서 공유하기 위한 간단한 클라이언트 스토어
// localStorage에 저장해 새로고침 후에도 유지

const KEY = "climbing_local_server_url";
const DEFAULT_URL = "http://localhost:8000";

export function getServerUrl(): string {
  if (typeof window === "undefined") return DEFAULT_URL;
  return localStorage.getItem(KEY) || DEFAULT_URL;
}

export function setServerUrl(url: string): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(KEY, url.replace(/\/$/, ""));
}

export function apiUrl(path: string): string {
  const base = getServerUrl();
  return `${base}${path.startsWith("/") ? path : "/" + path}`;
}

/**
 * ngrok 무료 플랜은 브라우저 직접 접근 시 경고 페이지를 띄운다.
 * `ngrok-skip-browser-warning` 헤더를 모든 API 요청에 자동으로 추가한다.
 * 일반 localhost 환경에서는 아무 영향 없음.
 */
function getNgrokHeaders(): HeadersInit {
  const url = getServerUrl();
  if (url.includes("ngrok")) {
    return { "ngrok-skip-browser-warning": "true" };
  }
  return {};
}

/** fetch를 ngrok 헤더와 함께 수행하는 공통 함수 */
export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const url = apiUrl(path);
  const headers = {
    ...getNgrokHeaders(),
    ...(init?.headers || {}),
  };
  return fetch(url, { ...init, headers });
}

/** SWR fetcher - ngrok 헤더 포함 */
export function swrFetcher(url: string) {
  const headers = getNgrokHeaders();
  return fetch(url, { headers }).then((r) => r.json());
}
