"use client";

// 진행 중인 분석 작업을 기억해서, 다른 화면에 갔다 오거나 새로고침해도 이어서 보여준다

export interface ActiveJob {
  jobId: string;
  fileName: string;
}

const KEY = "cpap_active_job";

export function loadActiveJob(): ActiveJob | null {
  try {
    const raw = localStorage.getItem(KEY);
    const v = raw ? JSON.parse(raw) : null;
    return v && typeof v.jobId === "string" ? v : null;
  } catch {
    return null;
  }
}

export function saveActiveJob(job: ActiveJob) {
  try {
    localStorage.setItem(KEY, JSON.stringify(job));
  } catch {
    /* 저장 불가 환경 - 무시 */
  }
}

export function clearActiveJob() {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* noop */
  }
}
