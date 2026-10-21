"use client";

// 클라이머(학생) 프로필: 이름 · 키 · 윙스팬
// 이 브라우저에만 저장된다 (다른 기기와 공유되지 않음)

export interface ClimberProfile {
  id: string;
  name: string;
  heightCm: number;
  wingspanCm: number;
}

const KEY = "cpap_climber_profiles";
const LAST_KEY = "cpap_last_profile_id";

export const HEIGHT_RANGE: [number, number] = [100, 230];
export const WINGSPAN_RANGE: [number, number] = [100, 240];

export function loadProfiles(): ClimberProfile[] {
  try {
    const raw = localStorage.getItem(KEY);
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}

function saveAll(list: ClimberProfile[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(list));
  } catch {
    // 저장 불가(사생활 보호 모드 등) - 이번 세션에서만 사용
  }
}

export function upsertProfile(p: Omit<ClimberProfile, "id"> & { id?: string }): ClimberProfile {
  const list = loadProfiles();
  const id = p.id || `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
  const item: ClimberProfile = { id, name: p.name.trim(), heightCm: p.heightCm, wingspanCm: p.wingspanCm };
  const idx = list.findIndex((x) => x.id === id);
  if (idx >= 0) list[idx] = item;
  else list.unshift(item);
  saveAll(list);
  return item;
}

export function deleteProfile(id: string) {
  saveAll(loadProfiles().filter((x) => x.id !== id));
}

export function getLastProfileId(): string {
  try {
    return localStorage.getItem(LAST_KEY) || "";
  } catch {
    return "";
  }
}

export function setLastProfileId(id: string) {
  try {
    localStorage.setItem(LAST_KEY, id);
  } catch {
    /* noop */
  }
}

export function validBody(heightCm: number, wingspanCm: number): string | null {
  if (!Number.isFinite(heightCm) || heightCm < HEIGHT_RANGE[0] || heightCm > HEIGHT_RANGE[1])
    return `키는 ${HEIGHT_RANGE[0]}~${HEIGHT_RANGE[1]}cm 사이로 입력하세요.`;
  if (!Number.isFinite(wingspanCm) || wingspanCm < WINGSPAN_RANGE[0] || wingspanCm > WINGSPAN_RANGE[1])
    return `윙스팬은 ${WINGSPAN_RANGE[0]}~${WINGSPAN_RANGE[1]}cm 사이로 입력하세요.`;
  const ratio = wingspanCm / heightCm;
  if (ratio < 0.85 || ratio > 1.2) return "키와 윙스팬 차이가 너무 큽니다. 값을 다시 확인하세요.";
  return null;
}
