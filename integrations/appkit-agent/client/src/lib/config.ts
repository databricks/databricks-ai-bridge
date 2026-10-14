export interface ChatClientConfig {
  apiBase: string;
  basePath: string;
}

declare global {
  interface Window {
    __CHAT_CONFIG__?: ChatClientConfig;
  }
}

export function getConfig(): ChatClientConfig {
  return (
    window.__CHAT_CONFIG__ ?? {
      apiBase: "/api/chat",
      basePath: "/",
    }
  );
}

export function apiUrl(path: string): string {
  const base = getConfig().apiBase;
  if (path.startsWith("/")) return `${base}${path}`;
  return `${base}/${path}`;
}
