import { ChatSDKError, type ErrorCode } from "../types.js";

export function generateUUID(): string {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export function apiUrl(apiBase: string, path: string): string {
  if (path.startsWith("/")) return `${apiBase}${path}`;
  return `${apiBase}/${path}`;
}

export async function fetchWithErrorHandlers(
  input: RequestInfo | URL,
  init?: RequestInit,
) {
  try {
    const response = await fetch(input, init);

    if (!response.ok) {
      const parsedResponse = await response.json();
      const { code, cause } = parsedResponse;
      throw new ChatSDKError(code as ErrorCode, cause);
    }

    return response;
  } catch (error: unknown) {
    if (typeof navigator !== "undefined" && !navigator.onLine) {
      throw new ChatSDKError("offline:chat");
    }

    throw error;
  }
}

export const fetcher = async (url: string) => {
  const response = await fetch(url);

  if (!response.ok) {
    const { code, cause } = await response.json();
    throw new ChatSDKError(code as ErrorCode, cause);
  }

  if (response.status === 204) {
    return { chats: [], hasMore: false };
  }

  return response.json();
};

export function isCredentialErrorMessage(errorMessage: string): boolean {
  const pattern =
    /Credential for user identity\([^)]*\) is not found for the connection/i;
  return pattern.test(errorMessage);
}
