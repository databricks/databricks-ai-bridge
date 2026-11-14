import type { MessageRow } from "./schema";

export function generateUUID(): string {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export function convertToUIMessages(messages: MessageRow[]): Array<{
  id: string;
  role: "user" | "assistant" | "system";
  parts: unknown[];
  metadata?: { createdAt: string };
}> {
  return messages.map((m) => ({
    id: m.id,
    role: m.role as "user" | "assistant" | "system",
    parts: m.parts as unknown[],
    metadata: { createdAt: m.createdAt.toISOString() },
  }));
}

export function truncatePreserveWords(
  input: string,
  maxLength: number,
): string {
  if (maxLength <= 0) return "";
  if (input.length <= maxLength) return input;
  const slice = input.slice(0, maxLength);
  const lastSpaceIndex = slice.lastIndexOf(" ");
  if (lastSpaceIndex <= 0) return slice;
  return slice.slice(0, lastSpaceIndex);
}
