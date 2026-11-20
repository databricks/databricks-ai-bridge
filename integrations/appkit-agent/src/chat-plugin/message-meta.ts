const MAX_ENTRIES = 10_000;

function setBounded<K, V>(map: Map<K, V>, key: K, value: V): void {
  if (map.size >= MAX_ENTRIES) {
    const oldest = map.keys().next().value as K;
    map.delete(oldest);
  }
  map.set(key, value);
}

const store = new Map<string, { traceId: string | null; chatId: string }>();

export function storeMessageMeta(
  messageId: string,
  chatId: string,
  traceId: string | null,
) {
  setBounded(store, messageId, { traceId, chatId });
}

export function getMessageMetadata(messageId: string) {
  return store.get(messageId) ?? null;
}

export function getMessageMetasByChatId(
  chatId: string,
): Array<{ messageId: string; traceId: string }> {
  const results: Array<{ messageId: string; traceId: string }> = [];
  for (const [messageId, meta] of store) {
    if (meta.chatId === chatId && meta.traceId != null) {
      results.push({ messageId, traceId: meta.traceId });
    }
  }
  return results;
}

const assessmentStore = new Map<string, string>();

export function storeAssessmentId(
  messageId: string,
  userId: string,
  assessmentId: string,
) {
  setBounded(assessmentStore, `${messageId}:${userId}`, assessmentId);
}

export function getAssessmentId(
  messageId: string,
  userId: string,
): string | null {
  return assessmentStore.get(`${messageId}:${userId}`) ?? null;
}
