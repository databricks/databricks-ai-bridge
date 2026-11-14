import { fetchWithErrorHandlers } from './utils';
import { apiUrl } from './config';
import type { VisibilityType } from '@/types';

/**
 * Update chat visibility (public/private)
 */
export async function updateChatVisibility({
  chatId,
  visibility,
}: {
  chatId: string;
  visibility: VisibilityType;
}) {
  const response = await fetchWithErrorHandlers(
    apiUrl(`/${chatId}/visibility`),
    {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include',
      body: JSON.stringify({ visibility }),
    },
  );

  return response.json();
}

/**
 * Delete messages after a certain timestamp
 */
export async function deleteTrailingMessages({
  messageId,
}: {
  messageId: string;
}) {
  const response = await fetchWithErrorHandlers(
    apiUrl(`/messages/${messageId}/trailing`),
    {
      method: 'DELETE',
      credentials: 'include',
    },
  );

  if (response.status === 204) {
    return null;
  }

  return response.json();
}
