import { matchPath } from 'react-router-dom';
import { getConfig } from './config';

/**
 * Soft navigation to a chat ID - updates the browser URL without triggering
 * React Router navigation. Uses window.history.replaceState so the URL bar
 * updates but no component remount occurs.
 *
 * When chat history is disabled (ephemeral mode) the URL stays unchanged.
 */
export function softNavigateToChatId(
  chatId: string,
  chatHistoryEnabled: boolean,
): void {
  if (!chatHistoryEnabled) return;

  const { basePath } = getConfig();
  const base = basePath.replace(/\/$/, '');
  window.history.replaceState({}, '', `${base}/chat/${chatId}`);
}

/**
 * Extract the chat ID from the current browser pathname.
 * Accounts for the configurable base path.
 */
export function getChatIdFromPathname(): string | undefined {
  const { basePath } = getConfig();
  const base = basePath.replace(/\/$/, '');
  const pathname = window.location.pathname;
  const relative = pathname.startsWith(base)
    ? pathname.slice(base.length)
    : pathname;
  return matchPath('/chat/:id', relative)?.params.id;
}
