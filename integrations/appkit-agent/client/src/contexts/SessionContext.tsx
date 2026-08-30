import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
} from 'react';
import type { ClientSession } from '@/types';
import { apiUrl } from '@/lib/config';
import { ChatContext } from './ChatProvider';

interface SessionContextType {
  session: ClientSession | null;
  loading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<ClientSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchSession = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(apiUrl('/session'), {
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error('Failed to fetch session');
      }

      const data = await response.json();
      setSession(data);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'));
      setSession(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSession();
  }, [fetchSession]);

  return (
    <SessionContext.Provider
      value={{ session, loading, error, refetch: fetchSession }}
    >
      {children}
    </SessionContext.Provider>
  );
}

/**
 * Read user session. Transparently delegates to ChatProvider when present
 * in the tree, otherwise falls back to the legacy SessionContext.
 */
export function useSession(): SessionContextType {
  const chatCtx = useContext(ChatContext);
  if (chatCtx) {
    return {
      session: chatCtx.session,
      loading: chatCtx.isLoading,
      error: null,
      refetch: async () => {},
    };
  }

  const legacy = useContext(SessionContext);
  if (!legacy) {
    throw new Error(
      'useSession must be used within a ChatProvider or SessionProvider',
    );
  }
  return legacy;
}
