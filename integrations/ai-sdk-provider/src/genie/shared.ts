import { combineHeaders, type FetchFunction, withoutTrailingSlash } from '@ai-sdk/provider-utils'

export type DatabricksGeniePollingSettings = {
  timeoutMs?: number
  initialPollIntervalMs?: number
  maxPollIntervalMs?: number
  backoffMultiplier?: number
}

export interface DatabricksGenieDebugEvent {
  body?: unknown
  headers?: Record<string, string>
  method: string
  metadata?: unknown
  phase: 'request' | 'response'
  status?: number
  statusText?: string
  url: string
}

export type DatabricksGenieDebugLogger = (event: DatabricksGenieDebugEvent) => void

export type DatabricksGenieDebugSettings =
  | boolean
  | {
      logBodies?: boolean
      logger?: DatabricksGenieDebugLogger
    }

export interface DatabricksGenieSettings extends DatabricksGeniePollingSettings {
  baseURL: string
  debug?: DatabricksGenieDebugSettings
  spaceId: string
  headers?: Record<string, string>
  fetch?: FetchFunction
  formatUrl?: (options: { baseUrl?: string; path: string }) => string
}

export type DatabricksGenieResolvedSettings = {
  baseUrl: string
  debug: {
    enabled: boolean
    logBodies: boolean
    logger: DatabricksGenieDebugLogger
  }
  spaceId: string
  headers: () => Record<string, string | undefined>
  fetch: FetchFunction
  url: (path: string) => string
  polling: Required<DatabricksGeniePollingSettings>
}

export const DEFAULT_GENIE_POLLING_SETTINGS: Required<DatabricksGeniePollingSettings> = {
  timeoutMs: 600_000,
  initialPollIntervalMs: 1_000,
  maxPollIntervalMs: 60_000,
  backoffMultiplier: 2,
}

export function resolveDatabricksGenieSettings(
  settings: DatabricksGenieSettings
): DatabricksGenieResolvedSettings {
  const baseUrl: string = withoutTrailingSlash(settings.baseURL) ?? settings.baseURL
  const debugSettings =
    typeof settings.debug === 'object' && settings.debug !== null ? settings.debug : {}
  const debugEnabled = Boolean(settings.debug)

  return {
    baseUrl,
    debug: {
      enabled: debugEnabled,
      logBodies: debugSettings.logBodies ?? true,
      logger:
        debugSettings.logger ??
        ((event) => {
          console.debug('[databricks-genie]', event)
        }),
    },
    spaceId: settings.spaceId,
    headers: () => combineHeaders(settings.headers),
    fetch: settings.fetch ?? globalThis.fetch.bind(globalThis),
    url: (path: string) => settings.formatUrl?.({ baseUrl, path }) ?? `${baseUrl}${path}`,
    polling: {
      timeoutMs: settings.timeoutMs ?? DEFAULT_GENIE_POLLING_SETTINGS.timeoutMs,
      initialPollIntervalMs:
        settings.initialPollIntervalMs ?? DEFAULT_GENIE_POLLING_SETTINGS.initialPollIntervalMs,
      maxPollIntervalMs:
        settings.maxPollIntervalMs ?? DEFAULT_GENIE_POLLING_SETTINGS.maxPollIntervalMs,
      backoffMultiplier:
        settings.backoffMultiplier ?? DEFAULT_GENIE_POLLING_SETTINGS.backoffMultiplier,
    },
  }
}

export async function ensureOkResponse(response: Response): Promise<void> {
  if (!response.ok) {
    let body = ''
    try {
      body = await response.text()
    } catch {
      // Ignore response body parsing errors and fall back to status text.
    }

    throw new Error(
      `Databricks Genie request failed with ${response.status} ${response.statusText}${
        body ? `: ${body}` : ''
      }`
    )
  }
}

export async function parseJsonResponse(response: Response): Promise<unknown> {
  await ensureOkResponse(response)

  return response.json()
}

export async function sleep(ms: number, abortSignal?: AbortSignal): Promise<void> {
  if (ms <= 0) {
    return
  }

  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => {
      abortSignal?.removeEventListener('abort', onAbort)
      resolve()
    }, ms)

    const onAbort = () => {
      clearTimeout(timer)
      reject(new Error('Databricks Genie request was aborted'))
    }

    if (abortSignal) {
      if (abortSignal.aborted) {
        onAbort()
        return
      }

      abortSignal.addEventListener('abort', onAbort, { once: true })
    }
  })
}

export function getGenieHeaders(
  settings: DatabricksGenieResolvedSettings,
  headers?: Record<string, string>
): Record<string, string> {
  const combinedHeaders = combineHeaders(
    {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    settings.headers(),
    headers
  )

  return Object.fromEntries(
    Object.entries(combinedHeaders).filter(
      (entry): entry is [string, string] => typeof entry[1] === 'string'
    )
  )
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

export function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string')
}

export function sanitizeGenieHeaders(
  headers: Record<string, string>
): Record<string, string> {
  return Object.fromEntries(
    Object.entries(headers).map(([key, value]) => {
      if (
        key.toLowerCase() === 'authorization' ||
        key.toLowerCase() === 'cookie' ||
        key.toLowerCase() === 'set-cookie'
      ) {
        return [key, '[REDACTED]']
      }

      return [key, value]
    })
  )
}

export function logGenieDebugEvent(
  settings: DatabricksGenieResolvedSettings,
  event: DatabricksGenieDebugEvent
): void {
  if (!settings.debug.enabled) {
    return
  }

  settings.debug.logger(event)
}
