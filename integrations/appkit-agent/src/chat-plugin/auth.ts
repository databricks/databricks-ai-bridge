/**
 * Consolidated Databricks authentication module.
 *
 * Supported auth methods (checked in priority order):
 *   1. PAT            — DATABRICKS_TOKEN env var
 *   2. OAuth (SP)     — DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET + DATABRICKS_HOST
 *   3. CLI (OAuth U2M)— DATABRICKS_CONFIG_PROFILE or DATABRICKS_HOST
 *                       (requires `databricks auth login`)
 */

import { execFile } from "node:child_process";

// ── Types ──────────────────────────────────────────────────────────

export type AuthMethod = "pat" | "oauth" | "cli" | "none";

export interface ScimUser {
  id: string;
  email: string;
  name?: string;
  preferredUsername?: string;
}

// ── Caches ─────────────────────────────────────────────────────────

let oauthToken: string | null = null;
let oauthTokenExpiresAt = 0;

let cliToken: string | null = null;
let cliTokenExpiresAt = 0;

let cachedCliHost: string | null = null;
let cliHostTime = 0;
const CLI_HOST_CACHE_MS = 10 * 60 * 1000;

let cachedScimUser: ScimUser | null = null;
let scimCacheExpiry = 0;
const SCIM_CACHE_MS = 30 * 60 * 1000;

// ── Auth method detection ──────────────────────────────────────────

export function getAuthMethod(): AuthMethod {
  if (process.env.DATABRICKS_TOKEN) return "pat";
  if (
    process.env.DATABRICKS_CLIENT_ID &&
    process.env.DATABRICKS_CLIENT_SECRET &&
    process.env.DATABRICKS_HOST
  ) {
    return "oauth";
  }
  if (process.env.DATABRICKS_CONFIG_PROFILE || process.env.DATABRICKS_HOST) {
    return "cli";
  }
  return "none";
}

// ── Host resolution ────────────────────────────────────────────────

function normalizeHost(host: string): string {
  return host.replace(/^https?:\/\//, "").replace(/\/$/, "");
}

/**
 * Get the Databricks workspace URL (`https://…`).
 *
 * Resolution order:
 *  1. `DATABRICKS_HOST` env var
 *  2. Host cached from a previous CLI call
 *  3. `databricks auth describe` (derives host from profile)
 */
export async function getHostUrl(): Promise<string> {
  const envHost = process.env.DATABRICKS_HOST;
  if (envHost) {
    return `https://${normalizeHost(envHost)}`;
  }

  if (cachedCliHost && Date.now() < cliHostTime + CLI_HOST_CACHE_MS) {
    return `https://${normalizeHost(cachedCliHost)}`;
  }

  const profile = process.env.DATABRICKS_CONFIG_PROFILE;
  if (!profile) {
    throw new Error(
      "Chat plugin: set DATABRICKS_HOST or DATABRICKS_CONFIG_PROFILE.",
    );
  }

  const stdout = await runCli([
    "auth",
    "describe",
    "--output",
    "json",
    "--profile",
    profile,
  ]);
  const data = JSON.parse(stdout) as { details?: { host?: string } };
  const host = data.details?.host;
  if (!host) {
    throw new Error(
      "Could not resolve workspace host from `databricks auth describe`. " +
        "Set DATABRICKS_HOST explicitly.",
    );
  }

  cachedCliHost = host;
  cliHostTime = Date.now();
  return `https://${normalizeHost(host)}`;
}

/**
 * Get the workspace hostname without protocol.
 */
export async function getHostDomain(): Promise<string> {
  const url = await getHostUrl();
  return normalizeHost(url);
}

// ── Token acquisition ──────────────────────────────────────────────

/**
 * Get a Databricks auth token using the best available method.
 */
export async function getToken(): Promise<string> {
  const method = getAuthMethod();
  switch (method) {
    case "pat":
      return process.env.DATABRICKS_TOKEN!;
    case "oauth":
      return getOAuthToken();
    case "cli":
      return getCliToken();
    case "none":
      throw new Error(
        "No Databricks auth configured. Set one of:\n" +
          "  • DATABRICKS_TOKEN (PAT)\n" +
          "  • DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET + DATABRICKS_HOST (OAuth SP)\n" +
          "  • DATABRICKS_CONFIG_PROFILE or DATABRICKS_HOST (CLI — run `databricks auth login` first)",
      );
  }
}

/**
 * OAuth client-credentials flow for service principals.
 */
async function getOAuthToken(): Promise<string> {
  if (oauthToken && Date.now() < oauthTokenExpiresAt) return oauthToken;

  const clientId = process.env.DATABRICKS_CLIENT_ID!;
  const clientSecret = process.env.DATABRICKS_CLIENT_SECRET!;
  const hostUrl = await getHostUrl();

  const response = await fetch(
    `${hostUrl}/oidc/v1/token`,
    {
      method: "POST",
      headers: {
        Authorization: `Basic ${Buffer.from(`${clientId}:${clientSecret}`).toString("base64")}`,
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: "grant_type=client_credentials&scope=all-apis",
    },
  );

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`OAuth token request failed: ${response.status} ${text}`);
  }

  const data = (await response.json()) as {
    access_token: string;
    expires_in?: number;
  };
  if (!data.access_token) {
    throw new Error("No access_token in OAuth response");
  }

  oauthToken = data.access_token;
  const expiresIn = data.expires_in ?? 3600;
  const buffer = Math.min(600, Math.floor(expiresIn * 0.2));
  oauthTokenExpiresAt = Date.now() + (expiresIn - buffer) * 1000;

  return oauthToken;
}

/**
 * CLI-based token acquisition (`databricks auth token`).
 * Supports OAuth U2M with automatic refresh.
 */
async function getCliToken(): Promise<string> {
  if (cliToken && Date.now() < cliTokenExpiresAt) return cliToken;

  const stdout = await runCli([
    "auth",
    "token",
    "--output",
    "json",
    ...cliProfileArgs(),
  ]);

  const data = JSON.parse(stdout) as {
    access_token?: string;
    expiry?: string;
  };
  if (!data.access_token) {
    throw new Error("No access_token in `databricks auth token` output");
  }

  cliToken = data.access_token;
  const bufferMs = 5 * 60 * 1000;
  cliTokenExpiresAt = data.expiry
    ? new Date(data.expiry).getTime() - bufferMs
    : Date.now() + 55 * 60 * 1000;

  return cliToken;
}

// ── SCIM (/Me) ─────────────────────────────────────────────────────

/**
 * Fetch the current user from the Databricks SCIM `/Me` endpoint.
 * Useful in local development to resolve a real Databricks identity
 * instead of falling back to the OS username.
 *
 * Returns `null` on any failure (network, auth, missing config).
 */
export async function getScimUser(): Promise<ScimUser | null> {
  if (cachedScimUser && Date.now() < scimCacheExpiry) return cachedScimUser;

  let token: string;
  let host: string;
  try {
    token = await getToken();
    host = await getHostUrl();
  } catch {
    return null;
  }

  try {
    const resp = await fetch(`${host}/api/2.0/preview/scim/v2/Me`, {
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
    });

    if (!resp.ok) return null;

    const data = (await resp.json()) as {
      id?: string;
      userName?: string;
      displayName?: string;
      emails?: Array<{ value: string; primary?: boolean }>;
    };

    const primaryEmail =
      data.emails?.find((e) => e.primary)?.value ??
      data.emails?.[0]?.value ??
      (data.userName ? `${data.userName}@databricks.com` : undefined);

    cachedScimUser = {
      id: data.id ?? data.userName ?? "unknown",
      email: primaryEmail ?? `${data.userName ?? "user"}@databricks.com`,
      name: data.displayName ?? data.userName,
      preferredUsername: data.userName,
    };
    scimCacheExpiry = Date.now() + SCIM_CACHE_MS;
    return cachedScimUser;
  } catch {
    return null;
  }
}

// ── CLI helpers ────────────────────────────────────────────────────

function cliProfileArgs(): string[] {
  const args: string[] = [];
  const profile = process.env.DATABRICKS_CONFIG_PROFILE;
  const host = process.env.DATABRICKS_HOST;
  if (profile) args.push("--profile", profile);
  if (host) args.push("--host", normalizeHost(host));
  return args;
}

function runCli(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile("databricks", args, { timeout: 15_000 }, (err, stdout, stderr) => {
      if (err) {
        reject(
          new Error(
            `\`databricks ${args.slice(0, 2).join(" ")}\` failed: ${stderr || err.message}\n` +
              'Ensure the Databricks CLI is installed and you have run "databricks auth login".',
          ),
        );
        return;
      }
      resolve(stdout);
    });
  });
}
