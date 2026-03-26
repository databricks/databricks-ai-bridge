import type { Request } from "express";
import type { ChatSession, GetSession } from "./types";
import { getScimUser } from "./auth";

/**
 * Resolve session for the request.
 *
 * Resolution order:
 * 1. Custom getSession callback (if provided in plugin config)
 * 2. x-forwarded-user headers (set by the Databricks Apps proxy in production)
 * 3. SCIM API /Me (local dev with Databricks auth available)
 * 4. process.env.USER fallback (bare local dev)
 */
export async function resolveSession(
  req: Request,
  getSession?: GetSession,
): Promise<ChatSession | null> {
  if (getSession) {
    return Promise.resolve(getSession(req));
  }

  const forwardedUser = req.header("x-forwarded-user");
  if (forwardedUser) {
    return {
      user: {
        id: forwardedUser,
        email: req.header("x-forwarded-email") ?? undefined,
        name: req.header("x-forwarded-preferred-username") ?? undefined,
        preferredUsername:
          req.header("x-forwarded-preferred-username") ?? undefined,
      },
    };
  }

  const scimUser = await getScimUser();
  if (scimUser) {
    return { user: scimUser };
  }

  const localUser = process.env.USER || process.env.USERNAME;
  if (localUser) {
    return {
      user: {
        id: localUser,
        email: `${localUser}@localhost`,
        name: localUser,
      },
    };
  }

  return null;
}
