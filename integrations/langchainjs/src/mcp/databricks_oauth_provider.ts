/**
 * OAuth provider for Databricks authentication with MCP servers.
 *
 * Bridges Databricks SDK authentication to MCP's OAuth interface,
 * allowing MCP servers to use Databricks authentication methods
 * (PAT, OAuth, Azure MSI, etc.) transparently.
 *
 * Based on the Python implementation at:
 * https://github.com/databricks/databricks-ai-bridge/blob/main/databricks_mcp/src/databricks_mcp/oauth_provider.py
 */

import type { Config } from "@databricks/sdk-experimental";
import type { OAuthClientProvider } from "@modelcontextprotocol/sdk/client/auth.js";
import type {
  OAuthClientMetadata,
  OAuthTokens,
  OAuthClientInformationMixed,
} from "@modelcontextprotocol/sdk/shared/auth.js";

const TOKEN_EXPIRATION_SECONDS = 60;

/**
 * DatabricksOAuthClientProvider bridges Databricks SDK authentication to MCP.
 *
 * This class implements the MCP SDK's OAuthClientProvider interface using
 * Databricks SDK authentication. It provides Bearer tokens from the Databricks
 * SDK's authentication chain (PAT, OAuth, Azure MSI, etc.) without requiring
 * interactive OAuth flows.
 *
 * The implementation mirrors the Python DatabricksOAuthClientProvider which
 * uses a similar token storage pattern to provide pre-authenticated tokens.
 *
 * @example
 * ```typescript
 * import { Config } from "@databricks/sdk-experimental";
 * import { DatabricksOAuthClientProvider } from "@databricks/langchain-ts";
 *
 * const config = new Config();
 * const authProvider = new DatabricksOAuthClientProvider(config);
 * ```
 */
export class DatabricksOAuthClientProvider implements OAuthClientProvider {
  private config: Config;

  constructor(config: Config) {
    this.config = config;
  }

  /**
   * Returns undefined to indicate non-interactive flow.
   * This disables the redirect-based OAuth flow.
   */
  get redirectUrl(): undefined {
    return undefined;
  }

  /**
   * Returns null/undefined client metadata to skip OAuth client registration.
   * The Python implementation passes client_metadata=None to achieve this.
   */
  get clientMetadata(): OAuthClientMetadata {
    // Return minimal metadata - the key is that we provide tokens() directly
    // so the OAuth flow doesn't need to do client registration
    return { redirect_uris: [] };
  }

  /**
   * Returns a dummy client information object to skip OAuth client registration.
   *
   * When this returns a value (not undefined), the MCP SDK skips the dynamic
   * client registration step and proceeds directly with token handling.
   * Since we provide tokens via tokens(), we don't need real client credentials.
   */
  clientInformation(): OAuthClientInformationMixed | undefined {
    // Return a dummy client_id to skip registration
    // The actual auth is done via Bearer tokens from tokens()
    return {
      client_id: "databricks-sdk-client",
    };
  }

  /**
   * No-op - we don't save client info for Bearer token auth.
   */
  saveClientInformation(_clientInfo: OAuthClientInformationMixed): void {
    // No-op
  }

  /**
   * Prepares token request parameters for non-interactive flows.
   *
   * This enables the SDK to use client_credentials grant type, which
   * will use our clientInformation to authenticate. However, the actual
   * tokens come from our tokens() method via the Databricks SDK.
   */
  prepareTokenRequest(_scope?: string): URLSearchParams | undefined {
    // Return client_credentials grant parameters
    // The actual token endpoint call may fail, but our tokens() method
    // provides the real tokens for Authorization headers
    const params = new URLSearchParams({
      grant_type: "client_credentials",
    });
    return params;
  }

  /**
   * Returns current OAuth tokens from Databricks authentication.
   *
   * This is the key method - it provides tokens directly from the Databricks SDK,
   * bypassing the need for OAuth discovery and client registration.
   */
  async tokens(): Promise<OAuthTokens | undefined> {
    // Ensure config is resolved (handles async auth like OAuth)
    await this.config.ensureResolved();

    // Create headers and authenticate using the SDK
    const headers = new Headers();
    await this.config.authenticate(headers);

    const authHeader = headers.get("Authorization");
    if (!authHeader?.startsWith("Bearer ")) {
      throw new Error("Invalid authentication token format. Expected Bearer token.");
    }

    const token = authHeader.split("Bearer ")[1];
    return {
      access_token: token,
      token_type: "Bearer",
      // Short expiration ensures regular token refresh
      expires_in: TOKEN_EXPIRATION_SECONDS,
    };
  }

  /**
   * No-op - Databricks SDK manages its own token lifecycle.
   */
  saveTokens(_tokens: OAuthTokens): void {
    // No-op
  }

  /**
   * Throws - redirect-based OAuth not supported.
   */
  redirectToAuthorization(_authorizationUrl: URL): void {
    throw new Error("Redirect-based OAuth not supported for Databricks authentication");
  }

  /**
   * No-op - PKCE not used for Bearer token auth.
   */
  saveCodeVerifier(_codeVerifier: string): void {
    // No-op
  }

  /**
   * Returns empty string - PKCE not used for Bearer token auth.
   */
  codeVerifier(): string {
    return "";
  }
}

export type { OAuthClientProvider };
