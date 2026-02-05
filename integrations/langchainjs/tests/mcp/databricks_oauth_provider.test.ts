/**
 * Unit tests for DatabricksOAuthClientProvider
 */

import { describe, it, expect, vi } from "vitest";
import { DatabricksOAuthClientProvider } from "../../src/mcp/databricks_oauth_provider.js";
import type { Config } from "@databricks/sdk-experimental";

function createMockConfig(authHeader = "Bearer test-token-123"): Config {
  return {
    ensureResolved: vi.fn().mockResolvedValue(undefined),
    authenticate: vi.fn().mockImplementation((headers: Headers) => {
      if (authHeader) {
        headers.set("Authorization", authHeader);
      }
      return Promise.resolve();
    }),
  } as unknown as Config;
}

describe("DatabricksOAuthClientProvider", () => {
  it("returns expected values for OAuth interface properties", () => {
    const provider = new DatabricksOAuthClientProvider(createMockConfig());

    expect(provider.redirectUrl).toBeUndefined();
    expect(provider.clientMetadata).toEqual({ redirect_uris: [] });
    expect(provider.clientInformation()).toEqual({ client_id: "databricks-sdk-client" });
    expect(provider.codeVerifier()).toBe("");
    expect(provider.prepareTokenRequest()?.get("grant_type")).toBe("client_credentials");
  });

  it("no-op methods do not throw", () => {
    const provider = new DatabricksOAuthClientProvider(createMockConfig());

    expect(() => provider.saveClientInformation({ client_id: "test" })).not.toThrow();
    expect(() => provider.saveTokens({ access_token: "t", token_type: "Bearer" })).not.toThrow();
    expect(() => provider.saveCodeVerifier("verifier")).not.toThrow();
  });

  it("throws on redirectToAuthorization since redirect OAuth is not supported", () => {
    const provider = new DatabricksOAuthClientProvider(createMockConfig());

    expect(() => provider.redirectToAuthorization(new URL("https://example.com/auth"))).toThrow(
      "Redirect-based OAuth not supported"
    );
  });

  describe("tokens", () => {
    it("returns parsed Bearer token from Databricks SDK", async () => {
      const config = createMockConfig("Bearer my-access-token");
      const provider = new DatabricksOAuthClientProvider(config);

      const tokens = await provider.tokens();

      expect(tokens).toEqual({
        access_token: "my-access-token",
        token_type: "Bearer",
        expires_in: 60,
      });
      expect(config.ensureResolved).toHaveBeenCalled();
      expect(config.authenticate).toHaveBeenCalled();
    });

    it("throws error for non-Bearer auth", async () => {
      const provider = new DatabricksOAuthClientProvider(createMockConfig("Basic creds"));

      await expect(provider.tokens()).rejects.toThrow("Expected Bearer token");
    });
  });
});
