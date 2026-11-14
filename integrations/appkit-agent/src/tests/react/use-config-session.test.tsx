// @vitest-environment happy-dom
import { describe, it, expect } from "vitest";
import { renderHook } from "@testing-library/react";
import { useConfig } from "../../react/hooks/use-config";
import { useSession } from "../../react/hooks/use-session";
import { createWrapper } from "./test-helpers";

describe("useConfig", () => {
  it("returns features and config from context", () => {
    const { result } = renderHook(() => useConfig(), {
      wrapper: createWrapper({
        apiBase: "/custom/api",
        basePath: "/app",
        features: { chatHistory: false, feedback: true },
      }),
    });

    expect(result.current.apiBase).toBe("/custom/api");
    expect(result.current.basePath).toBe("/app");
    expect(result.current.features).toEqual({
      chatHistory: false,
      feedback: true,
    });
    expect(result.current.isLoading).toBe(false);
  });

  it("reflects loading state", () => {
    const { result } = renderHook(() => useConfig(), {
      wrapper: createWrapper({ isLoading: true }),
    });
    expect(result.current.isLoading).toBe(true);
  });

  it("returns default features from context", () => {
    const { result } = renderHook(() => useConfig(), {
      wrapper: createWrapper(),
    });
    expect(result.current.features).toEqual({
      chatHistory: true,
      feedback: false,
    });
  });
});

describe("useSession", () => {
  it("returns user from session in context", () => {
    const session = {
      user: { email: "test@example.com", name: "Test User" },
    };
    const { result } = renderHook(() => useSession(), {
      wrapper: createWrapper({ session }),
    });
    expect(result.current.user).toEqual(session.user);
  });

  it("returns null user when session is null", () => {
    const { result } = renderHook(() => useSession(), {
      wrapper: createWrapper({ session: null }),
    });
    expect(result.current.user).toBeNull();
  });

  it("returns null user when session.user is null", () => {
    const { result } = renderHook(() => useSession(), {
      wrapper: createWrapper({ session: { user: null } }),
    });
    expect(result.current.user).toBeNull();
  });

  it("reflects loading state", () => {
    const { result } = renderHook(() => useSession(), {
      wrapper: createWrapper({ isLoading: true }),
    });
    expect(result.current.isLoading).toBe(true);
  });
});
