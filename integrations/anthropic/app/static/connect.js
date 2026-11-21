// The sign-in link page. No inline script (the page's CSP is script-src 'self'); text is only written with
// textContent. The code comes from the link (?code=...): Databricks Apps' sign-in redirect keeps the query but drops a
// URL fragment, so the query is the only way it survives a first sign-in. The page then clears it from the address
// bar and history, and sends it back to this App only in request bodies.
(function () {
  "use strict";
  function $(id) { return document.getElementById(id); }

  function setStatus(text, cls) {
    var el = $("status");
    el.textContent = text || "";
    el.className = cls || "";
  }

  async function api(method, path, body) {
    var opts = { method: method, credentials: "same-origin", headers: {} };
    if (method !== "GET") {
      opts.headers["X-CMA-Request"] = "1";
      opts.headers["Content-Type"] = "application/json";
    }
    if (body !== undefined) opts.body = JSON.stringify(body);
    var resp = await fetch(path, opts);
    var data = await resp.json().catch(function () { return {}; });
    if (!resp.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : "Request failed (HTTP " + resp.status + ")");
    }
    return data;
  }

  var code = new URLSearchParams(window.location.search).get("code") || "";
  // Keep the code out of the address bar and history once read.
  if (window.history && window.history.replaceState) window.history.replaceState(null, "", "/connect");

  async function load() {
    if (!code) { $("intro").textContent = "This link has no sign-in code."; return; }
    try {
      var info = await api("POST", "/api/connect/info", { code: code });
      $("session").textContent = info.session_id;
      $("agent").textContent = info.agent || "(unknown)";
      $("user").textContent = info.signed_in_as;
      if (info.problem) {
        $("intro").textContent = "This session can't run on Databricks: " + info.problem + ".";
        return;
      }
      $("intro").textContent = info.already_yours
        ? "This session is already yours. Connect again to let its next tool call start a sandbox as you."
        : "Connect this Claude session to your Databricks identity.";
      $("details").hidden = false;
    } catch (e) {
      $("intro").textContent = e.message;
    }
  }

  $("confirm").addEventListener("click", async function () {
    $("confirm").disabled = true;
    setStatus("Connecting...");
    try {
      var data = await api("POST", "/api/connect", { code: code });
      setStatus(data.note, "ok");
      $("confirm").hidden = true;
    } catch (e) {
      setStatus(e.message, "error");
      $("confirm").disabled = false;
    }
  });

  load();
})();
