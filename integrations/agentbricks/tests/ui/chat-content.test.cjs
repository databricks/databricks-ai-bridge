// Run with node --test tests/ui/chat-content.test.cjs; no npm install is needed.
const assert = require("node:assert/strict");
const path = require("node:path");
const { test } = require("node:test");

for (const framework of ["agent-openai", "agent-langgraph"]) {
  const ui = path.resolve(__dirname, "../../src/databricks_agentbricks/templates/ui", framework, "ui");
  globalThis.markdownit = require(path.join(ui, "vendor/markdown-it-15.0.2.min.js"));
  const { extractText, reasoningSummary, renderMarkdown, errorText } = require(path.join(ui, "chat-content.js"));

  test(`${framework}: typed text keeps opaque reasoning out of live and replayed messages`, () => {
    const content = [
      { type: "reasoning", text: "opaque-text", encrypted_content: "opaque-data", summary: [{ type: "summary_text", text: "A visible summary" }] },
      { type: "signature", text: "opaque-signature" },
      { type: "output_text", text: "**Answer**" },
      { type: "text", text: "with code" },
    ];
    assert.equal(extractText(content), "**Answer**\nwith code");
    assert.equal(reasoningSummary(content), "A visible summary");
    assert.equal(extractText({ encrypted_content: "opaque-data" }), "");
    assert.equal(extractText([{ type: "refusal", refusal: "Cannot answer" }]), "Cannot answer");
  });

  test(`${framework}: Markdown supports code, lists and tables without executing HTML or loading images`, () => {
    const html = renderMarkdown("# Heading\n\n**Bold** and `code`\n\n- one\n- two\n\n```python\nprint(1)\n```\n\n|a|b|\n|-|-|\n|1|2|");
    for (const expected of ["<h1>", "<strong>Bold</strong>", "<code>code</code>", "<ul>", "<pre>", "<table>"]) assert.ok(html.includes(expected), expected);
    const untrusted = renderMarkdown('<script>untrusted</script>\n\n[unsafe](javascript:untrusted)\n\n![image](https://example.com/image.png)');
    assert.ok(!untrusted.includes("<script>"));
    assert.ok(!untrusted.includes('href="javascript:'));
    assert.ok(!untrusted.includes("<img"));
    assert.ok(untrusted.includes("&lt;script&gt;"));
  });

  test(`${framework}: structured API failures are readable`, () => {
    assert.equal(errorText({ code: "MODEL_UNAVAILABLE", message: "Choose another model" }), "MODEL_UNAVAILABLE: Choose another model");
    assert.equal(errorText([{ msg: "session_id is required" }]), "session_id is required");
  });
}
