# Markdown renderer

`markdown-it-15.0.2.min.js` is the unmodified `dist/browser/markdown-it.umd.min.js` from the
`markdown-it@15.0.2` npm package (https://github.com/markdown-it/markdown-it).
The SHA-256 is recorded below. Use `npm pack markdown-it@15.0.2 --ignore-scripts`
to retrieve the release; no npm install or CDN is needed by generated projects.
`THIRD-PARTY-NOTICES.txt` contains the bundled dependencies' licenses.

Chat creates the renderer with `html: false`, `linkify: false`, and images disabled.
Keep those restrictions when updating it. Run `node --test tests/ui/chat-content.test.cjs`
from `integrations/agentbricks` after a version change.

SHA-256: `635972b985228e8af9f0143647c68616b7a3bb09f6946e7e4a52e43dcf5e7be5`
