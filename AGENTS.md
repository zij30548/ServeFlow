# ServeFlow development instructions

Read README.md before changing the application. ServeFlow is a customer-service
assistant with dialogue and monitoring surfaces. The former TripPilot application
is retired from this working tree; do not restore its planner or legacy UI.

- Backend: Python 3.12, FastAPI, `backend/`. Frontend: Vue/Vite, `frontend/`.
- Keep role routing, tool permissions, native tool feedback, strict contracts,
  cancellation, operation IDs, session isolation and bounded execution intact.
- Treat policy documents as reference material, business tools as record sources,
  and model text as unverified output. Unknown amounts are not zero.
- Skills under `backend/skills/` are runtime instructions, not disposable docs.
- Run relevant tests and frontend production build for affected changes. Use
  Python `-B`; ordinary tests must not contact real model or business services.
- Real model calls require explicit applicable authorization. Old trials and
  remaining paused allowances are not reusable budgets. Do not auto-retry unknown
  sends, rewrite old scores, or claim semantic quality from mocked checks.
- Keep credentials, databases, user data, logs, media and local environments out
  of Git. Do not alter existing private stores or initialize them automatically.
- Public documentation belongs in README.md and necessary source/license notices.
  Do not add resume packaging, interview scripts, a document center or unsupported
  claims about scale, accuracy or production readiness.
- Preserve applicable source and license notices. The maintainer confirmed full
  publication permission for this delivery; do not expand that into unsupported
  claims that every component was independently written from scratch.
- Keep changes scoped, inspect staged files, and use normal Git operations.
  Never force-push or deploy cloud services without explicit authorization.
