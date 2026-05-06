# AGENTS.md

> A concise cheat‑sheet to avoid common pitfalls and speed up development.

## Environment
- Python ≥ 3.10 (project uses 3.10‑3.12; 3.10 is the minimal version).
- **Poetry** is used for dependency and virtual‑environment management.

## Installation
```bash
# Install deps and create the virtual env
poetry install
```
Use `poetry shell` if you prefer to work interactively.

## Running tests
The repository ships a small Makefile.
```bash
make test  # runs pytest with coverage built into pyproject.toml
make cov   # runs tests and outputs an XML coverage report
```
Alternatively, run directly:
```bash
poetry run pytest tests/
```
Coverage is enforced at 90 % (`--cov-fail-under=90`). If it falls below, CI will fail.

## Linting & type checking
```bash
poetry run ruff check .      # lint + format
poetry run mypy src           # strict type check
```
`ruff` also applies isort; it knows that `model_track` is a first‑party package.

## Security audit
```bash
poetry run pip-audit
```
The CI skips `CVE‑2026‑4539`.

## Common pitfalls
- The lock file (`poetry.lock`) may be missing. Run `poetry lock` after a fresh install.
- Tests live in `tests/`; running from the repo root is fine because `pythonpath` is set in `pyproject.toml` to include `src`.

## CI details
- Triggers on pushes or PRs to `main`/`develop` and on new tags `v*`.
- Tests run on Python 3.10‑3.13, lint on the default test job.
- The `publish` job is gated on a `v*` tag and expects the package to build and pass all checks.

## Documentation & workflow
The README contains the full library overview and usage examples.
For contribution guidelines and the mandatory testing checklist, see [CONTRIBUTING.md](CONTRIBUTING.md).

### Workflows de Issue (Mandatory)

Use os workflows automatizados abaixo para garantir o rito completo:

| Slash Command | Quando usar | Arquivo |
|---|---|---|
| `/issue-start <number>` | Ao iniciar qualquer issue | `.agent/workflows/issue-start.md` |
| `/issue-close <issue> <pr>` | Após merge de um PR | `.agent/workflows/issue-close.md` |

### Pull Request Protocol (Mandatory)

Ao criar um Pull Request, é **obrigatório** usar o template em **`.agent/templates/pr_description.md`** e garantir os seguintes metadados:
- **Projeto**: Associar ao projeto correto do repositório.
- **Milestone**: Deve corresponder à milestone definida na issue original.
- **Labels**: Adicionar labels de tipo (ex: `type: feature`, `type: bug`) e módulo (ex: `module: stats`).

**Seções obrigatórias da descrição do PR** (nunca omitir):

| Seção | O que colocar |
|-------|--------------|
| `📝 Contexto` | Por que o PR existe; link para a issue. |
| `🛠️ Implementação` | O que foi feito e decisões de design. |
| `🛡️ Risco & Rollback` | Nível de risco (`Baixo/Médio/Alto`), breaking changes, como reverter. |
| `🧪 Impacto & Testes` | Cobertura, tipos de teste, como validar. |
| `🔗 Vínculos` | Issue, Milestone, Projeto. |
| `✅ Checklist` | Acceptance criteria, testes, lint, cobertura. |
| `🖥️ Evidência Técnica` | Output de teste ou execução que comprova o funcionamento. |

**Fluxo completo:**
```
/issue-start 46   →  branch criada, plano aprovado, implementação
                  →  commits incrementais durante o trabalho
                  →  PR criado com template completo + Projeto, Milestone e Labels.
/issue-close 46 71 →  checkboxes, labels, comentário, branch cleanup
```

> Não execute estes passos manualmente. Use sempre os workflows acima para garantir consistência.

