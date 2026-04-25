# 🛠️ Development Workflow

Este documento descreve o fluxo de trabalho adotado para o desenvolvimento da biblioteca **model-track-cr**, garantindo consistência, rastreabilidade e qualidade técnica.

## 🌳 Gitflow & Branching Strategy

Adotamos uma versão pragmática do Gitflow:

- **`main`**: Branch de produção. Contém apenas código estável e tagueado com versões (`v*`).
- **`develop`**: Branch principal de integração. Todos os novos desenvolvimentos devem ser integrados aqui primeiro.
- **`feature/*`**: Branches para novas funcionalidades (baseadas em `develop`).
- **`bugfix/*`**: Branches para correções de bugs (baseadas em `develop`).
- **`hotfix/*`**: Branches para correções críticas em produção (baseadas em `main`).

### Convenção de Commits
Usamos [Conventional Commits](https://www.conventionalcommits.org/):
- `feat`: Nova funcionalidade
- `fix`: Correção de bug
- `docs`: Alterações na documentação
- `test`: Adição ou refatoração de testes
- `refactor`: Alteração de código que não corrige bug nem adiciona feature

## 📝 Pull Request Standards

Todos os merges para `develop` ou `main` devem ocorrer via Pull Request (PR) seguindo o template padronizado em `.github/PULL_REQUEST_TEMPLATE.md`.

### Seções Obrigatórias:
1.  **Summary**: Explicação clara do problema e da necessidade.
2.  **Changes**: Lista técnica do que foi alterado.
3.  **Test Plan**: Evidências de validação manual e automatizada.
4.  **Risk & Rollback**: Avaliação de risco e plano de contingência.

## 🎯 Task & Issue Tracking

- **Milestones**: Agrupam issues por marcos de entrega (Ex: `M1 - Foundation Fixes`).
- **Issues**: Devem conter critérios de aceitação claros e UML (quando necessário).
- **Labels**:
  - `type:` (feature, bug, task, docs)
  - `module:` (base, woe, stats, etc.)
  - `status:` (triage, ready, in progress, blocked)

## 🛡️ Quality Gate (CI/CD)

O pipeline (`ci.yml`) é disparado em todo PR e exige:
1.  **Linting**: Ruff (sem erros).
2.  **Type Checking**: MyPy (strict mode).
3.  **Security**: Bandit e Pip-audit.
4.  **Tests**: Pytest com **cobertura mínima de 90%**.

---
*Este workflow foi desenhado para escalar a colaboração mantendo o rigor técnico exigido em projetos de risco e crédito.*
