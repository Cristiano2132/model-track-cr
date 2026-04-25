---
name: git-workflow
description: Automates and enforces the standard Git/GitHub workflow for the model-track-cr project.
---

# Git Workflow Skill

Este skill define e automatiza o fluxo de trabalho padrão do projeto no GitHub. Ele garante que cada alteração de código esteja vinculada a uma Issue, possua uma branch dedicada e siga os padrões de documentação e testes.

## 🚀 O Fluxo Padrão

Todo desenvolvimento deve seguir estas 4 fases:

1. **Discovery & Issue**:
   - Identificar a tarefa.
   - Criar ou localizar a Issue no GitHub.
   - **MANDATÓRIO**: Mover manualmente o card para **In Progress** no GitHub Project Board.
   - **MANDATÓRIO**: Adicionar comentário na Issue informando o início.

2. **Branching**:
   - Criar uma branch seguindo o padrão `feature/{issue_number}-{slug}` ou `fix/{issue_number}-{slug}`.
   - Nomear de forma concisa.

3. **Implementation & Tests**:
   - Codar seguindo o `CONTRIBUTING.md`.
   - Garantir as 4 dimensões de testes (Unit, PBT, Integration, Benchmark).

4. **Pull Request & Closure**:
   - Criar PR com descrição detalhada (Contexto, Mudanças, Verificação).
   - Vincular a PR à Issue (ex: `Closes #123`).
   - **MANDATÓRIO**: Marcar todos os checkboxes (Acceptance Criteria) na Issue original antes de fechar.
   - **MANDATÓRIO**: Adicionar observações finais na Issue se houver detalhes técnicos relevantes.
   - Mover a Issue para "Done" ou fechar após o merge.
   - **MANDATÓRIO**: Perguntar ao usuário se deseja excluir a branch temporária (local e origin) após a conclusão.


## 🛠 Comandos de Automação (Scripts)

### 1. Iniciar Task (`start_task.py`)
Automatiza a criação/comentário da Issue e criação da branch local.
```bash
python .agent/skills/git-workflow/scripts/start_task.py --issue 62 --title "Refactor Context"
```

### 2. Finalizar Task (`finish_task.py`)
Automatiza a criação da PR e o fechamento da Issue.
```bash
python .agent/skills/git-workflow/scripts/finish_task.py --issue 62
```

## 📏 Regras de Ouro (MANDATÓRIO)

- **Sempre mover o card no Board para 'In Progress' antes de codar.**
- **Nunca codar na `develop` ou `main` diretamente.**
- **Sempre comentar na Issue antes de começar.**
- **Sempre rodar `make test` antes de abrir a PR.**
- **PRs sem checklist de testes preenchido serão REJEITADAS.**
