# Guia de Configuração Local - Ambiente e GitHub CLI

Este guia registra os passos de configuração do ambiente local necessários para permitir que o Agente AI gerencie fluxos complexos do GitHub (como assinalar issues a Projetos, Assignees, Labels e Milestones via automação). 

Ele foi configurado para ser ignorado pelo Git (`.gitignore`), garantindo que não suba para o repositório principal.

---

## 1. O Problema de Autenticação Inicial

Inicialmente, a automação baseada na API/CLI do GitHub (`gh`) esbarrava em erros de **permissão insuficiente**, especificamente o erro `GraphQL: Resource not accessible by personal access token`.

Isso ocorreu porque:
1. Havia uma variável de ambiente exportada (`GITHUB_TOKEN`) no terminal do Mac.
2. Este token primário tinha as permissões básicas (repositório), mas **não possuía os escopos avançados** (`project`, `read:org`, `read:discussion`) necessários para manipular Projetos (Projects V2) do GitHub.

## 2. Como resolvemos (A Configuração Local)

Para habilitar a automação completa de Pull Requests, seguimos a seguinte reautenticação local:

1. **Remoção de Variáveis Restritas**:
   O GitHub CLI avisa quando você tenta reautenticar enquanto a variável `GITHUB_TOKEN` está definida (`The value of the GITHUB_TOKEN environment variable is being used...`). Para a CLI funcionar com permissões expandidas, foi necessário desvincular ou atualizar as credenciais via navegador.

2. **Renovação de Permissões (Scopes) da CLI**:
   Rodamos o comando oficial para renovar e adicionar permissões à chave local da sua máquina:
   ```bash
   gh auth refresh -s project,read:org,read:discussion
   ```
   *Isso abriu o navegador, onde você aprovou e vinculou seu usuário (`Cristiano2132`), garantindo que qualquer script ou Agente rodando no seu terminal agora tenha poder para injetar cards em projetos.*

## 3. Padrão de Fechamento de PRs Consolidado

Com a permissão garantida, nós consolidamos que a **Criação de PRs e Fechamento de Issues** agora ocorre de maneira profissional:

1. **Geração do PR**: O agente cria o PR usando a estrutura estrita descrita no repositório (`.github/PULL_REQUEST_TEMPLATE.md`).
2. **Injeção de Metadados via CLI**: Após a criação do PR, executamos o GitHub CLI para enriquecer o Pull Request usando os mesmos dados da Issue original:
   ```bash
   gh pr edit <PR_NUMBER> \
     --add-assignee "@me" \
     --milestone "M2 - Evaluation Module" \
     --add-label "type: feature,module: evaluation,priority: critical" \
     --add-project "quadro-de-tarefas"
   ```

## 4. Pastas de Automação Ignoradas

Por fim, toda a inteligência do agente, guardada na pasta oculta `.agent/`, bem como este arquivo de guia (`LOCAL_SETUP_GUIDE.md`), foram adicionadas explicitamente ao `.gitignore`. 
Caso a pasta `.agent/` estivesse rastreada, o comando executado foi:
```bash
git rm -r --cached .agent/
```
Isso desvincula arquivos de uso local do versionamento central, mantendo os merges limpos apenas com o código fonte (Python) do `model-track-cr`.
