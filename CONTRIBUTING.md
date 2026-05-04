# Guia de Contribuição - model-track-cr

Este projeto adota padrões rigorosos de qualidade para garantir a estabilidade estatística e técnica da biblioteca. Todas as contribuições devem seguir este guia.

## 🧪 Padrão de Testes (Checklist Multi-Dimensão)

Antes de abrir um Pull Request para qualquer refatoração ou nova funcionalidade, você **deve** garantir que as seguintes dimensões de testes estão cobertas:

### 1. Testes Unitários (Unit Tests)
- [ ] Cobertura de 100% das linhas do novo código/modificado.
- [ ] Teste de caminhos de falha (exceções) com `pytest.raises`.
- [ ] Mocking de dependências externas (ex: I/O, rede).
- **Pasta:** `tests/unit/`

### 2. Testes Estatísticos (Property-Based Testing - PBT)
- [ ] Uso de `Hypothesis` para validar invariantes matemáticas.
- [ ] Teste com DataFrames extremos (NaNs, constantes, outliers).
- [ ] Verificação de sanidade estatística (ex: correlações entre -1 e 1).
- **Pasta:** `tests/statistical/`

### 3. Testes de Integração (Integration Tests)
- [ ] Fluxo completo "ponta a ponta" (ex: fit -> transform -> save -> load).
- [ ] Verificação de compatibilidade com o `ProjectContext`.
- **Pasta:** `tests/integration/`

### 4. Benchmarks de Performance
- [ ] Medição de tempo para $N=10^6$ linhas.
- [ ] Garantia de que a complexidade não exceda $O(n)$ ou $O(n \log n)$ onde esperado.
- **Pasta:** `tests/benchmarks/`

---

## 📐 Regras de Ouro
1. **Backward Compatibility**: Nenhuma mudança deve quebrar a carga de um `ProjectContext` gerado por versões anteriores sem um aviso de depreciação claro e caminho de migração.
2. **Defesa em Profundidade**: Se o código roda mas o resultado matemático é duvidoso (ex: IV negativo), o teste deve falhar.
3. **Documentação**: Atualize os docstrings (padrão Google/NumPy) e o `README.md` se a API pública for alterada.
4. **GitFlow**: Trabalhe sempre em branches de `feature/` ou `refactor/` a partir da `develop`.
5. **PR Metadata**: Todo Pull Request deve obrigatoriamente estar vinculado a um **Projeto**, possuir a **Milestone** correspondente e os **Labels** de tipo/módulo.

---

## 🚀 Como Executar
```bash
# Rodar todos os testes
make test

# Verificar cobertura detalhada
make cov
```
