# Pacote de reprodução da rodada de 26/09/2026

Este repositório contém os arquivos necessários para conferir os resultados quantitativos apresentados no TCC, sem chamadas a modelos nem banco de dados. A reprodução usa a rodada preservada de 26/09/2026 e os julgamentos humanos existentes.

## Conteúdo

- `src/`, `sql/`, `pyproject.toml`, `uv.lock`, `Dockerfile` e `compose.yaml`: implementação dos quatro métodos e ambiente da execução.
- `data/cases_reviewed.jsonl`: os 100 casos sintéticos revisados usados na rodada.
- `data/public_corpus.json` e `data/publication_manifest.json`: os 86 trechos dos dez documentos autorizados e o recorte de publicação.
- `runs/20260926T183838Z-c8c5b03b/results.json`: 400 registros completos, com os rastros da execução e o manifesto de hashes.
- `output/revisao/avaliacao-humana-simplificada-2026-09-27.json`: 400 julgamentos do autor, com histórico de 407 gravações. O esclarecimento posterior sobre o significado dos bloqueios está no JSON ao lado.
- `latex/`: scripts de análise da monografia, tabelas e figuras regeneradas, além dos resumos numéricos.
- `manifest.json`: hashes SHA-256 dos arquivos de entrada e das saídas esperadas.

O manifesto da execução contém os hashes de `data/cases.jsonl` e `data/cases_legacy_134.jsonl`, dois conjuntos antigos de 134 casos que o executor registrou entre os arquivos de origem. A rodada examinada recebeu explicitamente `data/cases_reviewed.jsonl`; por isso, os arquivos antigos foram excluídos deste recorte. Seus hashes originais permanecem no manifesto da execução. Os demais 40 arquivos ali listados estão presentes e são conferidos pelo gerador.

## Conferir os resultados

Na raiz deste diretório, com Python 3.11 ou posterior:

```sh
python3 scripts/reproduce.py
```

O verificador confere os hashes do pacote, os 100 casos, 400 registros, 400 julgamentos e a vinculação entre os três arquivos de entrada; regenera as tabelas, figuras e resumos; e compara seus hashes com as saídas da versão revisada do TCC. Usa somente a biblioteca padrão de Python. Os arquivos `latex/generated/*.tex` podem ser incluídos no LaTeX da monografia, mas a compilação do PDF exige o projeto TCC e sua classe institucional.

O pacote permite recalcular as análises da rodada preservada. Executar novamente os quatro métodos exigiria serviços externos, configuração do provedor e seus custos; uma nova execução não reproduziria necessariamente as mesmas respostas ou latências.

## Alcance da avaliação

Os julgamentos globais foram feitos por um único autor, que via o motivo de bloqueio. `incorrect` em um bloqueio indica minuta julgada adequada para entrega; `correct` não classifica automaticamente a qualidade factual da minuta. Os 60 casos com entrega comum são um subconjunto selecionado. Os intervalos, testes e latências são descritivos ou exploratórios nas condições documentadas na monografia.
