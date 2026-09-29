# Avaliação rápida de respostas e bloqueios

Esta versão reduz a avaliação a uma decisão por item. Ela mostra a pergunta, a resposta sugerida ao consumidor e se houve bloqueio. Nos bloqueios, mostra a última minuta e o motivo registrado pelo sistema. A aplicação detalhada em `scripts/run_response_review_app.py` continua disponível.

## Executar

Use Python 3.11 ou superior em macOS ou Linux, sem Docker nem chaves de API:

```sh
cd /caminho/para/mba-poc
python3 scripts/run_quick_response_review_app.py \
  --results runs/20260926T183838Z-c8c5b03b/results.json
```

Mantenha o terminal aberto e acesse <http://127.0.0.1:8083>.

Se a porta estiver ocupada, acrescente `--port 8084` e abra a porta 8084. Os caminhos relativos são resolvidos a partir da raiz do repositório.

## Avaliar

1. Informe seu nome uma vez.
2. Leia a pergunta e a resposta. Nos bloqueios, leia também o motivo informado.
3. Clique em **Resposta correta** ou **Resposta incorreta** quando houve entrega. Considere se o texto responde à solicitação e se as informações são corretas.
4. Clique em **Bloqueio correto** ou **Bloqueio incorreto** quando houve bloqueio. Considere se impedir a entrega da minuta foi uma decisão adequada. O fato de o sistema registrar uma falha de citação ou de revisão não prova, sozinho, que a minuta estava errada ou que o bloqueio era necessário.
5. Use **Em dúvida** quando não conseguir decidir. O item fica salvo para retomar depois. **Anterior** e **Próxima** trocam de item sem avaliar.

Em itens ainda não avaliados, **correto** fica pré-selecionado. **Enter** ou **Salvar e avançar** confirma essa seleção e abre o próximo item por avaliar. A pré-seleção não grava nada sozinha. Ao voltar a um item já avaliado, Enter mantém a classificação salva, inclusive incorreto ou em dúvida.

Use **←** e **→** para navegar para o item anterior ou seguinte na ordem da fila, respeitando os filtros, sem salvar. Por padrão, a fila mostra todos os itens para permitir voltar aos já avaliados. Nos campos de nome, busca e comentário, as teclas mantêm a edição normal; saia do campo para usar os atalhos. Manter Enter pressionado não avalia itens em sequência.

Cada clique em uma classificação salva e avança. O comentário é opcional em todos os casos. Você pode voltar pela fila e alterar a classificação. Filtros permitem concentrar a revisão em bloqueios, respostas entregues, dúvidas ou itens já avaliados.

A tela destaca `suggested_talk_track`, a minuta destinada ao consumidor. Os demais campos ficam em **Ver orientação completa ao consultor**, pois o bloqueio pode ter sido provocado por outra parte da saída. Fontes e referências também podem ser consultadas, sem preenchimento de rubricas.

A rodada indicada contém 400 itens: 360 respostas entregues e 40 minutas finais bloqueadas. A ferramenta avalia apenas o resultado final por caso/método, sem acrescentar tentativas iniciais reformuladas. Não executa modelos nem altera os dados experimentais.

## Salvar, retomar e exportar

As avaliações ficam em `local_reviews/quick_responses.json`, separado da aplicação detalhada e ignorado pelo Git. Reinicie com os mesmos argumentos para continuar. Faça backup desse arquivo. Para outro avaliador, use `--store local_reviews/quick_avaliador_b.json` e, se necessário, outra porta.

**Exportar** baixa `avaliacao-rapida.json` com as decisões, comentários, métodos, identificadores dos casos, motivos de bloqueio, hashes das entradas e histórico das alterações. A exportação separa:

- `target=response`: julgamento da correção da resposta entregue.
- `target=block`: julgamento da adequação do bloqueio.
- `correct`, `incorrect`, `uncertain` e `pending`: decisão humana ou ausência dela.

Dúvidas não contam como avaliações concluídas. Pendências não são classificadas automaticamente. A gravação mantém a proteção contra sobrescrita por abas ou processos concorrentes. Arquivos da revisão detalhada não podem ser reutilizados nesta versão.

## Interpretação no trabalho

Este é um julgamento global simplificado, com o estado de bloqueio e o motivo visíveis. Deve ser descrito dessa forma na metodologia. Não equivale à avaliação independente por fato esperado, não mede cobertura e não fornece uma medida de alucinação por afirmação. Os motivos automáticos podem influenciar a decisão humana.

Apresente as decisões sobre respostas e bloqueios separadamente, com seus denominadores e dúvidas. Um bloqueio correto não significa uma resposta correta entregue. A aplicação preserva a identificação dos métodos na exportação para permitir essa análise posterior, mas não transforma as marcações em confirmação automática de hipóteses.

## Testes

```sh
python3 -m unittest discover -s tests -p 'test_*response_review.py' -v
```
