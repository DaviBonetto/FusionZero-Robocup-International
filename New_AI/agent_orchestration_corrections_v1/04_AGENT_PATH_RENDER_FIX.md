# Prompt de Correção Pós-Implementação: Estouro de Memória do Path

**Agente Responsável:** `Agent_Path_Render`
**Modo Ativo:** Debug & Fix
**Arquivos Alvo:** `src/ui_overengineering/components/path_canvas.py` (ou componente que controla o 2D Plot das Rodas).

## Problema Identificado (Memory Leak)

Após rodadas longas simuladas ou na competição (acima de 3-4 minutos), o histórico de posições preenche indefinidamente o Canvas e Arrays, derrubando a interface Qt ou PyGame e travando toda a memória da placa (OOM/Buffer Overflow).

## Tarefa de Correção

> ATUAÇÃO: Agent_Path_Render. MODO: Debug & Fix.
> TAREFA: Refatorar o componente renderizador do caminho (Path 2D). Você não pode permitir o uso de Lists puras (`[]`) crescendo infinitamente para armazenar coordenadas (X, Y).
>
> 1. Troque o array infinito por um buffer rotatório estático importando `collections.deque(maxlen=1000)` ou tamanho que julgar estável para UI.
> 2. Se o robô ficar bloqueado/parado na pista (Ex: `Distância entre nova coordenada e coordenada[-1] < 1 pixel`), NÃO a adicione na `deque` (Implemente um cálculo rápido de Distância Euclidiana antes do `.append()`).
> 3. No loop de renderização (PaintEvent), desenhe apenas os segmentos de linha unidos de uma vez no `QPainter` a partir desse Deque estático, mitigando 100% de memory leak de desenho por retenção indiscriminada.
