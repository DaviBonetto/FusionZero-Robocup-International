# Prompt de Correção Pós-Implementação: Otimização Condicional e FSM

**Agentes Responsáveis:** `Agent_StateMachine` & `Agent_Vision_Optimization`
**Modo Ativo:** Debug & Integration Fix
**Arquivos Alvo:** `src/core/state_machine.py` e `src/modules/vision/pipelines.py`

## Problema Identificado (O Gargalo de Processamento)

A YOLO (ou outro modelo pesado de segmentação/detecção de círculos) continua rodando a todo instante, até mesmo quando o robô está apenas seguindo linha de forma segura. Isso está drenando os recursos ($CPU/GPU$) da Raspberry Pi / Computador Embarcado.

## Tarefa de Correção

> ATUAÇÃO: Agent_Vision_Optimization e Agent_StateMachine. MODO: Debug & Integration Fix.
> TAREFA: A arquitetura atual de FSM não está cortando funções pesadas dinamicamente. Refatore `vision/pipelines.py` injetando uma leitura rigorosa do evento (getter) `current_state()` da Máquina de Estados.
>
> **Regra Absoluta:**
>
> - Se `estado == SEARCHING_LINE` ou `FOLLOWING_LINE` ou `VALIDATING_GAP` -> RETORNE IMEDIATAMENTE (Return early) na função _antes_ de carregar o modelo neural de resgate (YOLO/HoughCircles). Apenas rode Binarização Preta para detecção de linha.
> - Se `estado == RESCUE_ZONE_DETECTED` -> Desative a máscara de linha negra (Binarização), e aloque 100% dos recursos visuais no tracking de Vitimas Vivas (Prata) e Mortas (Preta).
>
> Mande que a mudança severa de pipeline de processamento da matriz de imagem dispare um Log robusto (`[INFO] CV_PIPELINE SWITCHED TO RESCUE_ZONE`) demonstrando a economia de pipeline no buffer.
