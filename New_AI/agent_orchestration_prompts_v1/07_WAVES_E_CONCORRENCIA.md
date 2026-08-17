# Ondas de Execucao (Simultaneo vs Separado)

## Wave 0 - Separado (bloqueante)
- Agent_StateMachine

Motivo:
- Este agente congela `docs/interfaces.md` v1. Sem contrato fechado, os outros agentes podem divergir.

## Wave 1 - Simultaneo
Rodar em paralelo:
- Agent_Vision_Optimization
- Agent_UI_Refactor
- Agent_Path_Render

Dependencia:
- Todos devem usar `interfaces.md` v1 da Wave 0.

## Wave 2 - Simultaneo parcial
Rodar em paralelo:
- Agent_RedColorCalibration
- Agent_Vision_Optimization (ajustes finais de integracao do red detector)

Dependencia:
- Preprocessador e pipeline base da Wave 1 disponiveis.

## Wave 3 - Separado (fechamento)
- Agent_TestValidation

Motivo:
- QA precisa validar o conjunto integrado.

## Regra fixa para TODAS as waves
- Trabalhar somente em `New_AI/obr_overengineering_v1`.
- Nao editar o restante do repositorio.
