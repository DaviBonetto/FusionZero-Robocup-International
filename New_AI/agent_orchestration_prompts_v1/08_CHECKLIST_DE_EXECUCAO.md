# Checklist rapido para rodar os agentes

1) Envie `00_GLOBAL_PROMPT.md` para todos os agentes.
2) Rode Wave 0 (StateMachine) e valide se `interfaces.md` v1 foi entregue.
3) Rode Wave 1 em paralelo (Vision + UI + Path).
4) Rode Wave 2 em paralelo (Red + ajuste Vision).
5) Rode Wave 3 (TestValidation).
6) Confirme que todos os artefatos estao apenas em `New_AI/obr_overengineering_v1`.
7) Bloqueie merge se testes criticos falharem.

Definition of Done (alinhado ao plano):
- Latencia E2E <= 50ms no cenario de referencia definido.
- Path 2D funcionando com gradiente e sem vazamento de memoria.
- Detector vermelho robusto em condicoes dificeis.
- Logs de transicao FSM aparecendo na UI e no terminal.
