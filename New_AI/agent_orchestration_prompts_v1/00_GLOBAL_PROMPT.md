# Prompt Global (enviar para todos)

```text
Contexto comum para todos os agentes:
- Objetivo final: criar uma nova arquitetura OBR desacoplada em New_AI, com PyQt6 + Event Bus tipado + Vision orientada por FSM, focada em Raspberry Pi 4 (4GB).
- Entregavel final esperado: sistema em `New_AI/obr_overengineering_v1` sem alterar os arquivos atuais que ja funcionam.

Restricoes obrigatorias:
1) Trabalhar somente em `New_AI/obr_overengineering_v1`.
2) Nao editar nem mover `pc_vision_ui.py` e `pc_vision_runner.py`.
3) Priorizar precisao + desempenho no Pi 4.
4) Evitar bloqueio da thread da UI.
5) Reutilizar ao maximo os detectores/modelos do FusionZero.

Reuso obrigatorio (referencia):
- LineDetector, BallDetector, ColorMarkerDetector de `pc_vision_runner.py`.
- Estrategia de status/FPS/composicao da `pc_vision_ui.py`.
- SilverLineDetector de `1_international/behaviours/silver_detection.py`.
- Modelos: silver_detector_pi4_quantized.pt e dead.tflite.

Direcao visual obrigatoria:
- Dashboard escuro estilo Overengineering.
- Duas views no topo (raw/processado), telemetria, estado textual curto, timer principal grande, bloco central de robo placeholder.

Regra de comunicacao:
- reportar progresso em blocos curtos.
- listar bloqueios explicitamente.

Formato de status obrigatorio:
Agent: <agent_name>
Status: <not_started|in_progress|blocked|done>
Completed:
- <item>
Next:
- <item>
Blockers:
- <item_or_none>
Artifacts:
- <file_or_result>
```
