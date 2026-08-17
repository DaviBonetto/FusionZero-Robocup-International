# Vision Offline Tooling

## Objetivo

Este pacote entrega tres blocos operacionais para a visao oficial:

- validacao de `silver line` em pista real sem trocar o pipeline oficial
- replay offline usando o mesmo `VisionNode`
- captura organizada de edge cases para dataset incremental

O fluxo foi desenhado para rodar no PC sem camera fisica e continuar compativel com o Raspberry Pi 3.

## Silver Line Real

O detector oficial de `silver_line` agora combina:

- modelo oficial, quando disponivel
- heuristica controlada por config para faixa clara real
- janela de estabilidade com votos
- modo `manual` para auditoria sem acionar o sinal publico

Chaves principais em `configs/vision_config.json`:

- `detectors.silver_line.mode`
- `detectors.silver_line.decision_policy`
- `detectors.silver_line.stability_window`
- `detectors.silver_line.required_votes`
- `detectors.silver_line.specular_v_min`
- `detectors.silver_line.specular_s_max`
- `detectors.silver_line.top_black_ratio_max`

Leitura recomendada de metadata em `vision.detections`:

- `metadata.silver_line.found`
- `metadata.silver_line.confidence`
- `metadata.silver_line.bbox`
- `metadata.silver_line.heuristic`
- `metadata.silver_line.model`
- `metadata.silver_line.decision`

Para validacao real:

1. aplicar o perfil `silver_validation`
2. usar `Force LINE` no dashboard quando quiser congelar o fluxo operacional
3. manter `include_raw=true` no recorder para replay fiel
4. revisar `silver_line_mask` e `composite` no replay
5. ajustar thresholds apenas por config

## Replay Offline

Ferramenta: `src/tools/vision_replay.py`

Fontes suportadas:

- diretorio de frames
- diretorio de sessao (`events.jsonl` + `frames/`)
- video (`.mp4`, `.avi`, `.mov`, `.mkv`)

Defaults saem de `offline_ops.replay`:

- `output_root`
- `default_state`
- `frame_kind`
- `save_overlay_frames`
- `save_debug_views`

Observacao importante:

- `frame_kind=raw` e o caminho recomendado para reproduzir a pipeline oficial de ponta a ponta
- `frame_kind=processed` serve para diagnostico visual rapido, mas reprocessa uma imagem ja tratada

Exemplo:

```powershell
python New_AI/obr_overengineering_v1/src/tools/vision_replay.py `
  --config New_AI/obr_overengineering_v1/configs/vision_config.json `
  --source New_AI/obr_overengineering_v1/artifacts/session_recordings/session_x `
  --source-type session_dir
```

Saidas:

- `detections.jsonl`
- `overlays/*.jpg`
- `debug/<view>/*.jpg`

## Edge Dataset

Ferramenta base: `src/tools/vision_edge_dataset.py`

Estrutura gerada:

- `images/<label>/`
- `debug/<view>/<label>/`
- `metadata.jsonl`

Campos minimos por amostra:

- `label`
- `timestamp`
- `frame_id`
- `state`
- `source_path`
- `image_path`
- `debug_paths`
- `event`
- `metadata`

Views recomendadas para edge cases:

- `processed`
- `silver_line_mask`
- `composite`

Labels sugeridos:

- `silver_line_candidate`
- `silver_line_false_positive`
- `green_corner_border_case`
- `silver_black_overlap`

## Bundle de Debug para Calibracao Assistida

Backend oficial exposto em `VisionNode`:

- `get_last_processed_frame()`
- `get_last_debug_bundle()`

Contrato do bundle:

- `frame_id`
- `timestamp`
- `state`
- `metadata`
- `views`

Views atuais:

- `raw`
- `processed`
- `line_mask`
- `green_mask`
- `red_mask`
- `victim_mask`
- `silver_line_mask`
- `composite`

Para integracao do Agent C:

- consumir `metadata.debug_views_available` no evento para saber o que existe
- consumir `VisionNode.get_last_debug_bundle()` para obter as matrizes sem serializar `numpy` no event bus
- tratar `views` como opcionais e dependentes de `runtime.debug_artifacts_enabled`

## Validacao Recomendada

1. capturar sessao com perfil `silver_validation`
2. reproduzir com `vision_replay.py`
3. revisar `silver_line_mask` contra `top_black_ratio` e `decision.votes`
4. promover frames ruins para `dataset/edge_cases`
5. rerodar testes de visao antes de atualizar thresholds
