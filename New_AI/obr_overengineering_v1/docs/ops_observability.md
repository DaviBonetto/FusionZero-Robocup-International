# Dashboard Ops e Observability

## Objetivo

Transformar o dashboard em uma ferramenta de operacao para bancada, Raspberry Pi 3 e depuracao de sessoes reais.

Os recursos novos cobrem:

- health indicators operacionais
- perfis reutilizaveis de configuracao
- session recording para debug
- calibracao assistida com freeze e mascaras

## Health Indicators

O painel `Ops Health` e a barra superior passam a exibir:

- CPU e memoria
- FPS de captura e processamento
- profundidade da fila do event bus
- latencia de rede no dashboard remoto
- estado da camera
- estado da serial
- status de undervoltage/throttling do Raspberry Pi, quando `vcgencmd` estiver disponivel
- perfil ativo
- status do session recorder

Leitura rapida recomendada:

- `ONLINE` ou `CONNECTED`: operacao normal
- `DEGRADED` ou `WARN`: ainda operacional, mas com risco de degradacao
- `OFFLINE`, `ERROR` ou `TIMEOUT`: tratar como bloqueio operacional

## Perfis

Os perfis ficam em `configs/vision_config.json`, no bloco `dashboard_ops.profiles`.

Perfis entregues:

- `lab_pc`
- `pi3_field`
- `rescue_test`
- `line_only`
- `silver_validation`

Cada perfil pode ajustar:

- camera
- tuning do dashboard/runner
- configuracao padrao do recorder

Como aplicar:

1. abrir o grupo `Profiles`
2. selecionar o preset
3. clicar `Apply profile`

Observacoes:

- o perfil nao reescreve a FSM
- `line_only` e um preset operacional de tuning; ele nao desliga detectores no backend
- `silver_validation` liga recorder mais denso e captura `raw` + `processed` para replay fiel
- se quiser forcar o fluxo da FSM, use `Force LINE` ou `Force RESCUE`

## Session Recording

O recorder grava:

- `UI_COMMAND`
- estado e transicoes da FSM
- deteccoes
- health events
- logs
- amostras de frame cru/processado, se habilitadas

Saida:

- raiz padrao: `artifacts/session_recordings/`
- um diretorio por sessao
- `events.jsonl`
- `manifest.json`
- `frames/*.jpg` para amostras

Fluxo recomendado:

1. escolher o perfil
2. abrir `Session Recording`
3. configurar `Sample raw frames`, `Sample processed frames` e `Every N frames`
4. clicar `Start / apply`
5. reproduzir o problema
6. clicar `Stop`

Para `silver line` real:

1. aplicar `silver_validation`
2. manter `Sample raw frames` ligado
3. usar `Force LINE` apenas quando precisar congelar o modo operacional
4. replayar a sessao com `src/tools/vision_replay.py`

Quando usar:

- regressao intermitente
- tuning em campo
- falha de serial
- perda de camera
- rescue com comportamento dificil de reproduzir

## Calibracao Assistida

O grupo `Calibration` foi desenhado para tuning auditavel.

Recursos:

- `Freeze current frame`
- visoes `Raw frame`, `Processed view`, `Line mask`, `Green mask`, `Red mask`, `Victim mask` e `Composite debug`
- backend oficial tambem expõe `silver_line_mask` via `VisionNode.get_last_debug_bundle()`
- `Audit snapshot`, que publica um snapshot com:
  - modo de visualizacao
  - estado de freeze
  - perfil ativo
  - tuning atual
  - ultimo metadata de visao conhecido

Fluxo recomendado:

1. selecionar o perfil mais proximo do ambiente
2. alternar a mascara relevante
3. ajustar thresholds observando a resposta visual
4. congelar o frame quando encontrar um caso ruim
5. usar `Audit snapshot`
6. se necessario, iniciar um session recording antes de repetir o experimento

Observacao:

- no backend oficial, as mascaras de calibracao e replay ficam fora do event bus para manter o contrato JSON do dashboard remoto

## Operacao Remota

No dashboard remoto:

- a latencia de rede e anexada ao `HealthEvent` pelo cliente TCP
- o painel de health continua usando o mesmo fluxo de eventos do dashboard local
- perfis, comandos de recorder e snapshots de calibracao continuam trafegando via `UI_COMMAND`

## Limitacoes Conhecidas

- se `vcgencmd` nao estiver disponivel, o status de power fica como `N/A`
- o ambiente atual de desenvolvimento pode nao ter `numpy`, `cv2`, `PyQt6` ou `pytest`; nesse caso a validacao completa precisa ser feita no ambiente do projeto
- o recorder amostra frames; ele nao tenta gravar video continuo
