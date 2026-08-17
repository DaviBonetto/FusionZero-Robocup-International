# Runtime, Boot and Recovery

## Objetivo

Deixar o pipeline oficial em `New_AI/obr_overengineering_v1` pronto para:

- install reproduzivel no Raspberry Pi 3
- execucao headless manual ou via `systemd`
- dashboard remoto no PC com reconnect automatico
- recovery rapido sem depender de paths manuais

## Artefatos

- env base:
  - `New_AI/obr_overengineering_v1/deploy/fusionzero.env.example`
- service template:
  - `New_AI/obr_overengineering_v1/deploy/fusionzero-live-dashboard.service.template`
- install/run Pi:
  - `New_AI/obr_overengineering_v1/scripts/install_pi.sh`
  - `New_AI/obr_overengineering_v1/scripts/run_pi_headless.sh`
  - `New_AI/obr_overengineering_v1/scripts/install_systemd_service.sh`
  - `New_AI/obr_overengineering_v1/scripts/runtime_status.sh`
- install/run PC:
  - `New_AI/obr_overengineering_v1/scripts/install_pc.ps1`
  - `New_AI/obr_overengineering_v1/scripts/run_pc_dashboard.ps1`

## Defaults seguros para Pi 3

Os defaults do arquivo `fusionzero.env.example` priorizam estabilidade:

- camera `640x480 @ 20 FPS`
- stream remoto `5 FPS`
- `JPEG quality 65`
- relay em `0.0.0.0:8765`
- serial em `115200`

Se o Pi 3 ainda ficar pesado, reduza primeiro:

- `FZ_CAMERA_WIDTH=320`
- `FZ_CAMERA_HEIGHT=240`
- `FZ_CAMERA_FPS=15`
- `FZ_REMOTE_STREAM_FPS=4`

## Install no Raspberry

No Raspberry, a partir da raiz do repo:

```bash
bash New_AI/obr_overengineering_v1/scripts/install_pi.sh
```

Isso faz:

- instala pacotes base via `apt`
- cria `New_AI/obr_overengineering_v1/.venv`
- instala dependencias Python base para runtime
- cria `New_AI/obr_overengineering_v1/deploy/fusionzero.env` se ainda nao existir

### Extras opcionais de ML

Se o ambiente tiver wheel compativel para `tflite-runtime`, rode:

```bash
bash New_AI/obr_overengineering_v1/scripts/install_pi.sh --with-optional-ml
```

Observacao operacional:

- o base install nao tenta instalar `torch/torchvision` automaticamente no Pi 3
- isso evita quebrar o bootstrap em ARMv7 por falta de wheel
- se voce precisar parity total das rotinas dependentes de Torch no Pi, trate isso como passo adicional controlado

## Configurar runtime no Raspberry

Edite `New_AI/obr_overengineering_v1/deploy/fusionzero.env`:

```bash
cp New_AI/obr_overengineering_v1/deploy/fusionzero.env.example \
   New_AI/obr_overengineering_v1/deploy/fusionzero.env
```

Campos mais importantes:

- `FZ_PROFILE=pi3_field`
- `FZ_RECORDINGS_ROOT=/path/opcional/para/artifacts/session_recordings`
- `FZ_ROBOT_SERIAL_PORT=/dev/ttyACM0`
- `FZ_ROBOT_DRY_RUN=1` para validar sem mover o robo
- `FZ_REMOTE_PORT=8765`

## Run manual no Raspberry

```bash
bash New_AI/obr_overengineering_v1/scripts/run_pi_headless.sh
```

Com env explicito:

```bash
bash New_AI/obr_overengineering_v1/scripts/run_pi_headless.sh \
  --env-file New_AI/obr_overengineering_v1/deploy/fusionzero.env
```

## Boot automatico via systemd

Instalar e habilitar:

```bash
sudo bash New_AI/obr_overengineering_v1/scripts/install_systemd_service.sh --enable-now
```

Com usuario/env explicitos:

```bash
sudo bash New_AI/obr_overengineering_v1/scripts/install_systemd_service.sh \
  --user pi \
  --env-file "$(pwd)/New_AI/obr_overengineering_v1/deploy/fusionzero.env" \
  --enable-now
```

Comandos operacionais:

```bash
sudo systemctl start fusionzero-live-dashboard.service
sudo systemctl stop fusionzero-live-dashboard.service
sudo systemctl restart fusionzero-live-dashboard.service
sudo systemctl disable --now fusionzero-live-dashboard.service
sudo systemctl enable fusionzero-live-dashboard.service
```

Status rapido:

```bash
bash New_AI/obr_overengineering_v1/scripts/runtime_status.sh
```

## Install no PC

No Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\New_AI\obr_overengineering_v1\scripts\install_pc.ps1
```

Isso cria por padrao um venv curto em `%LOCALAPPDATA%\FusionZero\venvs\obr_overengineering_v1-pc` e instala as dependencias do dashboard remoto.

O bootstrap do PC agora tambem instala `pytest`, para que a bateria oficial possa ser rodada no mesmo ambiente do dashboard.

## Abrir dashboard remoto no PC

```powershell
powershell -ExecutionPolicy Bypass -File .\New_AI\obr_overengineering_v1\scripts\run_pc_dashboard.ps1 -Host <IP_DO_RASPBERRY>
```

## Estrategia de recovery implementada

### Camera no Raspberry

- falhas isoladas de `read()` nao derrubam o processo
- ao atingir streak de falha, o runner libera e reabre a camera automaticamente
- quando a visao volta, o safety monitor limpa o fault e segue sem restart global

### Relay TCP do dashboard

- novo cliente remoto recebe `hello` de sessao
- ultimo snapshot de `FSM`, deteccoes, health, frames e logs recentes e replayado ao reconectar
- conexoes mortas sao fechadas por heartbeat timeout no servidor

### Cliente remoto no PC

- envia heartbeat periodico
- se ficar sem atividade do Raspberry por tempo acima do limite, fecha o socket e reconecta sozinho
- quando reconecta, recebe snapshot imediato e nao precisa esperar o proximo evento de estado

## Logs e troubleshooting

### No Raspberry

Seguir logs ao vivo:

```bash
sudo journalctl -u fusionzero-live-dashboard.service -f
```

Ultimas linhas:

```bash
sudo journalctl -u fusionzero-live-dashboard.service -n 80 --no-pager
```

### No dashboard remoto

O painel de logs passa a mostrar tambem:

- `connected to raspberry ...`
- `remote dashboard session ready ...`
- `remote dashboard disconnected ...`
- `remote dashboard heartbeat timeout ...`
- `camera ... read failure streak=...; reopening capture`

## Playbook rapido

### Camera caiu

1. aguarde o auto-reopen
2. confirme no log se apareceu `camera ... connected`
3. se a camera fisica foi trocada, use `camera reconnect` na UI ou reinicie o service

### Wi-Fi caiu

1. nao reinicie o Pi de imediato
2. o cliente do PC reconecta sozinho quando o relay voltar
3. confirme no log do PC `remote dashboard session ready`

### Service morreu

1. `systemctl status fusionzero-live-dashboard.service`
2. `journalctl -u fusionzero-live-dashboard.service -n 120`
3. `sudo systemctl restart fusionzero-live-dashboard.service`

### Serial caiu

1. confirme o cabo/porta em `/dev/ttyACM*` ou `/dev/ttyUSB*`
2. o adaptador serial continua com reconnect automatico
3. se mudou de porta, atualize `FZ_ROBOT_SERIAL_PORT` e reinicie o service
