# Raspberry Pi 3 + PC Dashboard

## Objetivo

Rodar camera, visao, FSM e integracao serial no Raspberry Pi 3 e usar o PC apenas como console remoto de operacao, tuning e observabilidade.

Guias relacionados:

- `New_AI/obr_overengineering_v1/docs/runtime_boot_recovery.md`
- `New_AI/obr_overengineering_v1/docs/ops_observability.md`
- `New_AI/obr_overengineering_v1/docs/pi3_arduino_integration.md`

## Arquitetura oficial

- Raspberry Pi 3:
  - captura camera
  - roda visao e FSM
  - conversa com o Arduino via serial
  - injeta telemetria de controle no stream remoto
  - abre o relay TCP para o dashboard
- Arduino:
  - fecha o loop de movimento
  - publica `TLM ...`
- PC:
  - conecta no relay TCP
  - mostra frames, estado, logs e telemetria
  - envia `UI_COMMAND` de operador

## O que o dashboard mostra agora

### Visao e estado

- `Processed View`
- `Raw View`
- badges de estado e deteccao
- logs do sistema
- estado FSM

### Telemetria de controle

Quando a serial esta ativa, o runner injeta os campos do Arduino em `metadata["control"]` e `metadata["serial"]`.

Campos mais uteis:

- `control_mode`
- `line_error`
- `pid_output`
- `vision_confidence`
- `obstacle_state`
- `green_instruction`
- `failsafe`
- `heartbeat_age_ms`
- `telemetry_age_ms`
- `assist_kind`

### Overlay operacional

O overlay do `Processed View` pode exibir:

- `VC xx%`
- `E +/-x.xx`
- `PID +/-x.x`
- `OBSTACLE ...`
- `GREEN ...`
- `FAILSAFE`

## Comandos remotos disponiveis

Comandos da UI que continuam ou foram ampliados:

- `reconnect camera`
- ajustes de tuning
- `Forward test`
- `STOP`
- `Force STOP`
- `Clear ESTOP`
- `Obstacle test`
- `Clear obstacle`
- `Force LINE`
- `Force RESCUE`

Mapeamento relevante:

- `robot.forward_test`
- `robot.stop`
- `robot.force_stop`
- `robot.clear_estop`
- `robot.obstacle_test`
- `robot.obstacle_clear`
- `fsm.force_mode`

## Bootstrap recomendado no Raspberry

```bash
bash New_AI/obr_overengineering_v1/scripts/install_pi.sh
bash New_AI/obr_overengineering_v1/scripts/run_pi_headless.sh
```

Recomendacoes para Pi 3:

- `FZ_PROFILE=pi3_field`
- `FZ_CAMERA_WIDTH=640`
- `FZ_CAMERA_HEIGHT=480`
- `FZ_CAMERA_FPS=20`
- `FZ_REMOTE_STREAM_FPS=5`
- `FZ_REMOTE_JPEG_QUALITY=65`

Se precisar aliviar mais:

- reduzir para `320x240`
- reduzir `camera-fps` para `15`
- reduzir `remote-stream-fps` para `4`

## Comando no PC

Troque `<IP_DO_RASPBERRY>` pelo IP real do Pi na rede local.

```powershell
powershell -ExecutionPolicy Bypass -File .\New_AI\obr_overengineering_v1\scripts\install_pc.ps1
powershell -ExecutionPolicy Bypass -File .\New_AI\obr_overengineering_v1\scripts\run_pc_dashboard.ps1 -Host <IP_DO_RASPBERRY>
```

## Reconnect implementado

- camera:
  - o runner detecta streak de falha e reabre a camera
- relay TCP:
  - conexoes mortas sao descartadas por heartbeat
  - snapshot recente e logs podem ser replayados
- cliente remoto:
  - o PC reconecta sozinho quando detecta timeout
- serial:
  - o adaptador tenta reabrir a porta e reprobar handshake com `PING/PONG`

## Teste rapido recomendado

1. descobrir o IP do Raspberry com `hostname -I`
2. subir o runner headless no Raspberry
3. abrir o dashboard no PC
4. confirmar frames e logs em tempo real
5. validar `Forward test`
6. validar `Obstacle test`
7. validar `Force STOP`
8. confirmar telemetria `mode`, `line_error`, `pid`, `failsafe`

## Requisitos de rede

- Raspberry e PC na mesma rede local
- rede estavel vale mais que banda alta
- hotspot pode funcionar, mas roteador dedicado tende a ser mais consistente

## Observacao importante

Se a serial estiver indisponivel, o dashboard ainda funciona com fallback de telemetria sintetica. Quando a serial volta, a telemetria do Arduino passa a ser a fonte preferencial.
