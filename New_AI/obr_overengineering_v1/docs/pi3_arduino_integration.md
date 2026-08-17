# Raspberry Pi 3 + Arduino + Dashboard remoto

Guia curto de bancada para a fase atual de comunicacao Pi <-> Arduino:

- `New_AI/obr_overengineering_v1/docs/pi_arduino_comm_test.md`

## Objetivo

- Raspberry Pi 3:
  - roda a IA oficial headless
  - executa visao, FSM e publicacao remota
  - envia apenas assistencias de alto nivel para o Arduino
- Arduino:
  - e o cerebro de tempo real do movimento
  - fecha o loop de linha localmente
  - aplica PID, watchdog e fail-safe
  - executa manobras verdes e desvio de obstaculo
- PC:
  - abre o dashboard remoto
  - recebe frames, estado, logs e telemetria
  - envia comandos de operador para o Raspberry

Guia operacional base:

- `New_AI/obr_overengineering_v1/docs/runtime_boot_recovery.md`

## Arquitetura oficial desta etapa

O contrato atual nao usa mais o Raspberry como controlador direto de motores para eventos de pista. O desenho valido agora e:

- visao no Raspberry estima `line_offset_norm`, `line_angle_deg`, confianca e eventos especiais
- `serial_robot_adapter.py` converte isso em mensagens `ASST ...`
- o Arduino funde sensores locais + assistencias recebidas
- o dashboard exibe telemetria do Arduino e estado serial em tempo real

Arquivos centrais:

- protocolo Python:
  - `New_AI/obr_overengineering_v1/src/modules/control/robot_link_protocol.py`
- adaptador serial:
  - `New_AI/obr_overengineering_v1/src/modules/control/serial_robot_adapter.py`
- firmware:
  - `New_AI/obr_overengineering_v1/arduino/fusionzero_line_follower_controller/fusionzero_line_follower_controller.ino`

## Protocolo serial oficial

Formato textual ASCII, uma linha por mensagem, sempre terminada com `\n`.

### Comandos Raspberry -> Arduino

```text
PING
CMD FORWARD <duration_ms>
CMD STOP 0
CMD ESTOP 0
CMD RESET_ESTOP 0
ASST LINE found=<0|1> offset=<float> angle=<float> conf=<float> gap=<int> source=<token>
ASST GREEN found=<0|1> instruction=<VERDE_ANTES|VERDE_DEPOIS|VERDE_MEIA_VOLTA> side=<LEFT|RIGHT|BOTH> conf=<float> hold_ms=<int> source=<token>
ASST OBSTACLE state=<CLEAR|AHEAD|TEST> conf=<float> hold_ms=<int> source=<token>
```

### Respostas Arduino -> Raspberry

```text
READY FUSIONZERO
PONG
ACK FORWARD <duration_ms>
ACK STOP
ACK ESTOP
ACK RESET_ESTOP
ACK ASST LINE
ACK ASST GREEN
ACK ASST OBSTACLE
ERR <reason>
EVENT WATCHDOG_STOP
TLM mode=<...> line_error=<...> pid=<...> confidence=<...> front=<...> left=<...> right=<...> yaw=<...> roll=<...> pitch=<...> gripper=<...> green=<...> obstacle=<...> failsafe=<0|1> left_pwm=<...> right_pwm=<...>
```

## Semantica operacional

### `PING`

- heartbeat do link serial
- qualquer `PING` valido renova o watchdog do Arduino

### `CMD FORWARD <ms>`

- mantido apenas para validacao manual e `Forward test`
- nao e mais o caminho oficial de decisao para `GREEN`

### `CMD STOP 0`

- parada segura normal
- usado em shutdown, perda de camera, parada manual e encerramento controlado

### `CMD ESTOP 0`

- parada de emergencia com latch
- bloqueia novos comandos ate `CMD RESET_ESTOP 0`

### `ASST LINE`

- pista encontrada no Raspberry
- envia erro lateral normalizado, angulo da linha, confianca e gap de frames
- o Arduino usa isso como assistencia, nao como substituto dos sensores locais

### `ASST GREEN`

- substitui o comportamento antigo de `GREEN => CMD FORWARD 5000`
- o Raspberry identifica a instrucao
- o Arduino decide e executa a manobra localmente com temporizacao deterministica

### `ASST OBSTACLE`

- aceita origem `vision` ou `dashboard`
- suporta `AHEAD`, `TEST` e `CLEAR`
- sensores frontais locais continuam podendo prevalecer

### `TLM ...`

- e a telemetria autoritativa do Arduino para o dashboard
- campos mais importantes:
  - `mode`
  - `line_error`
  - `pid`
  - `confidence`
  - `front`, `left`, `right`
  - `green`
  - `obstacle`
  - `failsafe`
  - `left_pwm`, `right_pwm`

## Robustez implementada no adaptador serial

- `ACK` obrigatorio por comando
- retry limitado com timeout curto
- reconnect automatico da serial
- `PING/PONG` em idle
- parser de `TLM` e `EVENT`
- status agregado para o dashboard com:
  - `heartbeat_age_ms`
  - `telemetry_age_ms`
  - `assist_kind`
  - `control_mode`
  - `line_error`
  - `pid_output`
  - `obstacle_state`
  - `green_instruction`
  - `failsafe`

## Watchdog e fail-safe

### No Raspberry

O runner pode emitir `STOP` ou `ESTOP` quando detectar:

- perda de heartbeat de camera/visao
- pipeline/FSM estagnado
- excecao critica
- estado invalido
- encerramento do runner

### No Arduino

O firmware `fusionzero_line_follower_controller.ino` aplica:

- watchdog serial local
- parada imediata em `ESTOP`
- corte de `manual forward` se aparecer obstaculo
- corte de `manual forward` quando houver perda de heartbeat
- fallback para `SAFE_STOP` quando nao ha linha local nem assistencia valida

## Dashboard remoto

O dashboard agora exibe informacao do Arduino em duas camadas:

- telemetria consolidada em `metadata["control"]`
- estado serial bruto em `metadata["serial"]`

Campos visiveis no overlay/health:

- `control_mode`
- `vision_confidence`
- `line_error`
- `pid_output`
- `obstacle_state`
- `green_instruction`
- `failsafe`
- heartbeat e idade da telemetria

## Comandos remotos disponiveis

- `robot.forward_test`
  - payload: `{"duration_ms": 1200}`
- `robot.stop`
- `robot.force_stop`
- `robot.clear_estop`
- `robot.obstacle_test`
- `robot.obstacle_clear`
- `fsm.force_mode` com `mode=line`
- `fsm.force_mode` com `mode=rescue`

## Configuracao no Raspberry

Descubra a serial do Arduino:

```bash
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
ls /dev/serial/by-id/* 2>/dev/null
```

Instalacao base:

```bash
bash New_AI/obr_overengineering_v1/scripts/install_pi.sh
```

Arquivo de ambiente:

```bash
cp New_AI/obr_overengineering_v1/deploy/fusionzero.env.example \
   New_AI/obr_overengineering_v1/deploy/fusionzero.env
```

Campos principais:

- `FZ_PROFILE=pi3_field`
- `FZ_ROBOT_SERIAL_PORT=/dev/ttyACM0`
- `FZ_ROBOT_BAUD=115200`
- `FZ_ROBOT_DRY_RUN=1` para validar sem mover o robo

Flags uteis do runner:

- `--robot-green-hold-ms`
- `--robot-obstacle-hold-ms`

Subida do runtime:

```bash
bash New_AI/obr_overengineering_v1/scripts/run_pi_headless.sh
```

Boot automatico:

```bash
sudo bash New_AI/obr_overengineering_v1/scripts/install_systemd_service.sh --enable-now
```

## Dashboard no PC

```powershell
powershell -ExecutionPolicy Bypass -File .\New_AI\obr_overengineering_v1\scripts\install_pc.ps1
powershell -ExecutionPolicy Bypass -File .\New_AI\obr_overengineering_v1\scripts\run_pc_dashboard.ps1 -Host <IP_DO_RASPBERRY>
```

## Smoke test local da serial

Antes de ligar o runner completo, voce pode testar a serial direto do PC:

```powershell
python .\New_AI\obr_overengineering_v1\scripts\robot_serial_smoke_test.py --port COM5
```

Opcionalmente:

- `--forward-ms 700` para testar o comando legado de frente
- `--skip-green` para isolar linha + telemetria
- `--skip-obstacle` para isolar linha + verde
- `--green-instruction VERDE_ANTES`
- `--obstacle-state TEST`

## Logs esperados

Exemplos coerentes com a implementacao atual:

```text
robot serial ready port=/dev/ttyACM0
robot serial connected port=/dev/ttyACM0 baud=115200
robot assist -> ASST LINE found=1 offset=0.250 angle=108.000 conf=0.840 gap=0 source=vision ack=ACK ASST LINE ...
robot assist -> ASST GREEN found=1 instruction=VERDE_DEPOIS side=LEFT conf=0.910 hold_ms=900 source=vision ack=ACK ASST GREEN ...
robot assist -> ASST OBSTACLE state=TEST conf=1.000 hold_ms=1200 source=dashboard ack=ACK ASST OBSTACLE ...
robot safe-stop -> CMD STOP 0 ack=ACK STOP ...
robot emergency-stop -> CMD ESTOP 0 ack=ACK ESTOP ...
robot serial event -> EVENT WATCHDOG_STOP
```

## Validacao minima em campo

1. carregar o sketch `fusionzero_line_follower_controller.ino`
2. ligar o Arduino ao Raspberry por USB
3. confirmar a serial real em `/dev/ttyACM*` ou `/dev/ttyUSB*`
4. subir o runner headless no Raspberry
5. abrir o dashboard remoto no PC
6. confirmar `READY FUSIONZERO`, `PONG` e telemetria `TLM`
7. validar `Forward test`
8. validar `Obstacle test` e `Clear obstacle`
9. mostrar marcadores verdes e observar `ASST GREEN`
10. verificar que o Arduino troca entre `FOLLOW_LINE`, `GREEN`, `OBSTACLE` e `SAFE_STOP`

## Gaps que ainda dependem do hardware real

- pinagem final dos motores e sensores
- calibracao real dos sensores de linha
- polaridade real de sensores frontais
- leitura real de IMU
- confirmacao local de verde por sensor dedicado, se existir
- escolha final entre obstaculo puramente visual ou sensor dedicado

Sem esses pontos, o protocolo e a arquitetura estao prontos, mas o ajuste fino de pista ainda depende da bancada.
