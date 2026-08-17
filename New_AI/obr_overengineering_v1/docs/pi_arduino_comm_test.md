# Teste Rapido Pi <-> Arduino <-> Dashboard

## Objetivo desta fase

Esta etapa nao e o robo final completo. O foco agora e validar:

- Raspberry Pi vendo a pista com a camera USB
- Raspberry Pi enviando assistencias seriais para o Arduino Mega
- Arduino Mega controlando o L298N e os motores
- PC acompanhando tudo pelo dashboard remoto simples

Nesta fase o firmware do Mega esta em `camera_assist_only=true`.
Isso significa:

- os sensores locais de linha `A0-A4` ficam ignorados
- a linha vem da camera no Raspberry
- `ASST LINE`, `ASST GREEN` e `ASST OBSTACLE` sao a fonte principal de decisao

## Pinagem final usada no Mega

L298N -> Arduino Mega

- `ENA -> D2`
- `IN1 -> D4`
- `IN2 -> D8`
- `IN3 -> D3`
- `IN4 -> D5`
- `ENB -> D7`

Motores no L298N:

- `OUT1/OUT2 -> motor esquerdo`
- `OUT3/OUT4 -> motor direito`

Observacoes:

- o canal A do L298N virou o motor esquerdo
- o canal B do L298N virou o motor direito
- se o robo andar para tras ou girar invertido, ajuste primeiro:
  - `LEFT_MOTOR_INVERTED`
  - `RIGHT_MOTOR_INVERTED`
- nao troque a pinagem antes de testar essas flags

## Arquivos principais desta fase

- firmware Mega:
  - `New_AI/obr_overengineering_v1/arduino/fusionzero_line_follower_controller/fusionzero_line_follower_controller.ino`
- runner no Pi:
  - `New_AI/obr_overengineering_v1/scripts/run_pi_comm_test.sh`
- dashboard simples no PC:
  - `New_AI/obr_overengineering_v1/scripts/run_pc_comm_dashboard.ps1`
- smoke test serial:
  - `New_AI/obr_overengineering_v1/scripts/robot_serial_smoke_test.py`

## O que fica conectado em cada momento

### 1. Upload inicial do firmware

Use o PC ligado por USB no Mega.

Nessa hora:

- PC -> USB -> Arduino Mega
- Raspberry ainda nao precisa estar ligado
- motores e camera podem ficar desligados

### 2. Teste real de comunicacao

Depois do upload, retire o USB do PC e passe o Mega para o Raspberry.

Nessa hora:

- Raspberry Pi -> USB -> Arduino Mega
- camera USB -> Raspberry Pi
- bateria dos motores -> L298N
- PC fica solto, sem cabo no robo
- PC so precisa estar na mesma rede do Raspberry

Resumo importante:

- o PC nao precisa ficar plugado no robo para o teste real
- o PC so abre o dashboard remoto
- quem manda no robo durante o teste e o conjunto Raspberry + Arduino

## Ordem recomendada de preparacao

### A. Preparar no PC

1. Atualize os arquivos do projeto no seu PC.
2. Grave o sketch no Mega.
3. Se quiser, rode o smoke test serial com o Mega ainda no PC.

Exemplo de smoke test:

```powershell
python .\New_AI\obr_overengineering_v1\scripts\robot_serial_smoke_test.py --port COM5
```

O esperado no minimo e:

- `READY FUSIONZERO`
- `PONG`
- `ACK ASST LINE`
- `ACK ASST GREEN`
- `ACK ASST OBSTACLE`
- `ACK STOP`

### B. Levar para o Raspberry por microSD

Se voce vai usar o microSD no lugar de SSH:

1. Desligue o Raspberry.
2. Tire o microSD.
3. Plugue o microSD no PC.
4. Copie a pasta atualizada `New_AI/obr_overengineering_v1` para o sistema de arquivos do Pi.
5. Recoloque o microSD no Raspberry.

Dica pratica:

- se a maioria dos arquivos ja esta no Pi, copie pelo menos:
  - `src/`
  - `scripts/`
  - `docs/`
  - `arduino/`
  - `deploy/fusionzero.env.example` ou seu `fusionzero.env`

### C. Subir o Pi para o teste real

No Raspberry, use o runner dedicado desta fase:

```bash
bash New_AI/obr_overengineering_v1/scripts/run_pi_comm_test.sh
```

Defaults importantes desse script:

- camera USB em `camera-index=0`
- serial do Mega em `/dev/ttyACM0`
- baud `115200`
- profile `pi3_field`
- relay remoto na porta `8765`

Se precisar mudar a serial:

```bash
FZ_ROBOT_SERIAL_PORT=/dev/ttyACM1 bash New_AI/obr_overengineering_v1/scripts/run_pi_comm_test.sh
```

## Abrindo o dashboard no PC

Com o Raspberry e o PC na mesma rede:

```powershell
powershell -ExecutionPolicy Bypass -File .\New_AI\obr_overengineering_v1\scripts\run_pc_comm_dashboard.ps1 -Host <IP_DO_RASPBERRY>
```

O dashboard simples mostra:

- `Pi relay`
- `Arduino serial`
- `Heartbeat`
- video processado
- visao:
  - `line`
  - `green`
  - `green_instruction`
  - `line_offset_norm`
  - `line_angle_deg`
- telemetria:
  - `mode`
  - `assist_kind`
  - `line_error`
  - `pid_output`
  - `green_instruction`
  - `obstacle_state`
  - `failsafe`
- badges:
  - `linha detectada`
  - `assistencia LINE enviada`
  - `duplo verde -> VERDE_MEIA_VOLTA`
  - `Arduino em GREEN`
  - `failsafe ativo`

Comandos disponiveis no dashboard:

- `Forward test`
- `STOP`
- `ESTOP`
- `Clear ESTOP`
- `Obstacle test`
- `Clear obstacle`

## Sequencia de teste recomendada

### Teste 1. Link basico

- ligue Pi + Mega
- abra o dashboard no PC
- confirme:
  - `Pi relay = CONNECTED`
  - `Arduino serial = CONNECTED`
  - `Heartbeat = OK`

### Teste 2. Comando manual

- clique `Forward test`
- confirme no dashboard que a serial continua viva
- clique `STOP`

### Teste 3. Linha preta

- posicione a camera para enxergar a linha preta
- confirme:
  - `line = YES`
  - badge `linha detectada`
  - badge `assistencia LINE enviada`
  - `assist_kind = LINE`
  - `mode = FOLLOW_LINE` no Arduino

### Teste 4. Dois verdes

- mostre os dois quadrados verdes para a camera
- confirme:
  - `green = YES`
  - `green_instruction = VERDE_MEIA_VOLTA`
  - badge `duplo verde -> VERDE_MEIA_VOLTA`
  - `mode = GREEN`

O esperado e o Mega executar a meia-volta de 180 graus.

### Teste 5. Fail-safe

- desconecte a serial ou derrube o link Pi↔Mega
- confirme:
  - estado serial fica stale / erro
  - badge `failsafe ativo`
  - motores param

## Alimentacao e cabos

Para o teste real:

- Mega pode receber logica pela USB do Raspberry
- motores devem ficar na bateria do robo via L298N
- mantenha GND comum entre os modulos de controle e potencia

## Quando mexer no codigo de novo

So mexa antes do proximo teste se acontecer um destes casos:

- motor esquerdo ou direito invertido
- serial do Mega apareceu em outra porta
- camera USB entrou em outro indice
- o Raspberry abre o runner, mas o dashboard nao conecta por IP/rede

Se aparecer qualquer duvida sobre eletrica restante, alimentacao ou indices reais de porta no Pi, vale me passar isso antes de ligar tudo no chao para o primeiro teste.
