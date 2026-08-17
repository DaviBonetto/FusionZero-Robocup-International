# Official Codex Handoff Prompt

```text
Voce vai continuar o desenvolvimento oficial do projeto FusionZero OBR.

Contexto geral:
- O pipeline oficial atual esta em `New_AI/obr_overengineering_v1`
- Esse pipeline virou a versao oficial para os proximos testes
- O objetivo imediato agora e integrar IA + Raspberry Pi 3 + Arduino + robo real
- O dashboard deve rodar no PC e a IA/camera devem rodar no Raspberry pela mesma rede Wi-Fi

Stack e arquitetura atual:
- Visao, FSM e dashboard em Python
- UI em PyQt6
- EventBus interno com eventos de frame, deteccao, estado, logs e comandos de UI
- Dashboard remoto por TCP ja implementado
- Raspberry executa o runner headless e publica eventos/frames comprimidos
- PC executa um cliente que consome esses eventos e abre o dashboard localmente

Arquivos principais:
- `New_AI/obr_overengineering_v1/src/live_dashboard_runner.py`
- `New_AI/obr_overengineering_v1/src/remote_dashboard.py`
- `New_AI/obr_overengineering_v1/src/remote_dashboard_client.py`
- `New_AI/obr_overengineering_v1/src/modules/vision/pipelines.py`
- `New_AI/obr_overengineering_v1/src/ui_overengineering/dashboard.py`
- `New_AI/obr_overengineering_v1/configs/vision_config.json`
- `New_AI/obr_overengineering_v1/docs/pi3_remote_dashboard.md`

O que ja foi corrigido e esta considerado oficial:
1. Separacao correta entre:
- `GREEN` = quadrado verde do segue-linha
- `GREEN CORNER` = marcador do resgate

2. `GREEN CORNER` nao conflita mais com `GREEN`
- no `FOLLOWING_LINE` o quadrado verde continua funcionando
- `GREEN CORNER` foi isolado para contexto de resgate

3. Deteccao de bolas no resgate melhorada:
- silver ball com suporte a multiplas deteccoes no mesmo frame
- overlay desenha caixas em mais de uma silver ao mesmo tempo
- black ball continua separada da silver
- filtro de linha foi endurecido para reduzir black ball sendo confundida com linha

4. Overlay e UI:
- confianca em `GREEN`, `GREEN CORNER`, `RED CORNER`, `SILVER`, `BLACK`
- botao `Force LINE`
- botao `Force RESCUE`
- setas do steering ja funcionando

5. Testes:
- suite passando com `78 passed`

Situacao atual de negocio:
- Ainda nao temos a silver line fisica para validar a troca automatica de modo em campo real
- Por isso foi adicionado o botao de troca manual de modo para testes
- O usuario agora quer validar integracao IA -> robo real

Proximo objetivo:
Integrar Raspberry Pi 3 com Arduino para que deteccoes e estados da IA gerem comandos reais para o robo.

Exemplo inicial de validacao pedido pelo usuario:
- quando detectar quadrado verde, o robo deve ir para frente por 5 segundos

Importante:
- Esse exemplo e um teste de integracao, nao a politica final de navegacao
- A prioridade agora e provar que IA e robo estao conversando corretamente

Sua missao agora:
1. Ler o pipeline oficial atual
2. Confirmar como extrair eventos de alto nivel da IA para controle do robo
3. Definir a interface Raspberry -> Arduino
4. Implementar um adaptador simples de comandos para teste real
5. Propor e/ou implementar uma rotina minima de validacao em campo

O que voce precisa decidir tecnicamente:
- protocolo Raspberry -> Arduino:
  - serial USB e o caminho preferido inicial
- formato de comando:
  - simples, robusto, textual ou binario curto
- taxa de envio:
  - evitar spam
  - enviar estado ou comando com histerese
- regras minimas:
  - `GREEN` => comando de teste de avancar por 5 segundos
  - `Force LINE` e `Force RESCUE` devem continuar uteis para debug

Escopo desejado agora:
- manter dashboard no PC
- manter IA no Raspberry Pi 3
- conectar Raspberry ao Arduino
- criar adaptador de comandos entre deteccoes/eventos e movimento do robo
- documentar como executar

O que o usuario provavelmente vai fazer:
- ligar camera no Raspberry
- ligar Raspberry no Arduino
- subir o runner headless no Raspberry
- abrir o dashboard no PC
- testar deteccoes
- testar comandos simples no robo

O que voce deve pedir ao usuario se necessario:
- porta serial do Arduino no Raspberry
- baud rate usado no firmware atual
- formato atual esperado pelo Arduino, se ja existir
- se o firmware do Arduino ja esta pronto ou se voce tambem vai ter que definir o protocolo dos comandos

Criterio de sucesso desta etapa:
- dashboard remoto funcionando do Raspberry para o PC
- Raspberry publicando eventos da IA
- Raspberry enviando comando ao Arduino quando evento relevante acontecer
- teste simples do tipo `GREEN => frente por 5 segundos` validado

Se houver duvida, preserve o pipeline oficial atual e nao regreda:
- separacao `GREEN` vs `GREEN CORNER`
- multiplas silver balls
- black ball nao pode voltar a virar linha
```
