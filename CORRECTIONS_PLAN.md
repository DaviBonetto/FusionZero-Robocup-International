# Plano de Correções e Ajustes Pós-Implementação (Fase de Lapidação)

Este plano deve ser executado no repositório `FusionZero-Robocup-International` **assumindo que as implementações base da Fase 1 a 4 já foram realizadas pelo Codex**.

O foco agora é corrigir assincronias, melhorar os gargalos de visão, calibrar o Red Marker e estabilizar a Interface do Usuário espelhada da _Over Engineering_.

---

## ❓ Esclarecimento sobre o "Clone" (Sobre a Dúvida)

**Os Agentes não deveriam clonar fisicamente o repositório da _Over Engineering_.**
O objetivo orquestrado no plano original (e neste) é **reproduzir a arquitetura visual e intelectual** deles para _dentro_ do nosso código. Clonar e tentar fazer um _merge_ de dois repositórios de robôs diferentes causaria falha catastrófica de dependências de hardware (sensores diferentes, portas seriais diferentes, etc.).

> **Abordagem Correta para os Agentes:** Os Agentes trabalharam exclusivamentne escrevendo código NO NOSSO repositório (`FusionZero-Robocup-International`), criando as classes de Interface, Visão e Event_Bus baseando-se estritamente nas regras ditadas pelos prompts, sem depender de pastas externas.

---

## 🛠️ Diagnóstico Pós-Implementação (Expected Issues)

Assumindo que os agentes já rodaram, os problemas listados abaixo são característicos dessa ponte de integração e devem ser corrigidos pelos respectivos especialistas em uma **Bateria de Correções**:

| Problema Identificado no Integrado                                                               | Agente Responsável pela Correção                       |
| :----------------------------------------------------------------------------------------------- | :----------------------------------------------------- |
| UI travando por alguns milissegundos enquanto a Visão formata os `cv2.findContours`.             | **Agent_UI_Refactor**                                  |
| Cores do traçado no Path 2D estão sumindo rápido ou causando lag com histórico enorme.           | **Agent_Path_Render**                                  |
| Falsa detecção de bolinhas laranjas como "Zona de Resgate Vermelha" sob luz de teto.             | **Agent_RedColorCalibration**                          |
| Módulo de Visão ativando a detecção de linha dentro da sala de resgate (desperdício processual). | **Agent_Vision_Optimization** & **Agent_StateMachine** |
| Variáveis de Estado da FSM mudando, mas a UI (`QLabels` no painel superior) não atualiza.        | **Agent_UI_Refactor** & **Agent_StateMachine**         |

---

## 🤖 Plano de Correção Multiagente (Prompts de Ajuste)

Para disparar no **Codex**, foque em passar o contexto do erro específico junto com estes prompts de correção.

### 🔴 Agente: Agent_UI_Refactor -> (Correção: Desengasgo do Qt)

**Bug a corrigir:** A interface congela momentaneamente na exibição de muitos frames (gargalo da Main Thread). O status textual da FSM ("... Following Line ...") não está mudando reativamente.
**Arquivos alvo:** `src/ui_overengineering/dashboard.py` (ou onde o PyQt6 foi montado).
**Prompt de Correção para Codex:**

> ATUAÇÃO: Agent_UI_Refactor. MODO: Debug & Fix.
> TAREFA: O Dashboard recém implementado em PyQt6 está sofrendo de frame-drops e congelando a Main Thread. Você deve refatorar o loop de atualização de imagens (`update_frame`).
>
> 1. Converta a emissão do Frame pelo módulo de Visão em um `pyqtSignal(np.ndarray)` assíncrono.
> 2. As imagens recebidas devem ser redimensionadas (`cv2.resize`) em uma QThread separada ANTES de serem enviadas ao `QLabel` principal via `QPixmap`.
> 3. Na barra central-superior de eventos FSM, garanta que ela possui um Observer local conectado ao `Event_Bus` via um `pyqtSlot` (Slot-Signal Qt puro), para alterar o texto instantaneamente de "... Following Line ..." para "... Picking up alive victim ...".

### 🔴 Agente: Agent_RedColorCalibration -> (Correção: Falso Positivo Luminoso)

**Bug a corrigir:** O filtro vermelho está abrangendo laranja escuro ou tons de madeira. Formas não retangulares estão sendo aceitas como borda final de resgate.
**Arquivos alvo:** Módulo responsável pelo processamento de contornos vermelhos.
**Prompt de Correção para Codex:**

> ATUAÇÃO: Agent_RedColorCalibration. MODO: Debug & Fix.
> TAREFA: A detecção dupla de HSV implementada está com range muito flexível. Aplique uma restrição aguda: O range inferior deve ir de [0, 150, 100] a [8, 255, 255] e o range superior de [170, 150, 100] a [180, 255, 255]. Aumente drasticamente a exigência mínima de Saturação (S) para matar laranjas e marrons. Além disso, reforce a função de `identify_red_context`: Injete validação de polinômio `cv2.approxPolyDP`. Se for a linha vermelha da pista, o bounding box DEVE ser estreito e longo. Se for sala de resgate, DEVE ser um mancha disforme vasta ou contorno côncavo. Expurga contornos abaixo de 500 pixels de Área (`cv2.contourArea`).

### 🔴 Agente: Agent_StateMachine + Agent_Vision_Optimization -> (Correção: Otimização Condicional)

**Bug a corrigir:** A YOLO ou o detector de círculos continua rodando na rampa, fazendo a temperatura da CPU bater 80%+.
**Arquivos alvo:** `state_machine.py`, `vision/pipelines.py`
**Prompt de Correção para Codex:**

> ATUAÇÃO: Agent_Vision_Optimization e Agent_StateMachine. MODO: Debug & Integratrion Fix.
> TAREFA: A arquitetura atual de FSM não está cortando funções pesadas. Refatore `vision/pipelines.py` injetando uma leitura rigorosa do `current_state` da Máquina de Estados.
> Regra Absoluta: Se estado == SEARCHING_LINE ou FOLLOWING_LINE ou VALIDATING_GAP -> RETORNE IMEDIATAMENTE antes de carregar modelo neural de resgate (YOLO/HoughCircles). Apenas rode Binarização Preta para linha.
> Se estado == RESCUE_ZONE_DETECTED -> Desative a máscara negra (Binarização), e ligue 100% dos recursos visuais no tracking de Vitimas Vivas (Prata) / Mortas (Preta). Faça com que a mudança de estado propague um Log severo demonstrando a economia de pipeline no buffer.

### 🔴 Agente: Agent_Path_Render -> (Correção: Estouro de Memória)

**Bug a corrigir:** Histórico do Path Canvas infinito derrubando a memória gráfica ao longo de tempos superiores a 3 minutos simulados. O buffer 2D fica lotado de pontos sobrepostos.
**Arquivos alvo:** componente de Path 2D (Renderizador do traçado Azul).
**Prompt de Correção para Codex:**

> ATUAÇÃO: Agent_Path_Render. MODO: Debug & Fix.
> TAREFA: Refatorar o renderizador 2D. Troque o cache acumulativo infinito de arrays por um buffer estático usando `collections.deque(maxlen=1000)`. Se o robô ficar parado enviando coordenadas iguais, não adicione-as na Deque de rastreamento (implemente distância Euclidiana mínima antes de salvar um `(x,y)` novo). Desenhe apenas os segmentos de linha em um Surface pré-compilado ou no canvas, evitando que o PyGame/Qt reconstrua todos os 1000 vetores desde o início a cada Frame de 30FPS.
