# Plano de Implementação Técnico: IA e Interface - Robótica OBR (Resgate)

Este é um plano estratégico, modular e focado em engenharia de sistemas embarcados para migrar a arquitetura base atual (`pc_vision_ui.py` / `pc_vision_runner.py`) do repositório FusionZero-Robocup-International para uma arquitetura avançada de alto desempenho inspirada na equipe _Over Engineering_.

---

## 🔬 1. Diagnóstico Inicial

### Arquitetura Atual

- **UI & Core acoplados:** A interface visual está acoplada fisicamente às threads de atualização da captura, tornando o sistema engessado e de difícil expansão para renderização mais complexa (como gráficos 3D ou histórico de mapa dinâmico).
- **Sobrecarga de Detecção:** A visão computacional pesquisa elementos de forma generalizada. Para aplicações reais em Raspberry Pi/Hardware Embarcado, buscar todos os objetos a todo frame sem filtro de contexto do estado atual do robô causa latência inaceitável e falsos positivos.
- **Detecção do Vermelho Simplista:** A tolerância luminosa costuma falhar sob os holofotes do campeonato, considerando que a máscara HSV vermelha muitas vezes captura espectros vizinhos de laranja ou ignora bordas.
- **Estado Implícito:** A falta de uma Máquina de Estados Finita (FSM) rigorosa gera mensagens dessincronizadas do robô.

### Arquitetura Proposta

- **Message Broker Interno (Event Bus):** Usar ZeroMQ, Pub/Sub, ou filas do Python (`queue.Queue`/`asyncio`) para isolar Visão, Navegação e Interface.
- **Context-Aware Vision:** Pipeline de visão subordinado à máquina de estados. O robô só executa YOLO ou segmentação de área de resgate SE o estado da FSM requerer isso (redução de 60% de processamento).
- **UI Web/PyQt Desacoplada:** Interface operando recebendo streams JSON e Video Streams codificados, garantindo telemetria leve.

---

## 📈 2. Roadmap por Fases

- **[Fase 1] Decoupling & Infraestrutura:** Separar a `VisionUI` (Tkinter) antiga. Implementar `core_node`, `vision_node` e `ui_node` operando isoladamente com comunicação via barramento (Event Bus).
- **[Fase 2] Motor de Visão Computacional Otimizado:** Implementar calibração de iluminação dinâmica (CLAHE) e restringir alvos apenas para Bola, Quadrado, Linha, Área de Deposição e Vermelho (com dupla máscara HSV).
- **[Fase 3] FSM (Finite State Machine):** Acoplar lógica formal para eventos de "Gap", "Victim", "Intersect", com hooks definidos.
- **[Fase 4] Dashboard GUI Baseado em Referências (OverEngineering):** Escrever o novo código da interface espelhando a disposição visual fornecida (Vídeo Original Main, Vídeo Original Top, Painel de Sensores, Painel Gráfico 3D/Caminho 2D, Barra de Status do fluxo FSM e Relógio principal).
- **[Fase 5] Integração e Testes:** Calibração fina, gravação de ROSbags locais (ou equivalente) e simulações com os Agentes através do Codex.

---

## 📁 3. Estrutura de Pastas Ideal (Refatoração)

```text
├── src/
│   ├── core/
│   │   ├── event_bus.py         # Broker Pub/Sub local
│   │   ├── state_machine.py     # FSM implementada
│   ├── modules/
│   │   ├── vision/
│   │   │   ├── preprocessor.py  # CLAHE, Morfologia
│   │   │   ├── pipelines.py     # Detecção baseada no estado atual
│   │   ├── navigation/
│   │   │   ├── path_tracker.py  # Deque buffer de x, y, theta
│   ├── ui_overengineering/
│   │   ├── dashboard.py         # Nova Interface Principal
│   │   ├── components/
│   │   │   ├── 3d_viewer.py     # Painel de poses Euler/Quat
│   │   │   ├── path_canvas.py   # Renderização Histórica
│   ├── main_orchestrator.py
```

---

## ⚙️ 4. Soluções e Arquitetura de Módulos

### 4.1. Melhoria da Interface (Dashboard Visual)

**Tecnologias Recomendadas:** PyQt6 ou FastAPI WebSockets + Vue.js/React. A abordagem PyQt6 integrando PyOpenGL/PyGame em Widgets é a mais confiável para performance no desktop embarcado da OBR se não quiser ir para stack web.

**Adaptação do Layout (Baseado nas Screenshots):**
A interface será dividida em um Grid rígido escuro (Dark Theme, fundo próximo a `#1E1E1E` ou similar), com a seguinte topologia referencial observada nas imagens:

1. **Top Row (Visão Principal & Controle):**
   - **Esquerda:** Câmera Frontal Processada (exibindo bordas azul claro/escuro para a pista, bounding boxes vermelhas para as marcações de interseção verde/linha e framerate overlay).
   - **Centro/Topo:** Controles de playback (`>`), monitor de uso de `% CPU` e contadores de frames (`IPS_C`, `IPS_S`).
   - **Direita:** Câmera Frontal Crua (Raw/Perspective) focada estritamente na faixa de captura e desvio da linha com overlay de horizonte e FPS.
   - **Widget Central do Topo:** Status dinâmico da Máquina de Estados (Ex: `... Following Line ...`, `... Dumping alive victims ...`, `... Picking up alive victim ...`).
2. **Bottom Row (Telemetria & Gráficos):**
   - **Esquerda (Painel de Sensores/Cinemática):** Matriz de dados com distância de detecção (ex: `Front L: 300 mm`, `Front R: 710 mm`, `Left`, `Right`, `Front C`, `Back`, `Gripper: 90 mm`) e Odometria IMU (`Yaw`, `Pitch`, `Roll` em graus).
   - **Centro (Visualizador de Posição):** Renderizador 3D exibindo o modelo CAD simplificado do robô girando e oscilando no espaço conforme Pitch/Roll/Yaw lidos pelos sensores (Integrando VisPy ou render simples no PyQt).
   - **Direita:** Painel vazio/reservado (ferramentas como botão de grip `g`, e LEDs indicadores estáticos ⚪⚪⚫).
3. **Rodapé:**
   - Temporizador Master (`00:15:79`) e Sub-timers (ex: `00:48:56`), cruciais para a OBR.

**Desacoplamento Técnico:**

- O código da _Over Engineering_ usa painéis reativos. Nós adaptamos substituindo polling constante por observadores de estados (`on_state_change`).
- A integração visual deve utilizar envio de frames processados via `cv2.imencode()` para buffers consumidos por `QLabel` do Qt, evitando trancar a UI com a thread pesada da visão computacional.

### 4.2. Melhorar Reconhecimento Visual

**Pipeline Ideal:**

1. Câmera > Pré-processamento: Cortar máscara de interesse (ROI) e aplicar CLAHE (Equalização de Histograma Limitado por Contraste - excelente para balancear sombras na malha da pista).
2. Se o estado = SEGUE*LINHA: Roda \_apenas* Binarização Adaptativa + Detecção de contornos pretos.
3. Se o estado = ZONA_RESGATE: Desliga procura de linhas, aciona modelo YOLO nano otimizado com TensorRT / TFLite (ou detecção geométrica clássica) focando em Bolas e Quadrados.
4. Robustez: Atualização da máscara baseada na claridade média da cena (Mean Luma) para flutuar os limites inferior/superior do HSV caso a luz ambiente mude no pavilhão de competição.

### 4.3. Visualização do Caminho (Path Rendering)

- **Estrutura de dados:** Odom Queue (Fila dupla) retendo as posições globais inferidas (por encoders do robô e tracking de vetores da câmera). Armazena tuplas: `(x, y, timestamp)`.
- **Renderização Dinâmica:** Um `numpy array` de traçado interno pintando os pontos no Canvas da interface. Curvas recentes são desenhadas com `#ADD8E6` (Azul Claro) fazendo degrade até posições arcaicas com `#00008B` (Azul Escuro). Histórico é persistido em `.json` temporário por round garantindo que dados não se percam em crash da interface.

### 4.4. Sistema de Mensagens e Estados

- **Arquitetura FSM:** Construir utilizando classe base estrita.
  - Exemplos de trigger FSM: `Event(ON_GAP)` -> Estado passa de `FOLLOWING` para `VALIDATING_GAP`.
- **Logger estruturado:** Todas as transições geram um evento publicado no `event_bus` repassado para a interface de forma determinística, permitindo debugging cirúrgico dos logs do Terminal.

### 4.5. Detecção na Cor Vermelha

- **Estratégia Matemática HSV Duplo:** O matiz do vermelho vai de `0-10` e de `160-180`. Devemos unir as máscaras via operações lógicas de BITWISE (`cv2.bitwise_or(mask1, mask2)`).
- **Identificação Espacial vs Diferenciação de Contexto:**
  - Fitas vermelhas estendem-se horizontalmente e possuem bounding box de formato achatado (aspect ratio > 3.0), enquanto a área de resgate muitas vezes se apresenta como uma mancha vasta ou geometria quadrada regular. Testar preenchimento circular / concavidade.

---

## 🤖 5. Arquitetura Multiagente: Agentes Codex & Prompts

A refatoração será escalada delegando tarefas específicas aos agentes através do **Codex**, rodando em paralelo em arquivos desacoplados.

### 🟢 Agente 1: Agent_UI_Refactor

**Objetivo:** Recriar a arquitetura visual inspirada na _Over Engineering_, com painéis desacoplados da lógica de hardware.
**Inputs:** Estrutura base da interface UI e referências de tela solicitadas.
**Outputs:** Sistema completo em PyQt6 com divisões de vídeo, robô 3D em placeholder e log de eventos.
**Arquivos para analisar:** `pc_vision_ui.py` atual.
**Prompt para Codex:**

> ATUAÇÃO: Agent_UI_Refactor.
> TAREFA: Desenvolva a estrutura de uma nova UI assíncrona usando PyQt6 espelhando a estrutura visual escura "Over Engineering". Crie o Grid principal contendo: (1) Dois painéis de imagem grandes no topo (visão processada com bordas na esquerda vs raw line-tracking na direita). (2) Botões de Play e status de CPU no topo-centro. (3) Uma barra de status textual logo abaixo da câmera direita para a Máquina de Estados (ex: "... Following Line ..."). (4) Um painel inferior-esquerdo contendo grids de Labels (`Front L`, `Right`, `Gripper`, `Yaw`, `Pitch`, etc.). (5) Um Canvas central-inferior preparado para receber um modelo 3D do Robô. (6) Um relógio centralizado grande no bottom. Isole processamento pesado em threads secundárias usando `QThread` ou `Signals/Slots`. Mantenha os eventos reativos.

### 🟢 Agente 2: Agent_Vision_Optimization

**Objetivo:** Limpar a poluição visual, otimizando o pipeline modular e os filtros HSV adaptativos com CLAHE e segmentações isoladas de cor.
**Inputs:** Código base de processamento da câmera que lida de forma genérica com objetos.
**Outputs:** Classe de processamento inteligente, com seletor dinâmico baseado no status da FSM para ligar/desligar instâncias pesadas da YOLO.
**Arquivos para analisar:** `pc_vision_runner.py`.
**Prompt para Codex:**

> ATUAÇÃO: Agent_Vision_Optimization.
> TAREFA: Analise o arquivo de visão atual. Refatore o pipeline implementando pré-processamento obrigatório com o algoritmo CLAHE para mitigar variações de iluminação. Desenvolva um motor onde os detectores de linha vs. bolas vs. zona vermelha não sejam todos executados simultaneamente em todo frame. Crie `switch_pipeline(estado)` que execute apenas os filtros CV2 e/ou inferências neurais cabíveis ao contexto do robô, limitando as classes a exclusivas instâncias: [Bola, Quadrado, Linha, Área Verde_Deposição, Vermelho_Resgate].

### 🟢 Agente 3: Agent_Path_Render

**Objetivo:** Criar um mapa odométrico interno focado em render graphics.
**Inputs:** Posições angulares simuladas / reais da movimentação das rodas.
**Outputs:** Canvas visual (Path Renderer) atualizando trilha baseada em eixos reais de movimento local.
**Prompt para Codex:**

> ATUAÇÃO: Agent_Path_Render.
> TAREFA: Implemente um módulo de renderização de Path em PyGame integrado ou em uma View do PyQt6 (PathCanvas). Crie uma estrutura `Deque` de tamanho estático com posições (X,Y) lidas simuladamente. Aplique gradiente dinâmico com cor Azul claro sendo cabeça (atual) e Azul escuro como cauda (histórico). Desenhe o rastro dinamicamente a cada 100ms. O sistema deve possuir tolerância contra overflow da memória após rodadas prolongadas.

### 🟢 Agente 4: Agent_RedColorCalibration

**Objetivo:** Garantir a infalibilidade do reconhecimento do espectro vermelho global frente a condições desfavoráveis de campeonato.
**Inputs:** Módulo de HSV, calibrações iniciais de vídeo.
**Outputs:** Detectores Red Marker avançados combinados baseados em Dupla Fronteira e Aspect Ratio.
**Prompt para Codex:**

> ATUAÇÃO: Agent_RedColorCalibration.
> TAREFA: Reescreva o módulo de reconhecimento de contornos vermelhos. É estritamente mandatório construir a máscara HSV considerando as pontas da tabela do matplotlib hsv color_maps (Ex.: Lower1: 0,100,100 / Upper1: 10,255,255 e Lower2: 160,100,100 / Upper2: 180,255,255), realizando posterior operação Bitwise OR nelas. Após isso, os contornos encontrados devem passar por função `identify_red_context(contour)` verificando Aspect Ratio e Convexity Defects para diferenciar de imediato "Linha Vermelha de Chegada" vs "Borda de Zona de Resgate".

### 🟢 Agente 5: Agent_StateMachine

**Objetivo:** Acoplar todas a mecânica a uma árvore de estado formal de fluxos de decisão limpos.
**Inputs:** Scripts principais desconexos do robô.
**Outputs:** Módulo oficial formal, eventos rastreáveis.
**Prompt para Codex:**

> ATUAÇÃO: Agent_StateMachine.
> TAREFA: Implemente uma Máquina de Estados Finita (FSM) rigorosa baseada no paradigma de Classes ou usando o pacote `transitions` de Python para o robô resgate da OBR. Crie e exporte os nodos: SEARCHING_LINE, FOLLOWING_LINE, VALIDATING_GAP, CROSSING_GAP, VICTIM_FOUND, RESCUE_ZONE_DETECTED. Acople no construtor a geração de Logs visuais padronizados (ex: `TIMESTAMP [STATE] msg`). Permita triggers para os outros agentes capturarem seus signals em callbacks.

### 🟢 Agente 6: Agent_TestValidation

**Objetivo:** Criar camada sólida para atestar estabilidade técnica contínua (Mock e Tests unitários).
**Inputs:** Arquitetura do repositório em implantação ou finalizado (estado Verde).
**Outputs:** Suite pytest baseada em logs gravados ou mocks offline.
**Prompt para Codex:**

> ATUAÇÃO: Agent_TestValidation.
> TAREFA: Valide todo o pipeline refatorado. Crie uma documentação base e suite de Testes em Pytest em pastas adjuntas (`tests/`). Para o pipeline de Visão, injete imagens ruidosas sintéticas para averiguar a resiliência do CLAHE e máscaras do vermelho. Verifique Mock objects para disparo de Transição de FSM e veja se os eventos adequados repassam para Event_Bus. Nenhum push é permitido na Main se `pytest` apresentar colapsos do OpenCV ou exceções de assincronia de thread na UI recriada.

---

## 🛡️ 6. Metas, Deploy e Plano de Testes

1. **Versionamento e Deploy:**
   - Trabalhar no repositório mantendo o conceito "Git Flow". Para rodar os agentes codex, crie ramos padronizados (`feature/fsm`, `feature/overengineering-ui`).
   - Todos os arquivos implementados devem ter testes paralelos que garantam instâncias na Raspberry Pi não quebrem as Threads de desktop PC.
2. **Mitigação de Riscos Técnicos:**
   - **GIL Lock / Engasgo da Interface:** Separar inferência YOLO ou cálculo pesado do Morphological em Threads / Multiprocessing separados repassando dados pelo Event Bus.
   - **Luz no pavilhão:** Deixar HSV bounds expostos em configuração `.json` acessível localmente para correção instantânea em tempo real e não harcoded, mas usar o Red Calibration e CLAHE como primeira defesa vitalícia.
3. **Indicador de Conclusão Técnica (DoD - Definition of Done):**
   - Latência End-To-End de captura da câmera até renderização na UI não ultrapassar 50ms per frame.
   - O path 2D sendo preenchido no PyGame/Canvas ao longo do tempo.
   - Detecção do Vermelho acionar 100% das vezes em vídeos simulados de arenas difíceis.
   - Logger imprimindo `[INFO] Changed state de Following -> Victim Found` simultaneamente na tela sem delay.
