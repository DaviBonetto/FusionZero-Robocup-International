# Prompt de Correção Pós-Implementação: Calibração de Vermelho

**Agente Responsável:** `Agent_RedColorCalibration`
**Modo Ativo:** Debug & Fix
**Arquivos Alvo:** Processador de Visão / Pipeline de detecção Vermelha.

## Problema Identificado (Falsos Positivos)

O filtro vermelho está detectando tons terrosos, madeira (parede) ou objetos laranjas sob reflexão. Formas não retangulares estão sendo aceitas como interseção da última linha da pista.

## Tarefa de Correção

> ATUAÇÃO: Agent_RedColorCalibration. MODO: Debug & Fix.
> TAREFA: A detecção dupla de HSV implementada está com range muito flexível. Aplique uma restrição aguda: O range inferior deve ir de [0, 150, 100] a [8, 255, 255] e o range superior de [170, 150, 100] a [180, 255, 255]. Aumente drasticamente a exigência mínima de Saturação (S) para matar laranjas e marrons. Além disso, reforce a função de `identify_red_context`: Injete validação de polinômio `cv2.approxPolyDP`. Se for a linha vermelha da pista, o bounding box DEVE ser estreito e longo. Se for sala de resgate, DEVE ser um mancha disforme vasta ou contorno côncavo. Expurga contornos abaixo de 500 pixels de Área (`cv2.contourArea`).
