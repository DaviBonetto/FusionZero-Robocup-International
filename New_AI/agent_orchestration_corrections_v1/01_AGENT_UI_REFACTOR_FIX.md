# Prompt de Correção Pós-Implementação: Interface Gráfica

**Agente Responsável:** `Agent_UI_Refactor`
**Modo Ativo:** Debug & Fix
**Arquivos Alvo:** `src/ui_overengineering/dashboard.py` (ou arquivo equivalente contendo o loop principal PyQt6/Tkinter).

## Problema Identificado (O Gargalo)

O Dashboard recém implementado está sofrendo de frame-drops severos e está congelando a Main Thread. Além disso, o status textual da Máquina de Estados ("... Following Line ...") não está mudando de forma reativa aos eventos do robô.

## Tarefa de Correção

> ATUAÇÃO: Agent_UI_Refactor. MODO: Debug & Fix.
> TAREFA: O Dashboard recém implementado em PyQt6 está sofrendo de frame-drops e congelando a Main Thread. Você deve refatorar o loop de atualização de imagens (`update_frame`).
>
> 1. Converta a emissão do Frame pelo módulo de Visão em um `pyqtSignal(np.ndarray)` assíncrono.
> 2. As imagens recebidas devem ser redimensionadas (`cv2.resize`) em uma QThread separada ANTES de serem enviadas ao `QLabel` principal via `QPixmap`.
> 3. Na barra central-superior de eventos FSM, garanta que ela possui um Observer local conectado ao `Event_Bus` via um `pyqtSlot` (Slot-Signal Qt puro), para alterar o texto instantaneamente de "... Following Line ..." para "... Picking up alive victim ...".
