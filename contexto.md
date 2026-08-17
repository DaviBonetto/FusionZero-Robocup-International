Contexto comum para todos os agentes:

- Objetivo final: evoluir o pipeline oficial FusionZero OBR em `New_AI/obr_overengineering_v1` para um estado pronto para integração robusta com Raspberry Pi 3, dashboard remoto no PC e robô real, incluindo as 12 melhorias priorizadas.
- Entregavel final esperado: codigo implementado, testes, documentacao operacional, scripts de install/run, integracao consolidada e verificacao final sem regressao do pipeline oficial.
- Restricoes:
  - preservar o pipeline oficial atual
  - nao regredir `GREEN` vs `GREEN CORNER`
  - nao perder suporte a multiplas `SILVER`
  - `BLACK` nao pode voltar a virar linha
  - `Force LINE` e `Force RESCUE` devem continuar funcionando
  - manter dashboard remoto no PC e IA headless no Raspberry
  - nesta rodada nao executar deploy final no Raspberry; preparar e validar localmente
  - mudar somente o necessario com justificativa tecnica clara
- Criterio de sucesso:
  - as 12 melhorias implementadas ou preparadas de forma completa dentro da arquitetura oficials
  - testes relevantes passando
  - docs coerentes com o codigo
  - integracao clara e sem gaps
- Regra de comunicacao: reportar progresso em blocos curtos e listar bloqueios explicitamente.
- Regra de handoff: ao concluir, enviar output estruturado para o agente integrador, exceto o integrador que envia ao verificador final.
