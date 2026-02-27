# Raspberry Pi 3 Setup Guide (FusionZero IA)

Este guia e para colocar a IA para rodar no Raspberry Pi 3 sem mexer no codigo da IA.
Fluxo completo: resetar SD, instalar sistema, instalar dependencias, testar camera USB e executar.

## 1. Hardware e estrategia com 2 cartoes

- Raspberry Pi 3 Model B/3B+
- 2x microSD 16GB
- 1 camera USB
- Fonte 5V 2.5A ou maior

Use os cartoes assim:

- `SD-A (estavel)`: sistema funcionando para treino/teste rapido.
- `SD-B (laboratorio)`: testes de instalacao e mudancas.

## 2. Melhor sistema para seu caso

Recomendado agora:

- `Raspberry Pi OS Lite (64-bit) - Bookworm`

Motivo:

- suporte atual da Raspberry Pi Foundation
- camera stack moderna (`libcamera`, `picamera2`)
- bom para Pi 3 quando voce quer desempenho e menos consumo de RAM

## 3. Resetar e gravar SD (do zero)

No seu computador:

1. Instale `Raspberry Pi Imager`.
2. Abra o Imager.
3. Escolha `Raspberry Pi OS Lite (64-bit)`.
4. Escolha o cartao SD.
5. Clique na engrenagem (configuracoes avancadas) e preencha:
   - hostname
   - usuario/senha
   - Wi-Fi (SSID/senha)
   - locale/timezone
   - habilitar SSH
6. Grave (`Write`).

Links oficiais:

- Raspberry Pi Downloads: https://www.raspberrypi.com/software/
- Instalar sistema com Raspberry Pi Imager: https://www.raspberrypi.com/documentation/computers/getting-started.html#installing-the-operating-system

## 4. Primeiro boot no Raspberry Pi

Depois de ligar o Pi:

```bash
sudo apt update
sudo apt full-upgrade -y
```

Habilite interfaces e grupos:

```bash
sudo raspi-config
```

No `raspi-config`, habilite o que voce usar:

- Interface Options -> `I2C`
- Interface Options -> `SPI`
- Interface Options -> `SSH` (se ainda nao ativou)

Depois:

```bash
sudo usermod -aG video,gpio,i2c,spi $USER
sudo reboot
```

Link oficial (`raspi-config` e interfaces):

- https://www.raspberrypi.com/documentation/computers/configuration.html

## 5. Pacotes base (sistema)

Depois do reboot:

```bash
sudo apt update
sudo apt install -y \
  git curl wget \
  python3-pip python3-venv python3-dev \
  python3-opencv python3-numpy python3-tk python3-pil \
  python3-picamera2 \
  v4l-utils fswebcam i2c-tools
```

## 6. Trazer o projeto para o Pi

Opcao A (Git):

```bash
cd ~
git clone <URL_DO_REPOSITORIO> FusionZero-Robocup-International
```

Opcao B (copiar do PC via `scp`, no PowerShell do Windows):

```powershell
scp -r "C:\Users\Davib\OneDrive\Área de Trabalho\OBR - Arquivos\FusionZero-Robocup-International" \
<usuario>@<ip-do-pi>:/home/<usuario>/
```

## 7. Ambiente Python (seguro no Bookworm)

No Pi:

```bash
cd ~/FusionZero-Robocup-International
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Instale apenas o que faltar para sua execucao:

```bash
pip install \
  adafruit-blinka \
  adafruit-circuitpython-ads7830 \
  adafruit-circuitpython-vl53l1x \
  adafruit-circuitpython-bno08x \
  adafruit-circuitpython-servokit
```

Observacao importante:

- Nao rode `pip install -r requirements.txt` direto no Pi 3 como primeiro passo.
- Primeiro valide camera + IA base. Depois adicionamos libs extras se necessario.

## 8. Testar camera USB antes da IA

```bash
ls /dev/video*
v4l2-ctl --list-devices
```

Teste captura simples:

```bash
fswebcam -r 640x480 --jpeg 90 test.jpg
ls -lh test.jpg
```

Link oficial de camera no Raspberry Pi (inclui USB webcam):

- https://www.raspberrypi.com/documentation/computers/camera_software.html

## 9. Rodar a IA no Pi 3

Ative o ambiente:

```bash
cd ~/FusionZero-Robocup-International
source .venv/bin/activate
```

Executar UI (com monitor no Pi):

```bash
python pc_vision_ui.py
```

Executar sem UI (mais leve):

```bash
python pc_vision_runner.py --mode all --camera 0
```

Se sua camera USB for outro indice:

```bash
python pc_vision_runner.py --mode all --camera 1
```

## 10. Se der erro de Python/pip no Bookworm

Esse erro e comum no Bookworm:

- `error: externally-managed-environment`

A forma correta e usar `venv` (ja coberto neste guia).

Link oficial:

- https://www.raspberrypi.com/documentation/computers/images.html#using-pip-with-rpi-os

## 11. Opcional: pycoral / tflite (somente se realmente precisar)

Se voce for usar Edge TPU/Coral, instale por `apt` quando possivel (mais estavel no Pi):

```bash
sudo apt update
sudo apt install -y python3-pycoral
```

Referencia oficial do projeto pycoral (recomendacao de `apt`):

- https://github.com/google-coral/pycoral

## 12. Reset rapido (quando quiser voltar do zero)

1. Desligue o Pi.
2. Grave novamente o SD no Raspberry Pi Imager.
3. Repita passos 4 a 9 deste guia.

Com 2 cartoes:

- mantenha `SD-A` sempre funcionando
- use `SD-B` para experimentar

## 13. Checklist final de validacao

Antes de testar IA no tapete:

- `v4l2-ctl --list-devices` mostra a camera USB
- `fswebcam` gera imagem
- `python pc_vision_ui.py` abre
- deteccoes visuais aparecem no preview

---

Quando voce enviar fotos do ponto onde esta, seguimos exatamente o proximo comando.
