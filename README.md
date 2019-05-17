AUTOGRAPH
==========


Versão do add-on do blender a ser usado
para o trabalho autoral "Autograph" - 
tese de doutorado de Lali Krotoszynski -



Conceito: Lali Krotoszynski
Código: João S. O. Bueno
Blender Operator e Expert em animações: Ângelo Benetti

Instalação
==========

O uso desse add-on depende de uma arquivo .blend
com as sequências de dança pré-configuradas, a
ser disponibilizado futuramente.


Após clonar o repositório:
Executar o script "parameter_reader.py" - isso vai gerar
o arquivo "autograph_action_data.py".
Copiar (ou link simbólico) os arqvuios autograph_action_data.py
e flipper.py para a pasta de "modules" do Blender.

Instalar o "pyautogui" de forma que fique disponível no Python
do Blender.
(No Windows, por exemplo, na pasta onde  está o Python.exe que acompanha o blender,
digitar "python -m pip install pyautogui"

Copiar a pasta de uma instalação do pyautogui para a pasta "modules"
do blender funciona também.
)

Instalar o arquivo "autograph.py" como add-on do Blender pela
janela de "user preferences".

Os comandos do add-on ficarão disponíveis na tool-shelf do Object Mode,
com o arquivo .blend apropriado aberto.





