# Tècniques avançades machine learning Nom:
## TERCERA PART: FINE TUNING (mig)
### Requisits:
En aquesta pràctica es vol fer un fine tuning d’un LLM, els requisits són:
* Fes servir un LLM petit (1b, 2b, 3b o 4b, màx 8b), per exemple llama 3.2 (1b).
* Fes servir Python.
* Fes l’entrenament fine tuning en local.
* Recomano fer servir:
  * Torch (PyTorch) és la llibreria fonamental per treballar amb tensors
  * PEFT (Parameter-Efficient Fine-Tuning) proporciona mètodes com LoRA,
  que permeten entrenar eficientment només una petita part dels paràmetres d'un model gran.
  * BitsandBytes permet fer quantització, que redueix el consum de memòria del
  model en GPU.
* CUDA permet entrenar fent servir la GPU (es pot fer en CPU també).
* Fes servir el document de coneixement proporcionat sobre deixalles a Olot.
* Fes servir l’anglès en les interaccions amb el Chat.

### Objectius:
* Fer el codi per a fer fine tuning en local d’un LLM petit en base a un document de coneixement
donat.
* Fer que l’LLM aprengui la informació del document de coneixement subministrat.
* Calcular el rendiment de l’LLM sobre tot el data set de preguntes del document de
coneixement abans de ser entrenat segons el GPTScore.
* Comparar el nou LLM amb el mateix indicador sobre tot el data set de preguntes del
document de coneixement després de ser entrenat per observar la millora.

### Entrega:
* Descripció del procés: llm escollit, pre-processos, configuració i híper-paràmetres principals.
* Codi i requeriments previs  funcional i comentat
* Preguntes, respostes i GPTscore abans de l’ajustament fi.
* Preguntes, respostes i GPTscore després de l’entrenament.

### Notes:
* Podeu afegir altres mètriques a més del GPTscore si voleu.
* Tot dins d’un zip o rar.