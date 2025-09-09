# KI-Chat mit lokalen Dokumenten

## Dokumentsammlungen mit GPT4All verarbeiten
GPT4All (www.nomic.ai/gpt4all) bietet eine komfortable Oberfläche für die Nutzung mehrerer KI-Modelle. 

GPT4All installieren: Nach einem Klick auf „Download für Ubuntu“ laden Sie die Datei „gpt4all-installer-linux.run“ für Ubuntu oder Linux Mint herunter. Wenn Sie die run-Datei im Verzeichnis „Downloads“ gespeichert haben, führen Sie die folgenden sechs Befehlszeilen im Terminal aus:
```
sudo apt update 
sudo apt install libxcb-xinerama0 libxcb-cursor0
mkdir ~/Desktop
cd ~/Downloads
chmod +x ./gpt4all-installer-linux.run
./gpt4all-installer-linux.run
```
Folgen Sie den Anweisungen des Installationsassistenten. Nach Abschluss kopieren Sie den Programmstarter vom Ordner „Desktop“ nach „Schreibtisch“. Ubuntu-Nutzer gehen im Kontextmenü des Starters auf „Start erlauben“, damit er aktiv wird.

## Zusammenfassungen per Script automatisieren
### Vorbereitungen für die folgenden Scripts
Miniconda (www.anaconda.com) ist ein Tool, das eine unabhängige Python-Installation im eigenen Home-Verzeichnis bereitstellt.
Python-Projekte organisiert man in virtuellen Umgebungen, die bei Bedarf unterschiedliche Python-Versionen nutzen können. 

**1.** Installieren Sie Miniconda (https://www.anaconda.com) mit diesen sieben Zeilen:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
conda deactivate
```

**2.** Die virtuelle Umgebung für unsere Beispielscripte erstellen und aktivieren Sie mit
```
conda create -n summarize python=3.12
conda activate summarize
```
Dem Prompt im Terminal ist jetzt „(summarize)“ vorangestellt.

**3.** Ollama ist eine Plattform und ein Tool, mit dem Sie kostenlose Large-Language-Modelle (LLMs) auf dem eigenen Rechner nutzen. Für die Installation verwenden Sie im Terminal diese beiden Befehlszeilen:
```
wget https://ollama.com/install.sh
sh install.sh
```
Ollama richtet einen Systemdienst ein und startet automatisch im Hintergrund. Das KI-Modell laden Sie mit 
```
ollama pull llama3.2:latest
```
herunter und das Embedding-Modell mit 
```
ollama pull nomic-embed-text:latest
```
Beide Modelle sind im Script „summarize_texts.py“ vorkonfiguriert. Weitere Modelle finden Sie über https://ollama.com/search. 
Der Befehl 
```
ollama list
```
zeigt die installierten Modelle an. Mit
```
ollama rm [Modellname]
```
entfernen Sie ein Modell, wenn Sie es nicht mehr benötigen.

**4.** Erstellen Sie ein Arbeitsverzeichnis in Ihrem Home-Verzeichnis, beispielsweise „~/Scripts/summarize“, und kopieren Sie alle Dateien des Projekts  hinein.
Download über „Code -> Download ZIP“.

Wechseln Sie mit cd in dieses Verzeichnis und starten Sie 
```
pip install -r requirements.txt
```
Damit richten Sie alle erforderlichen Python-Module in der virtuellen Umgebung ein. 

## KI-Chat mit grafischer Oberfläche
Für einen Chat mit lokalen Dokumenten, also gezielte Fragen zum Inhalt, können Sie eine einfache Python-Oberfläche im Webbrowser verwenden. Unser Beispielscript „run_ollama_pdf_rag“ verwendet Streamlit (https://streamlit.io) für die Webanwendung. Der Start erfolgt in der virtuellen Python-Umgebung mit
```
python run_ollama_pdf_rag.py
```
## Dokument-Chat mit Deepseek

Für unser Beispielscript „run_chatpdf-rag-deepseek.py“ müssen die Modelle „deepseek-r1:latest“ und „nomic-embed-text:latest“ für Ollama installiert sein. Die Konfiguration des Scripts ist in der Datei „chatpdf_rag_deepseek/app/rag.py“ enthalten.
Starten Sie das Beispielscript in de virtuellen Umgebung mit
```
python run_chatpdf-rag-deepseek.py
```
Nach einem Klick auf „Browse files“ wählen Sie eine oder mehrere PDF-Dateien und stellen dann eine Frage.

