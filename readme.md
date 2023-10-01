# Code für BWKI 2023
Hier befindet sich aller Code, den ich für das Projekt Brainhome 1.0 bis jetzt entwickelt habe.

## 1. KI (Ordner KI)
Hier befindet sich der Code für:
- den MaxViT + Code zum Trainieren (in MaxViT/MaxDaVit, trainierte Parameter: MaxVit11.pt)
- Data-Processing (in Dataset/Dataset.py)
- ein Programm welches die Inference simuliert (in Inference.py)
- Zwei Test-Notebooks um diese beiden zu testen (in Testing) 
    - diese befinden sich eigentlich woanders -->funktionieren hier nicht, aber man kann die Eingaben und Ausgaben sehen

## 2. Arduino (Ordner Arduino)
Hier befindet sich der Code für:
- die Kommunikation zwischen Analog-zu-Digital-Wandler und Arduino (in ReadMCP3903)
- die ersten Versuche der Bluetooth-Kommunikation (in Bluetooth)

## 3. WPF-App
Diese befindet sich in einem anderen Repository: https://github.com/jonathanreinhart/EEGVis-V2

## 4. Hardware-Dateien:
Web-Projektübersicht: https://oshwlab.com/jonathanreinhart81/eeg-prefinal
Web-Editor-Ansicht: https://easyeda.com/editor#project_id=8a6bf672870946c99ba875b9d5694126
(Die Ansicht hat hier teilweise kleine Bugs: Silk layer unter anderen etc.)

## 5. Abbildungen (Ordner Abbildungen):
Beschreibung:

EEGOverView: Allgemeiner Überblick wie er auch im Pitch gezeigt wurde.

BeispielDaten: 5s von Daten (und die Frequenzen) die von der selbstenwickelten Hardware aufgenommen wurden (oben ungefiltert, unten gefiltert)

Boardx: Prototypen-Boards, bei 3 gibt es ein ADC-Baord (groß), und ein reines Verstärkungs-Board (klein)

Elektrode: die eigenentwickelte Elektrode

EEGKappe: die eigenentwickelte EEG Kappe

ConfMatrix/W: Einmal die ConfMatrix bei der Klassifikation auf ein Fenster, und auf 15 Fenster

PrecisionRecall/W: '' Precision-Recall-Curve ''

## 6. App für Handy (noch nicht fertig)
Diese befindet sich in einem anderen Repository: https://github.com/jonathanreinhart/brainhome


## Quellen
Vergleichspaper: https://doi.org/10.1016/j.eswa.2021.115968
MaxVit: https://arxiv.org/abs/2204.01697
Datensatz: Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). 
PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. 
Circulation [Online]. 101 (23), pp. e215–e220.