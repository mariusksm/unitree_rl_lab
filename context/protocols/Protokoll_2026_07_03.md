# Protokoll vom 03.07.2026 erstellt von Andy

## Fehlen im Präsenz:
 - Niklas

## Generelles
Jan meint, dass wir eventuell eine 4080 Workstation gestellt bekommen mit der wir dann drauf arbeoten können

## Diese Woche Erledigt:

Yakup:
- gearbeitet am Tiefpassfilter, damit das Zittern rausgeht
- script für umformen von fbx zu bvh, da nur bvh nimmt
- richtigen Roboter mit den Händen für Minhs Retargeter erstellt

Phillip:
- Aufnahmen nicht optimal
- Script hat das Hand umknicken verhindert, aber mehr Probleme gefunden, e.g. Hüfte sind unterschiedlich voneinander entfernt, das muss ausgefiltert werden
- Heute implementiert, dass sich die Arme nicht mehr überkreuzen.
- Script kalibriert, sodass man gute Ergebnisse herauskriegt, das auch soweit automatisiert, sodass es auf den Ordner mit Dateien den Script anwendet und dann den Script von Yakup anwendet

Simon:
- Rewards implementiert in Gitlab, versucht zu trainieren nach 16h bemerkt, das es nicht gut ist, 300 steps in der szene also 0,6 sekunden nur, lässt sich noch finetunen und rewards an passen, ist nur im Fallen in den 0,6, muss jetzt trainieren, das er einfach steht. 


Steffen & Ivanna & Yasin:
- Aufnahmen gemacht, länger als vorherige 4-6 sec, fbx Version scheinbar noch buggy, bei live und in post sieht eigentlich gut aus
- müssen jetzt anschauen warum das buggy ist und fixen, in Post stichpunktprobenhaft eigentlich sehen gut aus

Maik:
- Hat Modell trainiert, dass es einen Schritt zur Seite geht
- Rewardsfunktion angepasst, ausprobiert mit MoCap Daten IL anzuwenden, trainiert mit dem Plan nur mit der Fangbewegung, beim Recapturing hängt es fest, bei den Daten als auch bei den Anforderungen in Gitlab
- Absprache mit Minh und Marius dazu

Marius:
- Weitergearbeitet an der Repo
- Pipeline fertig gemacht und durch gedebuggt
- aber hat Nvidea error, so kann man verifizieren dass man es trainiert wie man möchte, aber kann es nicht visualisieren

Sarah:
- Training mit weniger DoF, wo Hände nicht mit drin sind
- Für Algorithmus braucht es Sites, rote Boxen im Bild, hat zu keinem guten Ergebnis geführt, eventuell anderen Algorithmus probieren.
- neue Dateien retargeted mit Ball versucht reinzukriegen. 
- Bitte darum, die Dateien nach Humanoid und Ball entsprechend zu benennen

Andy:
- Konvertierungsscript angepasst .pkl zu .npz von 23 DoF zu 53 DoF, um die Hände mitabzubilden
- konvertierte Dateien verwendet für IL in LocoMujoco aber musste Parameter anpassen, sodass es auf RTX 3060 mit 12 GB VRAM läuft; Trainieren lassen für 2h
- keine guten Ergebnisse, muss Script anpassen, länger trainieren lassen

Malte:
- Script geschrieben, das selfcollisions erkennt, gibt aus wenn es selfcollision findet, findet den Zeitraum an Frames, wo es stattfindet, nutzt Mujoco API

Minh:
- Aufnahmen vom 26.06 alle retargeted
    - man sieht linke Hand ist unmenschlich
    - von 230 Dateien sind ca. 100 gemerged und gefiltert, ca. 20-30% sind erfolgreich, bei vielen sieht man, dass der Roboter collided in sich selber, die Hände sind größer und gelb, übernommen von Yakup; 
- linken Arm schon angefangen zu fixen, muss sich noch anschauen; 
- Jan meint hierzu: einen Schritt zurückgehen und Skelett anschauen
    - eigentlich sollten Selfcollisions nicht in GMR geschehen, da diese in IK verhindert werden sollten; 

## Pläne für Nächste Woche:

Yakup:
- switcht zu Eventkamera mit dem Filter, ob das gut passt, ob man die Tiefe gut bestimmen kann.  

Phillip:
- Zusammensetzen mit MoCap, sodass man sich das anschaut am Montag, für gute Daten
-  nebenbei das Script anpassen.

Simon:
- Weiter RL das Fangen antrainieren

Steffen & Ivanna & Yasin:
- Steffen im nächste Woche Urlaub 
- größeren Ball kaufen einer und Rechnung Jan geben, um Ball besser von unten fangen zu können. 
- Schauen warum FBX buggy ist und das Fixen, Hüftgelenke und Hände Joints

Maik:
- Weiter recapturing der MoCap daten Mimic und IL

Marius:
- Fehlerdaten fixen und kurzschließen mit Maik und Minh, 
- das in Mimic reinbekommen
- Rewards von Simon angucken, einige andere Ideen anschauen. 
- in zwei Wochen einmal per Discord joinen

Sarah:
- Versuchen Ball in IL reinkriegen
- andere LocoMujoco Algorithmen ausprobieren, schauen ob die besser sind
- nutzt eventuell 4080 von Fraunhofer.

Andy:
- Script anpassen, sodass Fingerbewegungen auch abgebildet werden
- Originaldaten anschauen und mit umgewandelten Vergleichen
- Länger trainieren lassen

Malte:
- Script auf das neue Modell von Yakup umstellen und das Script auch einen batch von Dateien sortiert in parse und nicht parse.

Minh:
- Am Problem im Retargeten anschauen 
- neue Daten anschauen, wenn hochgeladen.
- Finger untersuchen, ob diese retargeted werden.
