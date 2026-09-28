### Abwesende:
Maik, Malte, Sarah (alle Urlaub)
# Was war letzte Woche
## Minh
Koordinaten werden richtig dargestellt jetzt beim Retargeting. Ball wird jetzt korrekt visualisiert, manchmal verschoben oder clippend, manchmal Hände verzögert. Neue Daten sind im Gitlab unter Retargeting/forfbx/archive.
## Niklas
Hat mitgeholfen beim Fixen der Daten für die Füße beim Retargeting
Hat geschaut ob Post-Processing Möglichkeiten für das Retargeting sinnvoll sind. Parabelberechnung für Ballwurf und eventuell Anpassung der Roboterbewegung dahingehend.
## Andy
Training mit den neuen Daten. Er zittert nicht und steht still. Testlauf mit unvollständigen Daten welche angepasst wurden. Debugbar durch gute visuelle Darstellung.
## Steffen, Yasin und Ivanna
Nachkalibrierung des MoCap Systems, Marker neu angebracht. Neue Aufnahmen erstellt. Mittlerweile sogar 60 Kameras (+4 Extra Kameras). Umknicken ist leider immer noch drin.
## Marius
Training von erster (IL) und zweiter (RL) Phase. Erste Phase scheint korrekt, trotz alter Daten. Zweite Phase hat kein funktionierendes Training, Debugging und Fixen von versteckten Fehlern. Überlegung ist, das alles in eine Phase zu mergen, da bei Änderungen der einen Phase meist die andere beeinträchtigt wird und dadurch Fehler entstehen.
## Simon
Bisher unrealistische Bewegungen des Roboters. Anpassung mit dem neuen Modell von Phillip. Natürliche Bewegung und Stillstehen ist möglich. Beste Version steht still und nur noch wenig unnatürliche Bewegung mit den Händen. Sterberate deutlich niedriger geworden. 
## Phillip
Aufnahmen zuende gemacht und danach IL ausprobiert. Training auf den Daten von letztem Montag. Er kann fast stehen. Repo: unitree_rl_mjlab.
## Yakub
Workstation war leider offline weil jemand die ausgesteckt hat. IL wurde gestartet. Behaviour Cloning. Nutzung von nem MLP um Bewegungsdaten zu predicten. Mit Physics noch oben drauf funktioniert das nicht. Idee: Actor Critic Model mit MLP Model als Warm-Start.
# Bis nächste Woche
## Yakub
Einlesen in IL
## Phillip
Weitermachen wie bisher und Aufnahmen verbessern
## Simon
Ist nächste Woche im Urlaub
## Steffen und Yasin
Klausur nächste Woche daher Fokus darauf
## Marius
Weitermachen mit Debugging bis positive Entwicklung im Training erkennbar
## Ivanna
Urlaub bis 4.9.2026
## Andy
Weiter Training mit den gefixten Daten, eventuell Modell austauschen
## Niklas
Post Processing von Retargeting Daten, eventuell mit dem Filter weiterarbeiten
## Minh
Neue Daten anschauen und Ballverzögerung und Überlappung verbessern.
# GENERELL
Zwischenbericht? Alle haben was zu tun daher machen wir den überhaupt?