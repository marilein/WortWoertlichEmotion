# WortWoertlichEmotion
Wort wörtlich Emotion ist ein Perzeptionsexperiment, das als Web-Experiment mit Hilfe [Percy](https://webapp.phonetik.uni-muenchen.de/WebExperiment/index.html) durchgeführt wurde. Diese Repository dient zur Vorbereitung der Annotationen und zur Datenanalyse. 


*************************************************
## Inhalt der Repository

************************************************


### Skripte

* normalize_raw_data.py :

  Die Normalisierung der Rohdaten wird von diesem Skript übernommen. Die Annotationen der Teilnehmer werden analysiert, geprüft und unter "normalized_data"   gespechert
  
* data_analysis.py :

Datenanalyse und Statistiken (Chi-Quadtrat-Test, Fischer's Test...)

* data_processor.py :

Allgeimeine Funktionen die für die Datenvorbereitung soweie Datenverarbeitung von unterschiedlichen Skripten in diesem Projekt benutzt werden.

* experiment_items.py : 

Das Skript extrahiert die eundeutigen IDs für die Eingabe-Items für jedes Experiment.

* experiment_overview.py :

Hier werden Metadaten als Überblick zusammengefasst. Außerdem werden Aggregationstabellen erstellt.

* Annotator_performance.py : 

Rater-Reliabilität

* Plotter.py : 

Dieses Modul wird benutzt um Diagramme zu erstellen und Datenanalyse zu visualisieren.


### Ordner

aggregation : Aggregationstabellen, wie Mehrheitsquote per Stimulus oder Trefferquote für jeden prosodischen Ausdruck per Kultur. Eine weitere Tabelle beinhaltet die durschnittliche Grundfrequenz der Sprecher für alle originalsprachsignale, die in diesem Experiment als Stimuli benutzt wurden.
analyze : Bei der Datenanalyse wurden die Annotationsdaten je nach Ziel der Analyse zerlegt, gruppiert oder sortiert. Das dient als Datenvorbereitung für die statistischen Tests
answer_hits : Die Tabellen in diesem Ordner beinhalten die Trefferquote für die intendierten prosodischen Ausdrücke.
confusion_matrices : Jede Datei beinhaltet eine Konfusionsmatrix.
item_mapping : Mapping-Listen für Eingabeoptionen der armenisch und der deutschen Version des Experiments.
normalized_data : Die Exportierten Annotationen, die durch das Web-Experiment erhoben wurden, kommen als Rohdaten in einer Datei. Nach der Qualitätsprüfung werden die Daten der erfolgreichen Teilnahmen für die weitere Datenverarbeitung vorbereitet und in diesem Ordner gespeichert. 
normalized_data_with_expected_labels : Hier befindet sich eine Ergänzung der normalisierten Annotationsdaten. Der Unterschied besteht darin, dass diese Daten zusätzlich die Labels für die intendierten prosodischen Ausdrücke beinhalten. 
overview : Übersichten der Bewertungen, der Teilnahmen oder der Teilnehmerdaten.
plots : Die durch Skripte erstellte Diagramme werden in diesem Ordner gespeichert.
raw_data : Rohdaten der Annotationen
results : Die Ergebnistabellen für statistischen Tests
