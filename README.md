# PoS tagging italiano con BERT

Il progetto adatta BERT italiano alla predizione UPOS e ne valuta le prestazioni sul benchmark standard UD Italian-ISDT 2.15, con split ufficiali e segmentazione gold. Il confronto con UDPipe 2 sullo stesso test e con la letteratura è centrale; encoder congelato e CRF supportano l'analisi, mentre FEATS è un'estensione morfologica.

## File da consultare

| File o cartella | Contenuto |
|---|---|
| [`report/report.tex`](report/report.tex) | Sorgente canonico della relazione, con bibliografia in `report/references.bib`. |
| [`report/report.pdf`](report/report.pdf) | Relazione compilata. |
| [`report/Report_BERT_UPOS_LaTeX.zip`](report/Report_BERT_UPOS_LaTeX.zip) | Sorgenti LaTeX, bibliografia e immagine del frontespizio; non contiene codice, dati o risultati sperimentali. |
| [`BERT_Italian_PoS_UD_2_15.ipynb`](BERT_Italian_PoS_UD_2_15.ipynb) | Pipeline di addestramento e valutazione BERT, con confronto UPOS esterno. |
| [`Ricognizione_bibliografica_UPOS.md`](Ricognizione_bibliografica_UPOS.md) | Fonti esterne e condizioni di comparabilità. |
| [`VERIFICA_FINALE.md`](VERIFICA_FINALE.md) | Esito della revisione scientifica, numerica e tecnica del 9 novembre 2026. |
| `outputs/bert_upos_then_feats_v2/` | Riepiloghi finali: `results.csv`, `comparison_mean_std.csv`, `comparison.csv`, variazioni appaiate e analisi degli errori. |
| `outputs/references/udpipe_2_15/` | Predizioni e risposte UDPipe salvate, metadati del riferimento e confronto morfologico ricalcolato. |

Il protocollo finale comprende **18 prove**, tre architetture per due fasi e tre seed (42, 123, 456). I riepiloghi `v2` riuniscono nove prove UPOS riutilizzate da `outputs/bert_upos_then_feats_v1/` e nove prove congiunte UPOS+FEATS conservate in `v2`. La precedente fase congiunta `v1` è preliminare e non entra nel confronto finale. Ciascuna fase dura cinque epoche; la seconda parte dal checkpoint UPOS della stessa architettura e dello stesso seed.

## Verifica offline del confronto morfologico

Per rileggere il progetto bastano relazione e risultati salvati. Per ricalcolare il confronto FEATS e UPOS+FEATS, dalla cartella `natural-language-processing/` eseguire:

```console
python -B evaluate_udpipe_morphology.py
```

Lo [script](evaluate_udpipe_morphology.py) usa soltanto la libreria standard Python e richiede i dati locali `data/UD_Italian-ISDT-r2.15/`, i metadati e le predizioni dei nove modelli congiunti `v2` e le risposte UDPipe memorizzate in `outputs/references/udpipe_2_15/`. Non scarica dati e non esegue training, inferenza o lettura dei pesi. Se si trasferisce il progetto, occorre includere queste cache; lo ZIP LaTeX serve soltanto alla relazione.

Aggiorna tre file nella cartella del riferimento:

- [`morphology_comparison.csv`](outputs/references/udpipe_2_15/morphology_comparison.csv): confronto aggregato, in percentuale;
- [`morphology_per_feature_metrics.csv`](outputs/references/udpipe_2_15/morphology_per_feature_metrics.csv): metriche per coppia FEATS, in frazioni;
- [`morphology_evaluation.json`](outputs/references/udpipe_2_15/morphology_evaluation.json): metriche in frazioni, verifiche e provenienza.

La riga UDPipe del distinto file `outputs/bert_upos_then_feats_v2/comparison.csv` riguarda soltanto UPOS. Il confronto morfologico usa insiemi FEATS completi e accuracy congiunta UPOS+FEATS, con le stesse convenzioni di BERT; i punteggi ufficiali UDPipe `UFeats` e `AllTags` hanno definizioni diverse.

## Esecuzione del notebook

Aprire il notebook dalla radice del repository oppure dalla cartella `natural-language-processing/` ed eseguire le celle in ordine. Il notebook riutilizza le prove complete compatibili, addestra quelle mancanti e rivaluta i modelli sul test. Questa esecuzione completa non è necessaria per revisionare il testo o verificare le metriche dalle predizioni salvate.

Le revisioni, gli hash dei dati e le versioni effettive dell'ambiente sono registrati negli `experiment.json` delle cartelle `v1` e `v2`. Il riuso delle cache verifica rigorosamente anche tali versioni: la cella di installazione non garantisce da sola un ambiente identico. Per riutilizzare le prove esistenti occorre ripristinare l'ambiente registrato; eventuali nuove esecuzioni con impostazioni o ambiente diversi richiedono cartelle di output distinte. I metadati storici documentano le prove già svolte.


## Spazio e conservazione degli output

Dopo la pulizia restano i 18 modelli selezionati nelle rispettive cartelle `best_model`, con predizioni, metriche, cronologie e metadati. Le copie intermedie e gli stati degli ottimizzatori sono stati eliminati; per inferenza e valutazione utilizzare `best_model`. Dettagli in [STORAGE_AUDIT.md](STORAGE_AUDIT.md).
