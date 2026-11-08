# Ricognizione bibliografica: UPOS tagging su Italian-ISDT

Ricerca consultata il 4 novembre 2026; indicazioni sul protocollo del progetto aggiornate l’8 novembre 2026. Ricognizione mirata di fonti primarie, non revisione sistematica esaustiva. Ambito: UPOS italiano su split ufficiali Italian-ISDT. Non aggregare in una classifica unica release diverse, accuracy su parole gold, F1 end-to-end e macro-F1.

## Risultato della ricerca

Le fonti consultate non permettono di certificare un unico vincitore SOTA attuale per questo preciso protocollo. Esistono risultati vicini al 99%, ma il numero più alto non basta: occorre documentare corpus, split, etichette e metrica. Il confronto del notebook deve includere sia riferimenti storici comparabili sia lavori recenti, dichiarandone le differenze.

## Fonti e risultati verificati

### 1. Tamburini (2020): riferimento per UD 2.5

Fabio Tamburini. *How “BERTology” Changed the State-of-the-Art also for Italian NLP*. CLiC-it 2020, Tabella 2.

| Modello | Accuracy UPOS (%) |
|---|---:|
| GilBERTo | 98,72 ± 0,03 |
| mBERT | 98,73 ± 0,05 |
| UmBERTo | 98,78 ± 0,08 |

Corpus dichiarato: UD-ISDT 2.5. Il lavoro riporta media e deviazione standard di almeno cinque run e adotta gli split già disponibili. È un riferimento storico su UD 2.5; il progetto attuale utilizza UD 2.15 e UDPipe 2 come confronto quantitativo principale. Non confondere questi valori con quelli su EVALITA 2007 o con XPOS.

[Articolo e metadati](https://aclanthology.org/2020.clicit-1.48/) · [PDF](https://aclanthology.org/2020.clicit-1.48.pdf)

### 2. Italian BERT/ELECTRA: stesso checkpoint del nostro esperimento

Il repository sperimentale di Stefan Schweter dichiara il commit ISDT `f20fa2b` e cinque run. Sul test: BERT italiano cased **98,84 ± 0,06**, BERT italiano XXL cased **98,90 ± 0,03**, mBERT cased **98,69 ± 0,07**, ELECTRA italiano XXL **98,88 ± 0,04**. Esiste una tabella dev separata.

Questi risultati documentano bene il riferimento originale. Prima di confrontarli direttamente con UD 2.5 occorre verificare l'identità dei file e il codice di valutazione: il commit non va automaticamente considerato equivalente alla release. È evidenza da repository degli autori, distinta da un articolo revisionato.

[Repository e tabelle](https://github.com/stefan-it/italian-bertelectra#pos-tagging)

### 3. UmBERTo: risultato alto, protocollo da chiarire

Il repository ufficiale di Parisi, Francia e Magnani (2020) riporta per UmBERTo Commoncrawl Cased su Italian-ISDT **98,977% accuracy** e **98,870 F1**. La tabella non identifica sufficientemente release e dettagli del protocollo per trattare il valore come direttamente confrontabile con il protocollo attuale su UD 2.15. Non assumere che l'F1 sia macro-F1. Inserire questo risultato come riferimento con comparabilità non accertata, senza usarlo per proclamare un vincitore.

[Repository ufficiale, Results](https://github.com/musixmatchresearch/umberto#results)

### 4. Grünewald, Friedrich e Kuhn (2021): STEPS e Trankit

*Applying Occam’s Razor to Transformer-Based Dependency Parsing: What Works, What Doesn’t, and What is Really Necessary*. IWPT 2021, Tabella 5.

Per Italian-ISDT la tabella riporta **98,81% UPOS accuracy** per STEPS MTL e **98,80%** per Trankit large. I risultati STEPS sono medie di tre run. Si tratta di sistemi che includono parsing e predizione di altre annotazioni: il claim SOTA principale del lavoro riguarda LAS. La release precisa e il protocollo della singola tabella vanno controllati prima di un confronto numerico diretto; qui non li assumiamo identici al notebook.

[Articolo](https://aclanthology.org/2021.iwpt-1.13/) · [Tabella 5 nel PDF](https://aclanthology.org/2021.iwpt-1.13.pdf)

### 5. Matsuda, Ma e Asahara (2025): confronto recente con LLM

*Step-by-step Instructions and a Simple Tabular Output Format Improve the Dependency Parsing Accuracy of LLMs*. IWPT/SyntaxFest 2025, Tabella 4, **UD 2.15**.

Per Italian-ISDT: UDPipe 2.0 **98,8**, Gemma-2-2B **98,6**, Qwen2.5-7B **98,4**, Llama-3.1-8B **98,7**, Gemma-2-9B monolingue **98,7**, Gemma-2-9B multilingue **98,8** nella colonna UPOS. I valori sono arrotondati a un decimale. Il claim principale riguarda parsing; non dimostra superiorità UPOS sui riferimenti precedenti. Il lavoro usa valutazione secondo le convenzioni UD, con gestione di output generati: non equiparare automaticamente tali punteggi alla nostra accuracy su tutte le parole gold.

[Articolo](https://aclanthology.org/2025.iwpt-1.2/) · [PDF](https://aclanthology.org/2025.iwpt-1.2.pdf)

### 6. UDPipe 2: riferimento documentato su release recenti

La documentazione ufficiale distingue esplicitamente input grezzo e tokenizzazione gold. Per `italian-isdt-ud-2.15-241121`, UPOS è **98,75** con tokenizzazione gold e **98,51** da testo grezzo. È un riferimento operativo utile, senza identificarlo automaticamente come SOTA. La differenza fra i due protocolli mostra perché non bisogna ordinare tutti i numeri nella stessa classifica. Nel progetto sono disponibili predizioni UDPipe 2 sul test ufficiale UD 2.15 con parole e confini di frase gold. Il notebook ne verifica modello, hash del test e allineamento e ricalcola accuracy e macro-F1 UPOS con le stesse convenzioni adottate per BERT.

[Documentazione ufficiale dei modelli](https://ufal.mff.cuni.cz/udpipe/2/models)

### 7. COMBO su UD 2.17: evidenza recente da model card

La scheda di `clarin-pl/combo-nlp-xlm-roberta-base-italian-isdt-ud2.17` riporta UPOS **97,95 F1 end-to-end** e **98,81 aligned accuracy**. La seconda è calcolata sui token correttamente segmentati, non su tutte le parole gold. Non è quindi la stessa misura del notebook. La release dichiarata è UD 2.17; questa è una model card degli autori, non una nuova dimostrazione di primato tramite articolo.

[Model card](https://huggingface.co/clarin-pl/combo-nlp-xlm-roberta-base-italian-isdt-ud2.17)

### 8. BERTino: riferimento per efficienza

La model card di BERTino riporta **0,9801 F1** su Italian-ISDT, contro **0,983** del teacher. È pertinente per discutere costo e dimensioni dei modelli, ma non come accuracy UPOS direttamente confrontabile: occorre verificare release e definizione dell'F1. Le versioni della pubblicazione e della model card riportano numeri leggermente diversi; citare sempre la fonte precisa del valore scelto.

[Model card degli autori](https://huggingface.co/indigo-ai/BERTino) · [Preprint](https://arxiv.org/abs/2303.18121)

## Protocollo adottato nel progetto

Il notebook utilizza gli split ufficiali di UD Italian-ISDT 2.15 e assume segmentazione gold. Confronta tre architetture: BERT congelato con classificazione lineare, BERT fine-tuned con classificazione lineare e BERT fine-tuned con CRF sulla sola sequenza UPOS.

Ogni architettura viene addestrata in due fasi: prima sul solo compito UPOS, poi su UPOS e FEATS a partire dal checkpoint UPOS selezionato per la stessa architettura e lo stesso seed. Nella seconda fase vengono aggiunte 14 teste lineari FEATS, mantenendo attiva la loss UPOS. Sono previste sei configurazioni, ciascuna con i seed 42, 123 e 456: 18 prove complessive, con cinque epoche per fase.

In entrambe le fasi il checkpoint è selezionato mediante macro-F1 UPOS sul development set. La valutazione finale riporta accuracy e macro-F1 UPOS su tutti i tag e sui tag presenti; per i modelli della seconda fase aggiunge accuracy e macro-F1 FEATS e accuracy congiunta. I risultati sono riassunti con media e deviazione standard sui seed e variazioni UPOS appaiate tra le fasi. L’interpretazione di queste variazioni considera anche l’addestramento aggiuntivo della seconda fase.

UDPipe 2, modello `italian-isdt-ud-2.15-241121`, è il riferimento esterno sul medesimo test gold per UPOS. I lavori su altre release e i sistemi con protocolli diversi rimangono riferimenti contestuali, con le rispettive condizioni di comparabilità. I risultati delle precedenti campagne sperimentali sono distinti da quelli del protocollo attuale.

## Limiti della ricognizione

Ricerca mirata su ACL Anthology, repository degli autori, documentazione ÚFAL e model card ufficiali. Query su Italian-ISDT, UPOS, PoS, BERT, STEPS, Trankit e UDPipe; inclusi risultati recenti e precedenti direttamente pertinenti. Non è stato trovato nelle fonti consultate un leaderboard unico, aggiornato e omogeneo che certifichi il miglior UPOS tagger attuale. Questa assenza nella ricerca non dimostra che un tale risultato non esista. Nessun sistema è stato eseguito per questa ricognizione.
