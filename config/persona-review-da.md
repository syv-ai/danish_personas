# Konservativ gennemgang af en eksisterende persona

Du får én syntetisk dansk personabeskrivelse skrevet af en tidligere model og
kontrollerede ændringer i strukturerede felter. Bevar originalteksten. Kontrollér
hver ændring mod teksten uden at gætte på oplysninger, der ikke står der.

Svar udelukkende med JSON efter det angivne skema:

- `patched`: Teksten modsiger én eller flere nye oplysninger, og højst to helt
  små, præcise tekstudskiftninger kan rette modsætningen. Brug `patches` med
  `old_excerpt`, som forekommer præcis én gang ordret, og `new_excerpt`.
  Begge øvrige felter er tomme, og `manual_review_reason` er `null`.
- `unchanged_consistent`: Ingen ændret oplysning modsiges af teksten. Angiv ét
  bevis for HVERT ændret felt i `unchanged_evidence`. Brug `new_value_present`
  med et ordret citat, hvis den nye oplysning allerede står der; ellers brug
  `fact_not_stated` med tomt citat, KUN hvis teksten virkelig ikke omtaler
  feltet, heller ikke med synonymer, indirekte formuleringer eller relationer.
  Brug tom `patches` og `manual_review_reason=null`.
- `needs_manual_review`: Brug dette, hvis en sikker afgørelse er umulig, hvis
  teksten bruger tvetydige udtryk, hvis mere end to ændringer kræves, eller
  hvis oplysninger om følsomme identiteter ville være nødvendige. Vælg en af
  `ambiguous`, `multiple_edits`, `sensitive` eller `insufficient_evidence` som
  `manual_review_reason`. Begge lister skal være tomme.

En civilstandskategori fra Danmarks Statistik kan hedde `married_or_separated`,
men den beskriver ikke den enkelte persons præcise juridiske status. Den nye
`legal_status_detail` er et syntetisk valg: `married` betyder gift, og
`separated` betyder separeret. Sammenbland ikke juridisk status og aktuel
partnerrelation. Skriv aldrig, at den syntetiske fordeling er observeret data.

Bevar stil, sætningsbygning, navn og resten af den oprindelige Mistral-tekst.
Ret ikke andre fejl under dække af dette tjek. Opfind ikke erhverv, relationer,
helbred, religion, seksualitet, oprindelsesbaseret adfærd eller lignende.
Udled aldrig identitet eller kulturelle træk af køn, alder eller lokalitet.
Hvis fakta ikke er tilstrækkelige til en sikker mikrorettelse, vælg manuel
vurdering frem for at opfinde en løsning.
