# Konservativ H90-gennemgang af en eksisterende persona

Du får én syntetisk dansk personabeskrivelse skrevet af Mistral og én
kontrolleret ændring i et struktureret uddannelsesfelt. Bevar originalteksten.

Ændringen `education_level: not_stated` med H90 betyder, at kilden ikke har en
uddannelsesobservation for personen. Det betyder IKKE, at personen ikke har en
uddannelse. Ret kun tekst, der påstår en konkret uddannelse, grad eller afsluttet
uddannelsesniveau, som ikke længere understøttes af den strukturerede oplysning.

Svar udelukkende med JSON efter det angivne skema:

- `patched`: Brug kun denne afgørelse, når én lille, entydig og ordret
  tekstudskiftning kan fjerne en ikke-understøttet uddannelsesgrad eller et
  ikke-understøttet uddannelsesniveau. `old_excerpt` skal forekomme præcis én
  gang i teksten, og `new_excerpt` må kun fjerne eller neutralisere den
  uddannelsesoplysning. Bevar resten af Mistral-teksten.
- `unchanged_consistent`: Brug denne afgørelse, når teksten ikke omtaler
  uddannelse, eller når den kun omtaler skole, læring, arbejde eller erfaring på
  en måde, der ikke modsiger manglende kildeobservation. Brug `fact_not_stated`,
  hvis uddannelse ikke omtales konkret.
- `needs_manual_review`: Brug denne afgørelse, hvis teksten er tvetydig, hvis
  graden ikke kan fjernes med en unik minimal ordret patch, hvis mere end én
  rettelse kræves, eller hvis rettelsen ville kræve gæt. Vælg
  `ambiguous`, `multiple_edits`, `sensitive` eller `insufficient_evidence` som
  `manual_review_reason`.

Ret ikke navn, stil, erhverv, interesser, relationer, helbred, religion,
seksualitet, oprindelse, lokalitet eller andre fakta. Opfind aldrig en ny
uddannelse og skriv aldrig, at H90 er observeret individdata. Hvis du er i
tvivl, så afstå med `needs_manual_review`.
