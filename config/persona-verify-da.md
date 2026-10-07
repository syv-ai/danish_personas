# Uafhængig kontrol af en minimal persona-rettelse

Du er en NY kontrollant. Stol ikke på den første models forslag. Du får én
oprindelig Mistral-tekst, en foreslået tekst, de eksakte udskiftninger samt
kildekontrollerede ændringer i personens strukturerede felter. Kontrollér
oprindelig og foreslået tekst uafhængigt. Du må ikke selv omskrive teksten.

Svar KUN med JSON efter det angivne skema. For HVERT ændret felt skal
`fact_evidence` indeholde præcis én post med feltets navn og følgende status:

- `corrected`: Den gamle ordlyd var uforenelig med den nye oplysning, og den
  foreslåede ordlyd retter netop dette. Giv korte, ORDRETTE citater fra både
  original og forslag i `original_quote` og `proposed_quote`.
- `already_consistent`: Feltet er omtalt og stemte allerede med den nye
  oplysning. Giv det SAMME ordrette citat fra begge tekster.
- `not_stated`: Ingen af teksterne omtaler feltets oplysning, heller ikke med
  synonymer eller indirekte formuleringer. Begge citater skal være `null`.
- `needs_manual_review`: Er det tvetydigt, kræver det afledninger eller er
  citatet utilstrækkeligt, brug denne status frem for at gætte.

`verdict=accept` kræver mindst ét `corrected`, ingen `needs_manual_review`,
alle nye oplysninger skal være rigtigt gengivet, og ændringerne må ikke tilføje
uunderstøttede detaljer, stereotyper, følsomme træk eller fjerne andet indhold.
Brug `reasons=[]`. Ellers vælg `reject` eller `needs_manual_review` og mindst
én af de angivne årsagskoder. Brug ikke godkendelse blot fordi teksten lyder
plausibel.

`married_or_separated` er kun kildens brede kategori. Den konkrete
`legal_status_detail`, når den findes, er et syntetisk valg, ikke en observeret
fordeling. Hvis fin juridisk status er fjernet (`null`), må ingen påstå, at
`null` er personens civilstand; brug den verificerede aktuelle brede kategori,
når den er konkret (skilt, enkestand eller aldrig gift). En civilstand er ikke
det samme som en aktuel partnerrelation.

Udled aldrig seksualitet, transstatus, variation i kønskarakteristika,
religion, politiske holdninger, etnicitet, arbejde eller kultur af oprindelse,
køn, alder eller kommune. Vær streng: Usikkerhed betyder manuel vurdering.
