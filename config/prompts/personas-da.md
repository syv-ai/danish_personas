# Danske personabeskrivelser

Du skriver syv korte, syntetiske personabeskrivelser på naturligt dansk ud fra faste
strukturerede input.

Returnér kun data, som passer til det krævede JSON-skema. Hver tekst skal være på
2-4 sætninger, konkret uden at være identificerende og indbyrdes forskellig.

Regler:

- Bevar demografi, personlighed og de allerede godkendte attributter i de
  relevante beskrivelser, men gengiv dem ikke i `visual_persona`.
- Opfind ikke navn, adresse, kontaktoplysninger, arbejdsplads eller institution.
- Nævn ikke CPR-numre, telefonnumre, e-mailadresser, links eller præcise steder.
- Udled ikke hudfarve, etnicitet, herkomst, nationalitet, kultur, sprog, religion,
  helbred, handicap, seksualitet, politisk overbevisning, kropsmål, udseende,
  tiltrækningskraft eller kriminalhistorik.
- `visual_persona` skal følge denne lukkede danske form og være 2-4 sætninger:
  `Personen vælger [farvet tøj] og [farvet tilbehør]. Baggrunden er [baggrund].`
  Du kan derefter tilføje `Lyset er [lys].` og `Rammen er neutral.` i netop den
  rækkefølge. Brug kun ét tilladt valg fra hver liste i
  `docs/visual-persona-format.md`, og brug farvens korrekte danske bøjning.
- Tøj er kun: bluse, cardigan, frakke, jakke, kjole, nederdel, skjorte, sweater,
  trøje, vest, halstørklæde eller tørklæde. Tilbehør er kun: broche, halskæde,
  hat, kasket, paraply, taske, armbånd, bælte, sjal, slips, tørklæde eller ur.
  Farver er kun blå, brun, grå, grøn, hvid, lilla, orange, pink, rød, sort,
  turkis eller gul.
- Baggrunden er kun en afdæmpet, enkel, ensfarvet, lys, neutral eller rolig flade
  eller et afdæmpet, enkelt, lyst eller roligt atelier/studie. Tilladt lys er
  kun blødt, klart, dæmpet, diffust, jævnt eller roligt.
- Skriv ingen andre ord i `visual_persona` end formatets faste ord og de
  allow-listede valg. Den kontrollerede form udelukker derfor nationalitet,
  sprog, religion, alder, køn, sex, region, uddannelse, civilstand,
  arbejdsmarkedsstatus, OCEAN, hår, øjne, hud, ansigt, krop, fysiske træk,
  følsomme egenskaber, steder og institutioner. Det er ikke billedgenerering.
- Undgå stereotyper og deterministiske forbindelser mellem demografi og personlighed.
- Behandl OCEAN-scorer som svage tilbøjeligheder, ikke som fakta om evner, problemer
  eller begrænsninger. Brug afbalancerede formuleringer som "kan foretrække" og
  "har muligvis lettere ved", og beskriv aldrig en lav eller høj score som en mangel.
- Undlad at gengive scoreetiketter eller psykologiske vurderinger i teksten.
- Hvis et domæne ikke passer naturligt, skal teksten sige det neutralt frem for at
  opfinde ekspertise.
- Hold hver specialiseret tekst til sit eget domæne; sportsafsnittet må eksempelvis
  ikke fyldes med kunstinteresser.
- Omtal syntetiske færdigheder som interesser eller mulige styrker, ikke som
  dokumenteret erfaring eller konkrete arbejdsopgaver.
- Gengiv højst den angivne region i de relevante ikke-visuelle beskrivelser; opfind
  ikke bystørrelse, lokale tilbud, rejsevaner eller andre stedsegenskaber. Nævn ingen
  region eller placering i `visual_persona`.
- Brug naturlige danske ord frem for unødige engelske sammensætninger.
- `persona` skal samle helheden uden at gentage de fem specialiserede tekster ordret.
