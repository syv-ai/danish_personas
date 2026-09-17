# Danske personabeskrivelser

Du skriver syv korte, syntetiske personabeskrivelser på naturligt dansk ud fra faste
strukturerede input.

Returnér kun data, som passer til det krævede JSON-skema. Hver tekst skal være på
2-4 sætninger, konkret uden at være identificerende og indbyrdes forskellig.

Regler:

- Bevar demografi, personlighed og de allerede godkendte attributter.
- Opfind ikke navn, adresse, kontaktoplysninger, arbejdsplads eller institution.
- Nævn ikke CPR-numre, telefonnumre, e-mailadresser, links eller præcise steder.
- Udled ikke hudfarve, etnicitet, herkomst, nationalitet, kultur, sprog, religion,
  helbred, handicap, seksualitet, politisk overbevisning, kropsmål, udseende,
  tiltrækningskraft eller kriminalhistorik.
- `visual_persona` skal være 2-4 sætninger med generisk portrætvejledning om
  selvvalgt, foranderlig visuel fremtoning, eksempelvis tøj, tilbehør, farver eller
  pleje. Beskriv også et generisk miljø, der passer til et portræt. Det er ikke
  billedgenerering. Vælg ikke detaljer eller miljø ud fra demografi, OCEAN,
  genererede attributter eller et eventuelt fremtidigt felt om oprindelsesland.
- `visual_persona` må ikke nævne eller antyde hudfarve, etnicitet, herkomst,
  nationalitet, kultur, sprog, religion, helbred, handicap, seksualitet, politik,
  kropsmål, tiltrækningskraft, præcis placering, arbejdsgiver, institution eller
  andre følsomme eller identificerende egenskaber.
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
- Gengiv højst den angivne region; opfind ikke bystørrelse, lokale tilbud, rejsevaner
  eller andre stedsegenskaber.
- Brug naturlige danske ord frem for unødige engelske sammensætninger.
- `persona` skal samle helheden uden at gentage de fem specialiserede tekster ordret.
